"""Orchestrate the collection of web-pages."""
import shutil
import logging
import asyncio
import contextlib
from pathlib import Path
from asyncio import Queue
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, List

import numpy as np

LOGGER = logging.getLogger(__name__)


class TaggingLogger(logging.LoggerAdapter):
    """Adds a tag to log messages."""
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['tag'], msg), kwargs


#: The function to run  to perform the collection
#: It must accept the input_path, output_dir, region_id, and client_id and
#: return True if the collection was successful, false otherwise.
TargetFn = Callable[[Path, Path, int, int], bool]


class TooManyFailuresException(RuntimeError):
    """Raised if the progress tracker reports too many failures."""


class Collector:
    """Performs collections for a single URL."""
    def __init__(
        self,
        target: TargetFn,
        input_file: Path,
        output_dir: Path,
        region_queues: Tuple[Queue, ...],
        *,
        delay: float = 0,
    ):
        self.target = target
        self.region_queues = region_queues
        self.input_file = input_file
        self.output_dir = output_dir
        self.delay = delay
        self.log = TaggingLogger(LOGGER, {"tag": "Collector-??"})

        self._init_directories()

    @property
    def n_regions(self) -> int:
        """Return the number of regions."""
        return len(self.region_queues)

    def _init_directories(self):
        for i in range(len(self.region_queues)):
            region_dir = self.output_dir / f"region_id~{i}"
            region_dir.mkdir(exist_ok=True)
            (region_dir / "status~success").mkdir(exist_ok=True)
            (region_dir / "status~failure").mkdir(exist_ok=True)

    def _check_for_prior_samples(self, progress):
        """Checks for prior successful runs."""
        for i in range(self.n_regions):
            success_dir = (self.output_dir / f"region_id~{i}/status~success/")
            n_successes = sum(1 for _ in success_dir.glob("run~*"))
            self.log.info("Found %d successes for region %d", n_successes, i)
            progress.completed[i] = n_successes

        if np.sum(progress.completed) == 0:
            # No successes, so all failures were sequential
            n_failures = sum(
                1 for _ in
                self.output_dir.glob("region_id~*/status~failure/**/run~*")
            )
            self.log.info("Found %d sequential failures", n_failures)
            progress.overall_failures = n_failures

    async def run(self, n_instances: int, max_failures: int = 3):
        """Run collection until there are at least n_instances collected
        across the various regions.
        """
        self.log.info(
            "Running collection of %d samples with max %d failure(s).",
            n_instances, max_failures
        )
        progress = ProgressTracker(
            n_instances=n_instances, n_regions=len(self.region_queues),
            max_failures=max_failures,
        )
        self._check_for_prior_samples(progress)

        if progress.is_complete():
            self.log.info("Already complete, doing nothing.")
            return
        if progress.has_too_many_failures():
            self.log.info("Collection already has too many failures.")
            raise TooManyFailuresException()

        region_tasks = [
            asyncio.create_task(self.collect_region(i, progress))
            for i in range(self.n_regions)
        ]

        try:
            # No point storing the resulting task since if there is a failure
            # we cannot use the task to cancel the rest as it would have
            # completed due to the failure.
            await asyncio.gather(*region_tasks)
        except Exception:
            # If one of the region collections fails then cancel all regions.
            for task in (t for t in region_tasks if not t.done()):
                with contextlib.suppress(asyncio.CancelledError):
                    self.log.warning("Cancelling task %s", task.get_name())
                    task.cancel()
                    # Wait for the cancelled error to raise in the task, will
                    # be as soon as it is scheduled so no other exception
                    # should be raised
                    await task
            # Reraise the TooManyFailuresException to let the caller know there
            # was a failure
            raise

    async def collect_region(self, region_id: int, progress):
        """Collect the samples for the specified region_id."""
        logger = TaggingLogger(LOGGER, {"tag": f"Region({region_id})"})
        loop = asyncio.get_running_loop()

        region_dir = self.output_dir / f"region_id~{region_id}"
        run_id = _get_initial_run_id(region_dir)
        logger.info("Starting from run_id %d", run_id)

        while not progress.is_complete(region_id):
            output_dir = region_dir / f"run~{run_id}"
            output_dir.mkdir(exist_ok=False)

            logger.debug("Locking client ...")
            async with self.get_region_client(region_id) as client_id:
                logger.debug("Starting run %d on client %d", run_id, client_id)
                is_success = await loop.run_in_executor(
                    None, self.target, self.input_file, output_dir, region_id,
                    client_id
                )
            if is_success:
                logger.debug("Run successful: %d", run_id)
                shutil.move(str(output_dir), region_dir / "status~success")
                progress.success(region_id)
            else:
                logger.debug("Run failed: %d", run_id)
                shutil.move(str(output_dir), region_dir / "status~failure")
                progress.failure(region_id)

                if progress.has_too_many_failures():
                    raise TooManyFailuresException()

            logger.debug("Released client: %d", client_id)

            run_id += 1
            await asyncio.sleep(self.delay)
        logger.info("Completed collection for region.")

    @contextlib.asynccontextmanager
    async def get_region_client(self, region_id: int):
        """Get the client_id of an available client for the specified
        region.
        """
        client_id = await self.region_queues[region_id].get()
        try:
            yield client_id
        finally:
            self.region_queues[region_id].put_nowait(client_id)


def _get_initial_run_id(path: Path) -> int:
    run_files = (str(filename) for filename in path.glob("**/run~*/"))
    return max(
        (int(filename.rsplit("~", maxsplit=1)[-1]) for filename in run_files),
        default=-1
    ) + 1


@dataclass
class ProgressTracker:
    """Track the progress of the collection of a web-page."""
    #: The number of instances to collect
    n_instances: int
    #: The total number of regions
    n_regions: int = 1
    #: The maximum number of failures
    max_failures: int = 3
    #: If use_region is non-negative, then all of the instances will be
    #: collected via a single region.
    use_only_region: int = -1

    #: The number of successes completed in the region
    completed: Optional[np.ndarray] = field(init=False, default=None)
    #: The total number of sequential failures across all regions
    overall_failures: int = field(init=False, default=0)
    #: The number of sequential failures within a region
    failures: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self):
        if self.failures is None:
            self.failures = np.zeros(self.n_regions, int)
        assert self.failures.shape == (self.n_regions, )

        if self.completed is None:
            self.completed = np.zeros(self.n_regions, int)
        assert self.completed.shape == (self.n_regions, )

    def has_too_many_failures(self) -> bool:
        """Return True iff the number of sequential failures has
        hit the maximum.
        """
        return self.sequential_failures() >= self.max_failures

    def use_all_regions(self):
        """Reset the use_only_region variable."""
        self.use_only_region = -1

    def remaining_regions(self) -> List[int]:
        """Return the regions remaining to be collected."""
        return [region_id for region_id in range(self.n_regions)
                if not self.is_complete(region_id)]

    def is_complete(self, region_id: int = -1) -> bool:
        """Return true iff the collection of this sample is complete."""
        regions = [region_id] if region_id >= 0 else list(range(self.n_regions))
        return all(self.remaining(region_id) == 0 for region_id in regions)

    def required(self, region_id: int) -> int:
        """Return the amount required for the region."""
        if self.use_only_region >= 0:
            if region_id != self.use_only_region:
                return 0
            return self.n_instances
        return self.n_instances // self.n_regions + (
            1 if region_id < (self.n_instances % self.n_regions) else 0)

    def remaining(self, region_id: int) -> int:
        """Return the number of samples remaining for the region."""
        assert self.completed is not None
        return max(0, self.required(region_id) - self.completed[region_id])

    def sequential_failures(self) -> int:
        """Return the number of sequential failures either overall, or
        within any region, whichever is higher.
        """
        assert self.failures is not None
        return max(self.overall_failures, max(self.failures))

    def failure(self, region_id: int):
        """Mark that a task for the specified region has failed."""
        assert self.failures is not None
        self.overall_failures += 1
        self.failures[region_id] += 1

    def success(self, region_id: int):
        """Mark that a task for the specified region has succeeded."""
        assert self.failures is not None and self.completed is not None
        self.completed[region_id] += 1
        self.failures[region_id] = 0
        self.overall_failures = 0
