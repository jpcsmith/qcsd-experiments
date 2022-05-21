"""Orchestrate the collection of web-pages."""
import shutil
import logging
import asyncio
import contextlib
from pathlib import Path
from asyncio import Queue
from itertools import islice
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, List, Dict

import numpy as np

LOGGER = logging.getLogger(__name__)
DEFAULT_DELAY: float = 30.0
DEFAULT_INTERVAL: float = 30.0


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
    def __init__(self, n_success: int, n_failure: int):
        super().__init__()
        self.n_success = n_success
        self.n_failure = n_failure


class Collector:
    def __init__(
        self,
        target: TargetFn,
        input_dir: str,
        output_dir: str,
        *,
        n_regions: int,
        n_clients_per_region: int,
        n_monitored: int,
        n_instances: int,
        n_unmonitored: int,
        max_failures: int,
    ):
        self.log = TaggingLogger(LOGGER, {"tag": "Collector"})
        self.target = target
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.n_regions = n_regions
        self.n_clients_per_region = n_clients_per_region
        self.n_instances = n_instances
        self.n_monitored = n_monitored
        self.n_unmonitored = n_unmonitored
        self.max_failures = max_failures

        self.wip_output_dir = self.output_dir.with_suffix(".wip")
        # TODO: Move to the run
        self.wip_output_dir.mkdir(exist_ok=True)

        self.region_queues: Tuple[Queue, ...] = tuple(
            Queue() for _ in range(n_regions)
        )
        for queue in self.region_queues:
            for client_id in range(n_clients_per_region):
                queue.put_nowait(client_id)

        self.runners: Dict[str, Tuple[TargetRunner, bool]] = {}

    async def run(self):
        """Collect the required number of samples."""
        unused_inputs = list(self.input_dir.glob("*.json"))
        unused_inputs = self.create_runners(unused_inputs)

        pending_tasks = {
            asyncio.create_task(runner.run(
                n_instances=(self.n_instances if is_monitored else 1),
                max_failures=self.max_failures
            ))
            for (runner, is_monitored) in self.runners.values()
        }

        while pending_tasks:
            # Run tasks for some duration then rebalance and repeat
            try:
                (done, pending_tasks) = await asyncio.wait(
                    pending_tasks, timeout=DEFAULT_INTERVAL,
                    return_when=asyncio.FIRST_EXCEPTION
                )
            except Exception:
                cancel_tasks(pending_tasks, self.log)
                raise

            new_tasks = self._handle_done_tasks(done, pending_tasks)

            pending_tasks.update(new_tasks)

    def _handle_done_tasks(self, done, pending_tasks):
        # Identify failed tasks and raise any encountered exceptions
        failed_ids = {t.get_name() for t in done if not t.result()}
        # Anything that is not in failed_ids or pending_ids successfully
        # completed
        pending_ids = {t.get_name() for t in pending_tasks}

        req_monitored = 0
        req_unmonitored = {i: 0 for i in range(self.n_regions)}
        reusable = set()

        for name in failed_ids:
            runner, is_monitored = self.runners[name]
            assert runner.progress_ is not None

            # Discard failed unmonitored cases as they are unlikely to be
            # suitable for the monitored setting
            if not is_monitored:
                assert runner.progress_.use_only_region >= 0
                req_unmonitored[runner.progress_.use_only_region] += 1

                del self.runners[name]
            elif runner.progress_.total_completed() == 0:
                # Discard monitored cases that failed to complete any samples
                req_monitored += 1
                del self.runners[name]
            else:
                # These monitored completed at least 1 sample and can be reused
                reusable.add(name)

        new_tasks = []
        # Now we need to recreate new tasks
        # Successful unmonitored can become monitored
        successful_unmonitored = [
            name for (name, (_, is_monitored)) in self.runners.items()
            if not is_monitored and name not in pending_ids
            and name not in failed_ids
        ]
        while req

        # Prior failed monitored can become unmonitored


    def create_runners(self, all_inputs: List[Path]):
        """Initlaise the runners and returns the input list with any
        used inputs removed and sorted in descending order by their
        stem.
        """
        # TODO: Detect previous monitored/unmonitored samples
        all_inputs.sort(reverse=True, key=lambda p: int(p.stem))

        for i in range(self.n_monitored + self.n_unmonitored):
            path = all_inputs.pop()
            # Use the sample id as the key for lookups
            key = path.stem

            is_monitored = (i < self.n_monitored)
            setting = "monitored" if is_monitored else "unmonitored"
            output_dir = self.wip_output_dir / f"setting~{setting}" / key

            self.runners[key] = (
                TargetRunner(
                    self.target, path, output_dir, self.region_queues,
                    delay=DEFAULT_DELAY
                ),
                is_monitored
            )
        return all_inputs

        # # Create the initial tasks for monitored and unmonitored samples
        # for i in range(n_samples):
        #     in_path = all_inputs.pop()
        #     # Use the sample id as the key for lookups
        #     key = in_path.stem

        #     n_instances = self.n_instances if i < self.n_monitored else 1
        #     output_dir = self.wip_output_dir / in_path.stem

        #     runners[key] = TargetRunner(
        #         self.target, in_path, output_dir, self.region_queues,
        #         delay=DEFAULT_DELAY
        #     )
        #     tasks[key] = asyncio.create_task(
        #         runners[key].run(n_instances, self.max_failures), name=key
        #     )
        #     is_monitored[key] = (i < self.n_monitored)


class TargetRunner:
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
        self.progress_: Optional[ProgressTracker] = None

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

    def _check_for_prior_samples(self):
        """Checks for prior successful runs."""
        assert self.progress_ is not None
        for i in range(self.n_regions):
            success_dir = (self.output_dir / f"region_id~{i}/status~success/")
            n_successes = sum(1 for _ in success_dir.glob("run~*"))
            self.log.info("Found %d successes for region %d", n_successes, i)
            self.progress_.completed[i] = n_successes

        if np.sum(self.progress_.completed) == 0:
            # No successes, so all failures were sequential
            n_failures = sum(
                1 for _ in
                self.output_dir.glob("region_id~*/status~failure/**/run~*")
            )
            self.log.info("Found %d sequential failures", n_failures)
            self.progress_.overall_failures = n_failures

    async def run(self, n_instances: int, max_failures: int = 3) -> bool:
        """Run collection until there are at least n_instances collected
        across the various regions.
        """
        self.log.info(
            "Running collection of %d samples with max %d failure(s).",
            n_instances, max_failures
        )
        self.progress_ = ProgressTracker(
            n_instances=n_instances, n_regions=len(self.region_queues),
            max_failures=max_failures,
        )

        self._check_for_prior_samples()
        if self.progress_.is_complete():
            self.log.info("Already complete, doing nothing.")
            return True
        if self.progress_.has_too_many_failures():
            self.log.info("Collection failed due to too many failures.")
            return False

        region_tasks = [
            asyncio.create_task(self.collect_region(i))
            for i in range(self.n_regions)
        ]

        try:
            # No point storing the resulting task since if there is a failure
            # we cannot use the task to cancel the rest as it would have
            # completed due to the failure.
            await asyncio.gather(*region_tasks)
        except TooManyFailuresException:
            await cancel_tasks(region_tasks, self.log)
            self.log.info("Collection failed due to too many failures.")
            return False
        except Exception:
            # If one of the region collections fails then cancel all regions.
            await cancel_tasks(region_tasks, self.log)
            # Reraise the error to let the caller know there was an unexpected
            # failure
            raise
        else:
            return True

    async def collect_region(self, region_id: int):
        """Collect the samples for the specified region_id."""
        assert self.progress_ is not None

        logger = TaggingLogger(LOGGER, {"tag": f"Region({region_id})"})
        loop = asyncio.get_running_loop()

        region_dir = self.output_dir / f"region_id~{region_id}"
        run_id = _get_initial_run_id(region_dir)
        logger.info("Starting from run_id %d", run_id)

        while not self.progress_.is_complete(region_id):
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
                self.progress_.success(region_id)
            else:
                logger.debug("Run failed: %d", run_id)
                shutil.move(str(output_dir), region_dir / "status~failure")
                self.progress_.failure(region_id)

                self.progress_.check_failures()
            logger.debug("Released client: %d", client_id)

            if self.progress_.is_complete(region_id):
                break

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


async def cancel_tasks(tasks, logger):
    """Cancel and supress the cancelled error for the iterable of tasks."""
    for task in (t for t in tasks if not t.done()):
        logger.warning("Cancelling task %s", task.get_name())
        task.cancel()
        # Wait for the cancelled error to raise in the task, will
        # be as soon as it is scheduled so no other exception
        # should be raised
        with contextlib.suppress(asyncio.CancelledError):
            await task


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

    def total_completed(self) -> int:
        """Return the total number of successes."""
        assert self.completed is not None
        return np.sum(self.completed)

    def has_too_many_failures(self) -> bool:
        """Return True iff the number of sequential failures has
        hit the maximum.
        """
        return self.sequential_failures() >= self.max_failures

    def check_failures(self):
        """Raises TooManyFailuresException if there are at least
        max_failures sequential failures.
        """
        if self.has_too_many_failures():
            assert self.completed is not None
            raise TooManyFailuresException(
                self.completed.sum(), self.sequential_failures()
            )

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
