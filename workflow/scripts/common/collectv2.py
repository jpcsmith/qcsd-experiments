"""Orchestrate the collection of web-pages."""
# pylint: disable=too-many-instance-attributes
import shutil
import random
import logging
import asyncio
import contextlib
from pathlib import Path
from asyncio import Queue
from itertools import islice
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, List, Dict, TypedDict

import numpy as np
import pandas as pd

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

        self.progress_ = ProgressTracker(0, len(self.region_queues))

        self._init_directories()
        self._check_for_prior_samples()

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
        assert self.progress_.completed is not None
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
        self.progress_.n_instances = n_instances
        self.progress_.max_failures = max_failures

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


class Collector:
    """Performs collection."""
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
        self.wip_output_dir.mkdir(exist_ok=True)

        self.region_queues: Tuple[Queue, ...] = tuple(
            Queue() for _ in range(n_regions)
        )
        for queue in self.region_queues:
            for client_id in range(n_clients_per_region):
                queue.put_nowait(client_id)

        self.runners: Dict[str, Tuple[TargetRunner, bool]] = {}
        self.next_region = 0

    def is_monitored(self, key: str) -> bool:
        """Return True if the sample identified by the key is assigned
        to the monitored setting, false otherwise.
        """
        return self.runners[key][1]

    def init_runners(self):
        """Initialises the runners considering assignments in prior runs."""
        input_files = list(self.input_dir.glob("*.json"))
        input_files.sort(reverse=True, key=lambda p: int(p.stem))
        input_map = {path.stem: path for path in input_files}

        counts = {True: 0, False: 0}

        for (is_monitored, setting) in [
            (True, "monitored"), (False, "unmonitored")
        ]:
            for output_dir in self.wip_output_dir.glob(f"setting~{setting}/*"):
                name = output_dir.stem
                self.runners[name] = (
                    self._new_runner(input_map[name], output_dir), is_monitored
                )

                if not is_monitored:
                    self._set_region(self.runners[name][0])

                counts[is_monitored] += 1
                # Remove it from future consideration
                del input_map[name]

        input_files = list(input_map.values())

        for (n_required, is_monitored) in [
            (self.n_monitored, True), (self.n_unmonitored, False)
        ]:
            n_required -= counts[is_monitored]
            self._create_runners(n_required, input_files, is_monitored)
        return input_files

    def _set_region(self, runner, region_id=None):
        if region_id is not None:
            runner.progress_.use_only_region = region_id
        elif runner.progress_.total_completed() > 0:
            assert runner.progress_.completed is not None
            region_id = np.argmax(runner.progress_.completed)
            runner.progress_.use_only_region = region_id
        else:
            runner.progress_.use_only_region = self.next_region
            self.next_region += 1
            self.next_region = self.next_region % self.n_regions

    def get_region(self, key: str) -> int:
        """Return the region associated with the runner or -1 if it
        collects from all regions.
        """
        return self.get_runner(key).progress_.use_only_region

    def get_runner(self, key: str) -> TargetRunner:
        """Return the runner associated with the key."""
        return self.runners[key][0]

    def _create_runners(
        self, n_required: int, input_files, is_monitored: bool,
        region_id=None
    ) -> List[Path]:
        setting = "monitored" if is_monitored else "unmonitored"
        (self.wip_output_dir / f"setting~{setting}").mkdir(exist_ok=True)

        for _ in range(n_required):
            path = input_files.pop()
            # Use the sample id as the key for lookups
            key = path.stem

            output_dir = self.wip_output_dir / f"setting~{setting}" / key
            output_dir.mkdir(exist_ok=False)

            self.runners[key] = (
                self._new_runner(path, output_dir), is_monitored,
            )
            if not is_monitored:
                self._set_region(self.get_runner(key), region_id)
        return input_files

    def _new_runner(self, indir, outdir) -> "TargetRunner":
        return TargetRunner(
            self.target, indir, outdir, self.region_queues, delay=DEFAULT_DELAY
        )

    async def run(self):
        """Collect the required number of samples."""
        unused_inputs = self.init_runners()

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

                # Identify failed tasks and raise any encountered exceptions
                failed_ids = {t.get_name() for t in done if not t.result()}
            except Exception:
                cancel_tasks(pending_tasks, self.log)
                raise

            pending_ids = {t.get_name() for t in pending_tasks}
            new_tasks = self._schedule_new_tasks(
                failed_ids, pending_ids, unused_inputs
            )

            pending_tasks.update(new_tasks)

    def _schedule_new_tasks(self, failed_ids, pending_ids, unused_inputs):
        self.log.info(
            "%d runs failed, %d are still pending", len(failed_ids),
            len(pending_ids)
        )

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

        new_tasks = self._add_monitored_runners(
            req_monitored, failed_ids, pending_ids, unused_inputs
        )
        new_tasks.extend(self._add_unmonitored_runners(
            req_unmonitored, reusable, unused_inputs
        ))
        return new_tasks

    def _add_monitored_runners(
        self, n_required, failed_ids, pending_ids, unused_inputs
    ):
        self.log.info("Require %d monitored more", n_required)
        # Anything that is not failed or pending_ids successfully
        # completed

        tasks = []
        # Now we need to recreate new tasks
        # Prior successful unmonitored can become monitored
        successful_unmon = [
            name for (name, (_, is_monitored)) in self.runners.items()
            if not is_monitored and name not in pending_ids
            and name not in failed_ids
        ]

        idx = 0
        while n_required > 0 and idx < len(successful_unmon):
            name = successful_unmon[idx]
            runner, _ = self.runners[name]

            # Reassign it to be monitored
            self.runners[name] = (runner, True)
            # Clear its associated region
            runner.progress_.use_all_regions()

            self.log.debug("Reassigned %s to be monitored", name)

            new_dir = self.wip_output_dir / "setting~monitored" / name
            runner.output_dir = runner.output_dir.rename(new_dir)

            # Create a task to collect the samples
            tasks.append(
                asyncio.create_task(
                    runner.run(self.n_instances, self.max_failures)
                )
            )

            n_required -= 1
            idx += 1

        if n_required > 0:
            tasks.extend(self._create_runners(n_required, unused_inputs, True))
        return tasks

    def _add_unmonitored_runners(
        self, required_by_region, reusable, unused_inputs
    ):
        tasks = []

        # Prior failed monitored can become unmonitored
        for name in reusable:
            runner, _ = self.runners[name]
            assert runner.progress_ is not None
            assert runner.progress_.completed is not None

            is_used = False
            for region_id in required_by_region:
                if (
                    required_by_region[region_id] > 0
                    and runner.progress_.completed[region_id] > 0
                ):
                    self.runners[name] = (runner, False)

                    # Set it to collect from this region
                    self._set_region(runner, region_id)
                    required_by_region[region_id] -= 1
                    self.log.debug("Reassigned %s to be unmonitored", name)

                    is_used = True

                    new_dir = self.wip_output_dir / "setting~unmonitored" / name
                    runner.output_dir = runner.output_dir.rename(new_dir)

                    # Create a task for this. It should result in no collection,
                    # just a confirmation of the files existing
                    tasks.append(
                        asyncio.create_task(runner.run(1, self.max_failures))
                    )

                    # Do not check any other region_id
                    break

            if not is_used:
                self.log.warning("Found no use for sample %s", name)

        for region_id, n_required in required_by_region.items():
            if n_requires == 0:
                continue
            self._create_runners(n_required, input_files, False, region_id)

        # self, n_required: int, input_files, is_monitored: bool,
        #     # TODO: Complete this function
