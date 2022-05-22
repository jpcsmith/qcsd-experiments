"""Orchestrate the collection of web-pages."""
# pylint: disable=too-many-instance-attributes
import re
import time
import shutil
import logging
import asyncio
import contextlib
from pathlib import Path
from asyncio import Queue, Task
from itertools import islice
from typing import (
    Callable, Tuple, List, Dict, Set, Iterator, Optional
)
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common.progress_tracker import ProgressTracker

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


class TargetRunner:
    """Performs collections for a single URL."""
    def __init__(
        self,
        target: TargetFn,
        input_file: Path,
        output_dir: Path,
        region_queues: Tuple[Queue, ...],
        *,
        delay: float = DEFAULT_DELAY,
        name: str = "??"
    ):
        self.name = name
        self.target = target
        self.region_queues = region_queues
        self.input_file = input_file
        self.output_dir = output_dir
        self.delay = delay
        self.log = TaggingLogger(LOGGER, {"tag": f"Runner({name})"})

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

            self.log.debug("Found %d successes for region %d", n_successes, i)
            self.progress_.completed[i] = n_successes

            failure_dir = (self.output_dir / f"region_id~{i}/status~failure/")
            n_failures = sum(1 for _ in failure_dir.glob("run~*"))

            if n_successes == 0 and n_failures > 0:
                self.log.debug(
                    "Found %d sequential failures for region %d", n_failures, i
                )
                self.progress_.failures[i] = n_failures

        if np.sum(self.progress_.completed) == 0:
            # No successes, so all failures were sequential
            n_failures = sum(
                1 for _ in
                self.output_dir.glob("region_id~*/status~failure/**/run~*")
            )
            self.log.debug("Found %d overall sequential failures", n_failures)
            self.progress_.overall_failures = n_failures

    async def run(self, n_instances: int, max_failures: int = 3) -> bool:
        """Run collection until there are at least n_instances collected
        across the various regions.
        """
        self.log.debug(
            "Collecting %d samples with max %d failure(s).", n_instances,
            max_failures
        )
        self.progress_.n_instances = n_instances
        self.progress_.max_failures = max_failures

        if self.progress_.is_complete():
            self.log.debug("Not running, already complete.")
            return True
        if self.progress_.has_too_many_failures():
            self.log.warning("Not running, too many failures.")
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
            self.log.warning("Collection failed due to too many failures.")
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

        logger = TaggingLogger(
            LOGGER, {"tag": f"Runner({self.name}-{region_id})"}
        )
        loop = asyncio.get_running_loop()

        region_dir = self.output_dir / f"region_id~{region_id}"
        run_id = _get_initial_run_id(region_dir)
        logger.debug("Starting from run_id %d", run_id)

        while not self.progress_.is_complete(region_id):
            output_dir = region_dir / f"run~{run_id}"
            output_dir.mkdir(exist_ok=False)

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

                if self.progress_.has_too_many_failures():
                    logger.debug("Aborting collection due to failures.")
                    raise TooManyFailuresException()

            if self.progress_.is_complete(region_id):
                break

            run_id += 1
            await asyncio.sleep(self.delay)

        logger.debug("Completed collection for region.")

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
        logger.debug("Cancelling task %s", task.get_name())
        task.cancel()
        # Wait for the cancelled error to raise in the task, will
        # be as soon as it is scheduled so no other exception
        # should be raised
        with contextlib.suppress(asyncio.CancelledError):
            await task


class BaseCollector:
    """Base class for Monitored and Unmonitored collectors."""
    def __init__(
        self,
        target: TargetFn,
        output_dir: Path,
        region_queues: Tuple[Queue, ...],
        *,
        n_web_pages: int,
        n_instances: int,
        max_failures: int,
        delay: float,
        logger,
    ):
        self.target = target
        self.output_dir = output_dir
        self.region_queues = region_queues
        self.max_failures = max_failures
        self.n_web_pages = n_web_pages
        self.n_instances = n_instances
        self.delay = delay
        self.log = logger

        self.runners: Dict[str, Tuple[TargetRunner, Optional[Task]]] = {}

    @property
    def n_regions(self) -> int:
        """Return the number of regions."""
        return len(self.region_queues)

    @property
    def deficit(self) -> int:
        """Return the number of web pages which this collector is lacking."""
        return max(0, self.n_web_pages - len(self.runners))

    def get_runner(self, key: str) -> TargetRunner:
        """Return the runner associated with the key."""
        return self.runners[key][0]

    def new_runner(self, infile: Path, outdir: Path) -> TargetRunner:
        """Return a new runner for the provided input file and output
        directory.
        """
        return TargetRunner(
            self.target, infile, outdir, self.region_queues, delay=self.delay,
            name=infile.stem
        )

    def resume(self, input_files: Set[Path]):
        """Identify and remove any samples in input_files which are
        already being collected.
        """
        input_map = {path.stem: path for path in input_files}

        for output_dir in self.output_dir.glob("*"):
            name = output_dir.stem
            input_file = input_map[name]

            self.runners[name] = (self.new_runner(input_file, output_dir), None)

            # Remove it from future consideration
            input_files.remove(input_file)

    def add_runners(self, input_files: List[Path]):
        """Consumes files from input_files to add runners to this
        collector.
        """
        self._create_runners(self.deficit, input_files)

    def _create_runners(self, n_required: int, input_files: List[Path]):
        new_runners = []

        for _ in range(n_required):
            input_file = input_files.pop()
            name = input_file.stem

            output_dir = self.output_dir / name
            output_dir.mkdir(exist_ok=False)

            self.runners[name] = (self.new_runner(input_file, output_dir), None)
            new_runners.append(name)

            self.log.debug("Added runner for %s", name)

        return new_runners

    def _filter_ids(self, names: Set[str]) -> Set[str]:
        """Return a copy of names with only managed runners."""
        return {name for name in names if name in self.runners}

    def create_tasks(self) -> Set[Task]:
        """Create tasks for runners which have not yet run and return them."""
        result_set = set()
        for name in self.runners:
            runner, task = self.runners[name]

            if task is not None:
                continue

            task = asyncio.create_task(
                runner.run(self.n_instances, self.max_failures), name=name
            )
            result_set.add(task)
            self.runners[name] = (runner, task)
        return result_set

    def __contains__(self, key) -> bool:
        return key in self.runners

    def relocate_files(self, output_dir: Path):
        """Relocate the run files to the output_dir."""
        assert len(self.runners) == self.n_web_pages
        for name, (runner, _) in self.runners.items():
            web_page_dir = (output_dir / name)
            web_page_dir.mkdir(exist_ok=False)

            n_files = [
                _relocate_files(runner.output_dir, web_page_dir, region_id)
                for region_id in range(self.n_regions)
            ]
            assert sum(n_files) >= self.n_instances


def _relocate_files(indir: Path, output_dir: Path, region_id: int):
    pattern = r"run~(?P<run_id>\d+)"
    files = list(indir.glob(f"region_id~{region_id}/status~success/run~*"))
    run_ids = [
        int(re.search(pattern, str(p))["run_id"]) for p in files  # type: ignore
    ]

    for i, idx in enumerate(np.argsort(run_ids)):
        files[idx].rename(output_dir / f"{region_id}_{i}")

    return len(files)


class MonitoredCollector(BaseCollector):
    """Stores and manages TaskRunners for collecting the monitored portion of
    the dataset.
    """
    def __init__(self, *args, n_monitored: int, n_instances: int, **kwargs):
        super().__init__(
            *args, n_web_pages=n_monitored, n_instances=n_instances,
            logger=TaggingLogger(LOGGER, {"tag": "Monitored"}), **kwargs
        )
        self.reusable_runners: Dict[str, TargetRunner] = {}

    def handle_completions(self, success_ids: Set[str], failed_ids: Set[str]):
        """Handle successes and failures of prior tasks."""
        success_ids = self._filter_ids(success_ids)
        failed_ids = self._filter_ids(failed_ids)
        # We leave the successes in the runners and remove the bad failures and
        # store the ones that could potentially be used
        for name in failed_ids:
            if self.get_runner(name).progress_.total_completed() > 0:
                # These monitored completed at least 1 sample and can be reused
                self.reusable_runners[name] = self.get_runner(name)
            del self.runners[name]

    def sacrifice_failed(self) -> Iterator[Tuple[str, TargetRunner]]:
        """Yield runners which have failed but show promise."""
        while self.reusable_runners:
            yield self.reusable_runners.popitem()

    def consume(self, generator: Iterator[Tuple[str, TargetRunner]]):
        """Add runners yielded by the generator."""
        for (name, runner) in islice(generator, self.deficit):
            # Move to the owned output directory
            new_dir = self.output_dir / name
            runner.output_dir = runner.output_dir.rename(new_dir)

            # Allow collecting from any region
            runner.progress_.use_all_regions()

            self.runners[name] = (runner, None)


class UnmonitoredCollector(BaseCollector):
    """Stores and manages TaskRunners for collecting the unmonitored
    portion of the dataset.

    Also responsible for balancing runners across regions.
    """
    def __init__(self, *args, n_unmonitored: int, **kwargs):
        super().__init__(
            *args, n_web_pages=n_unmonitored, n_instances=1,
            logger=TaggingLogger(LOGGER, {"tag": "Unmonitored"}), **kwargs
        )
        self.next_region = 0
        self.success_ids: List[str] = []

    def handle_completions(self, success_ids: Set[str], failed_ids: Set[str]):
        """Handle successes and failures of prior tasks."""
        success_ids = self._filter_ids(success_ids)
        failed_ids = self._filter_ids(failed_ids)
        # We make note of the successes and remove all failures
        self.success_ids.extend(success_ids)
        self.success_ids.sort(reverse=True, key=int)

        for name in failed_ids:
            # Discard failed unmonitored cases as they are unlikely to be
            # suitable for the monitored setting
            del self.runners[name]

    def resume(self, input_files: Set[Path]):
        super().resume(input_files)
        for name in self.runners:
            self.set_region(name)

    def set_region(self, key: str, region_id=None):
        """Set the region of the runner at key.

        If a region_id is specified, it is set to that, otherwise it is
        set to the first region with already collected samples. If no
        samples have been collected, it is set to the value defined by
        self.next_region and the next region is incremented.
        """
        progress = self.get_runner(key).progress_

        if region_id is not None:
            progress.use_only_region = region_id
        elif progress.total_completed() > 0:
            region_id = np.argmax(progress.completed)
            progress.use_only_region = region_id
        else:
            progress.use_only_region = self.next_region
            self.next_region = (self.next_region + 1) % self.n_regions

    def sacrifice_completed(self) -> Iterator[Tuple[str, TargetRunner]]:
        """Yield and remove completed task runners."""
        while self.success_ids:
            name = self.success_ids.pop()

            if self.get_runner(name).progress_.has_too_many_failures():
                # Not suitable for the monitored setting since it failed too
                # many times already
                continue

            # It succeeded without too many failures, we can use it
            (runner, _) = self.runners.pop(name)

            yield (name, runner)

    def consume(self, generator: Iterator[Tuple[str, TargetRunner]]):
        """Add runners yielded by the generator."""
        for (name, runner) in islice(generator, self.deficit):
            # Move to the owned output directory
            new_dir = self.output_dir / name
            runner.output_dir = runner.output_dir.rename(new_dir)

            self.runners[name] = (runner, None)

            # Set a region for collection
            self.set_region(name)

    def _create_runners(self, n_required: int, input_files: List[Path]):
        added_runners = super()._create_runners(n_required, input_files)
        for name in added_runners:
            assert self.get_runner(name).progress_.use_only_region == -1
            self.set_region(name)


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
        delay: float = DEFAULT_DELAY
    ):
        self.log = TaggingLogger(LOGGER, {"tag": "Collector"})
        self.target = target
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.n_regions = n_regions
        self.n_clients_per_region = n_clients_per_region

        self.wip_output_dir = self.output_dir.with_suffix(".wip")
        self.wip_output_dir.mkdir(exist_ok=True)

        self.region_queues: Tuple[Queue, ...] = tuple(
            Queue() for _ in range(n_regions)
        )
        for queue in self.region_queues:
            for client_id in range(n_clients_per_region):
                queue.put_nowait(client_id)

        path = (self.wip_output_dir / "setting~monitored")
        path.mkdir(exist_ok=True)
        self.monitored = MonitoredCollector(
            target, path, self.region_queues, max_failures=max_failures,
            n_monitored=n_monitored, n_instances=n_instances, delay=delay
        )

        path = (self.wip_output_dir / "setting~unmonitored")
        path.mkdir(exist_ok=True)
        self.unmonitored = UnmonitoredCollector(
            target, path, self.region_queues, max_failures=max_failures,
            n_unmonitored=n_unmonitored, delay=delay
        )

    def is_monitored(self, key: str) -> bool:
        """Return True if the sample identified by the key is assigned
        to the monitored setting, false otherwise.
        """
        return key in self.monitored

    def get_region(self, key: str) -> int:
        """Return the region associated with the runner or -1 if it
        collects from all regions.
        """
        return self.get_runner(key).progress_.use_only_region

    def get_runner(self, key: str) -> TargetRunner:
        """Return the runner associated with the key."""
        if key in self.monitored:
            return self.monitored.get_runner(key)
        return self.unmonitored.get_runner(key)

    def init_runners(self) -> List[Path]:
        """Initialises the runners considering assignments in prior runs."""
        input_files = set(self.input_dir.glob("*.json"))

        self.monitored.resume(input_files)
        self.unmonitored.resume(input_files)

        remaining_files = list(input_files)
        remaining_files.sort(reverse=True, key=lambda p: int(p.stem))

        self.monitored.add_runners(remaining_files)
        self.unmonitored.add_runners(remaining_files)

        return remaining_files

    async def run(self):
        """Collect the required number of samples."""
        # Configure the threadpool
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            ThreadPoolExecutor(self.n_regions * self.n_clients_per_region)
        )

        start_time = time.perf_counter()
        unused_inputs = self.init_runners()

        pending_tasks = self.monitored.create_tasks()
        pending_tasks.update(self.unmonitored.create_tasks())

        round_id = 0
        while pending_tasks:
            # Run tasks for some duration then rebalance and repeat
            try:
                (done, pending_tasks) = await asyncio.wait(
                    pending_tasks, timeout=DEFAULT_INTERVAL,
                    return_when=asyncio.FIRST_EXCEPTION
                )

                # Identify failed tasks and raise any encountered exceptions
                success_ids = {t.get_name() for t in done if t.result()}
                failed_ids = {t.get_name() for t in done if not t.result()}
            except Exception:
                cancel_tasks(pending_tasks, self.log)
                raise

            self.log.info(
                "Round %d: successes=%d, failures=%d, pending=%d",
                round_id, len(success_ids), len(failed_ids), len(pending_tasks)
            )
            self.monitored.handle_completions(success_ids, failed_ids)
            self.unmonitored.handle_completions(success_ids, failed_ids)

            self.monitored.consume(self.unmonitored.sacrifice_completed())
            self.unmonitored.consume(self.monitored.sacrifice_failed())

            self.monitored.add_runners(unused_inputs)
            self.unmonitored.add_runners(unused_inputs)

            pending_tasks.update(self.monitored.create_tasks())
            pending_tasks.update(self.unmonitored.create_tasks())

            round_id += 1

        duration = (time.perf_counter() - start_time) / 60
        self.log.info(
            "Completed after %f minutes with %d inputs remaining",
            duration, len(unused_inputs)
        )

        self._relocate_files()

    def _relocate_files(self):
        self.monitored.relocate_files(self.output_dir)
        self.unmonitored.relocate_files(self.output_dir)
