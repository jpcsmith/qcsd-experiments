"""Orchestrate the collection of web-pages"""
# pylint: disable=too-many-arguments,too-many-instance-attributes
# pylint: disable=too-few-public-methods,broad-except
import sys
import shutil
import logging
import threading
import dataclasses
from queue import Queue
from pathlib import Path
from itertools import islice
from typing import List, Dict, Callable, Optional, Set

import numpy as np

import common

#: The function to run  to perform the collection
#: It must accept the input_path, output_dir, region_id, and client_id and
#: return True if the collection was successful, false otherwise.
TargetFn = Callable[[Path, Path, int, int], bool]


def _output_directory(
    input_: Path, output_dir: Path, region_id: int, region_sample: int
) -> Path:
    return Path(output_dir, f"{input_.stem}/region-{region_id}/{region_sample}")


class Client(threading.Thread):
    """A client for collecting a web-page sample, potentially through a
    VPN gateway.
    """
    def __init__(
        self, target: TargetFn, region_id: int, client_id: int, queue: Queue
    ):
        super().__init__(name=f"client-{region_id}-{client_id}")
        self.target = target
        self.region_id = region_id
        self.client_id = client_id
        self.queue = queue
        self.log = logging.getLogger(__name__)

    def run(self):
        while True:
            task: Optional[Task] = self.queue.get(block=True)
            if task is None:
                self.log.debug("Received sentinel. Stopping thread")
                self.queue.task_done()
                break

            # Run the collection and mark the task as done
            # We can modify the task here because the caller waits on the queue,
            # not on the task
            try:
                self.log.debug("Processing task: %s", task)
                if task.output_dir.is_dir():
                    self.log.debug("Skipping as directory already exists.")
                    task.is_success = True
                else:
                    task.output_dir.mkdir(parents=True, exist_ok=True)
                    task.is_success = self.target(
                        task.input_file, task.output_dir, self.region_id,
                        self.client_id)
            except Exception:
                self.log.exception("Encountered an exception")
                task.is_success = False
                raise
            finally:
                task.is_done = True
                if not task.is_success:
                    self.log.debug("Removing directory %s", task.output_dir)
                    shutil.rmtree(task.output_dir)
                self.queue.task_done()


@dataclasses.dataclass
class Task:
    """A collection task."""
    #: The path to the input dependency file
    input_file: Path
    #: The directory to which outputs should go
    output_dir: Path
    #: The region on which this sample should be collected
    region_id: int
    #: True iff the task is done
    is_done: bool = False
    #: True iff the task completed successfully
    is_success: bool = False


@dataclasses.dataclass
class ProgressTracker:
    """Track the progress of the collection of a web-page."""
    #: The number of instances to collect
    n_instances: int
    #: The total number of regions
    n_regions: int = 1
    #: If use_region is non-negative, then all of the instances will be
    #: collected via a single region.
    use_only_region: int = -1

    #: The number of successes completed in the region
    completed: np.ndarray = dataclasses.field(init=False, default=None)
    #: The total number of sequential failures across all regions
    overall_failures: int = dataclasses.field(init=False, default=0)
    #: The number of sequential failures within a region
    failures: np.ndarray = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        if self.failures is None:
            self.failures = np.zeros(self.n_regions, int)
        assert self.failures.shape == (self.n_regions, )

        if self.completed is None:
            self.completed = np.zeros(self.n_regions, int)
        assert self.completed.shape == (self.n_regions, )

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
        return max(0, self.required(region_id) - self.completed[region_id])

    def sequential_failures(self) -> int:
        """Return the number of sequential failures either overall, or
        within any region, whichever is higher.
        """
        return max(self.overall_failures, max(self.failures))

    def failure(self, region_id: int):
        """Mark that a task for the specified region has failed."""
        self.overall_failures += 1
        self.failures[region_id] += 1

    def success(self, region_id: int):
        """Mark that a task for the specified region has succeeded."""
        self.completed[region_id] += 1
        self.failures[region_id] = 0
        self.overall_failures = 0


class Collector:
    """Orchestrates the collection of web-pages.

    Collects n_monitored × n_instances + n_unmonitored web-page samples
    using the provided function (target).

    The instances are distributed across the regions 0, ..., n_regions-1
    with n_clients_per_region created for each region to perform the
    collection.

    The input directory (input_dir) must contain files <stem>.json used
    as dependencies for the collection.  For each *.json file in
    input_dir, target is called with the path to file, an output
    directory, the region_id, and client_id.  The results are written to
    <output_dir>.wip/<stem>/<region_id>_<region_sample>/ before being
    moved to output_dir.

    A web-page is can have at most max_failures successive failures
    within or across its regions before it is discarded.
    """
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
        self.log = logging.getLogger(__name__)
        self.input_dir = input_dir
        self.output_dir = Path(output_dir)
        self.n_regions = n_regions
        self.n_instances = n_instances
        self.n_monitored = n_monitored
        self.n_unmonitored = n_unmonitored
        self.max_failures = max_failures
        self.n_clients_per_region = n_clients_per_region

        self.wip_output_dir = self.output_dir.with_suffix(".wip")
        self.bad_inputs: Set[Path] = self._load_bad_inputs()
        self.region_queues: List[Queue] = [Queue() for _ in range(n_regions)]
        self.workers = [Client(target, region_id, client_id, queue)
                        for region_id, queue in enumerate(self.region_queues)
                        for client_id in range(self.n_clients_per_region)]

    def _load_bad_inputs(self):
        cache = self.wip_output_dir / "bad-inputs-list.txt"
        bad_inputs = set()
        if cache.is_file():
            with cache.open(mode="r") as infile:
                bad_inputs = set(Path(p.strip()) for p in infile.readlines())
        self.log.debug("Loaded %d bad inputs", len(bad_inputs))
        return bad_inputs

    def _save_bad_inputs(self):
        cache = self.wip_output_dir / "bad-inputs-list.txt"
        cache.write_text("\n".join(str(path) for path in self.bad_inputs))

    def run(self):
        """Collect the required number of samples."""
        # Create the list of web-pages and sort in reverse order so that we can
        # pop with O(1)
        unused_web_pages = list(Path(self.input_dir).glob("*.json"))
        unused_web_pages.sort(reverse=True, key=lambda p: int(p.stem))

        trackers: Dict[Path, ProgressTracker] = dict()
        self._add_trackers(trackers, unused_web_pages)

        # Start the client workers
        for worker in self.workers:
            worker.start()

        try:
            while self._run_batch(trackers):
                if any(not worker.is_alive() for worker in self.workers):
                    self.log.critical("Exiting due to worker failure.")
                    sys.exit(1)

                if self._remove_excessive_failures(trackers):
                    self._add_trackers(trackers, unused_web_pages)
        finally:
            self._close()

        # Prune any empty directories
        # for (dirpath, dirs, files) in os.walk(
        #     self.wip_output_dir, topdown=False
        # ):
        #     if not files and not dirs:
        #         Path(dirpath).rmdir()
        # Move the work in progress directory to the final location
        self.wip_output_dir.rename(self.output_dir)

    def _add_trackers(self, trackers, unused_web_pages):
        if len(trackers) == self.n_monitored + self.n_unmonitored:
            return

        while len(trackers) != self.n_monitored + self.n_unmonitored:
            web_page = unused_web_pages.pop()
            if web_page in self.bad_inputs:
                self.log.debug("Skipping %s since it a bad input", web_page)
                continue

            trackers[web_page] = ProgressTracker(
                n_regions=self.n_regions, n_instances=1)

        for tracker in islice(trackers.values(), 0, self.n_monitored):
            # Set the correct number of instances
            tracker.n_instances = self.n_instances
            # Reset the region since this may have been unmonitored
            tracker.use_all_regions()

        # Count the regions currently being used by the unmonitored trackers
        # so that we can balance the regions specified for the new samples
        needs_region_specified = []
        region_counts = np.zeros(self.n_regions, int)
        for tracker in islice(
            trackers.values(), self.n_monitored, len(trackers)
        ):
            assert tracker.n_instances == 1
            if tracker.use_only_region >= 0:
                region_counts[tracker.use_only_region] += 1
            else:
                needs_region_specified.append(tracker)

        # Update the region to use for the newly added trackers
        for tracker in needs_region_specified:
            region_id = np.argmin(region_counts)
            tracker.use_only_region = region_id
            region_counts[region_id] += 1

    def _remove_excessive_failures(self, trackers) -> bool:
        # Check the batch for any excessive failures and remove them
        to_drop = [path for (path, tracker) in trackers.items()
                   if tracker.sequential_failures() >= self.max_failures]
        for path in to_drop:
            self.log.debug("Dropping %s as it has too many failures: %d",
                           path, trackers[path].sequential_failures())
            self.bad_inputs.add(path)
            # Stop using the sample for collection and remove any files
            # collected
            del trackers[path]
            shutil.rmtree(Path(self.wip_output_dir, f"{path.stem}"))

        self._save_bad_inputs()
        return bool(to_drop)

    def _run_batch(self, trackers) -> bool:
        """Return True if any jobs were run, false otherwise."""
        tasks = []
        for (path, tracker) in trackers.items():
            for r_id in tracker.remaining_regions():
                region_sample = tracker.completed[r_id]
                output_dir = Path(self.wip_output_dir, f"{path.stem}",
                                  f"{r_id}_{region_sample}")
                tasks.append(Task(path, region_id=r_id, output_dir=output_dir))
                self.region_queues[r_id].put(tasks[-1])

        if not tasks:
            return False

        # Wait for all of the tasks to be taken and completed
        for queue in self.region_queues:
            queue.join()

        # Record the status of the tasks
        has_a_success = False
        for task in tasks:
            assert task.is_done
            if task.is_success:
                trackers[task.input_file].success(task.region_id)
                has_a_success = True
            else:
                trackers[task.input_file].failure(task.region_id)
        return True

    def _close(self):
        for worker in self.workers:
            if worker.is_alive():
                self.region_queues[worker.region_id].put(None)
        for worker in self.workers:
            self.log.debug("Waiting for %s to close.", worker.name)
            worker.join()
        self.workers = []


def debug():
    """Run the collector with default debugging arguments."""
    common.init_logging(verbosity=2, name_thread=True)
    collector = Collector(
        target=lambda *_: True,
        input_dir="results/determine-url-deps/dependencies",
        output_dir="/tmp",
        n_regions=1,
        n_clients_per_region=2,
        n_monitored=5,
        n_instances=10,
        n_unmonitored=10,
        max_failures=3,
    )
    collector.run()


if __name__ == "__main__":
    debug()
