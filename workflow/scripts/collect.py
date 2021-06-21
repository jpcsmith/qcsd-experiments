"""Perform the collection.
"""
import logging
import threading
from queue import Queue
from pathlib import Path
from typing import List
import dataclasses

import common


class Client(threading.Thread):
    """A client for collecting a web-page sample, potentially through a
    VPN gateway.
    """
    def __init__(self, region_id: int, client_id: int, queue: Queue):
        super().__init__(name=f"client-{region_id}-{client_id}")
        self.region_id = region_id
        self.client_id = client_id
        self.queue = queue
        self.log = logging.getLogger(__name__)

    def collect(self, task):
        """Collect and update the task with the result."""
        self.log.debug("Processing task: %s", task)
        # We can modify the task here because the caller waits on the queue,
        # not on the task
        task.is_done = True
        task.is_success = True

    def run(self):
        while True:
            task = self.queue.get(block=True)
            if task is None:
                self.log.info("Received sentinel. Stopping thread")
                self.queue.task_done()
                break
            self.collect(task)
            self.queue.task_done()


@dataclasses.dataclass
class Task:
    input_file: Path
    region_id: int
    is_done: bool = False
    is_success: bool = False


class Sample:
    def __init__(self, input_file: Path, *, n_instances: int, n_regions: int):
        self.input_file = input_file
        self.n_instances = n_instances
        self.n_regions = n_regions
        self.region_collected = [0] * n_regions

    def is_complete(self) -> bool:
        """Return true iff the collection of this sample is complete."""
        return False

    def task_succeeded(self, region_id: int):
        """Mark that a task for the specified region has succeeded."""

    def get_tasks(self) -> List[Task]:
        return [Task(self.input_file, i) for i in range(self.n_regions)]


class Collector:
    def __init__(
        self,
        n_regions: int = 3,
        n_clients_per_region: int = 4,
        n_monitored: int = 10,
        n_instances: int = 5,
        n_unmonitored: int = 20,
        input_dir: str = "results/determine-url-deps/dependencies",
    ):
        self.log = logging.getLogger(__name__)
        self.input_dir = input_dir
        self.n_regions = n_regions
        self.n_instances = n_instances
        self.n_clients_per_region = n_clients_per_region

        self.region_queues: List[Queue] = [Queue() for _ in range(n_regions)]

    def run(self):
        samples = dict()
        for path in sorted(list(Path(self.input_dir).glob("*.json"))):
            id_ = path.stem
            samples[id_] = Sample(path, self.n_instances, self.n_regions)

        threads = []
        for region_id, queue in enumerate(self.region_queues):
            for client_id in range(self.n_clients_per_region):
                threads.append(Client(region_id, client_id, queue))

        # TODO: Should be a subset, monitored and unmonitored
        while True:
            tasks = []
            for sample in samples.values():
                tasks.extend(sample.get_tasks())
            if not tasks:
                break

            for task in tasks:
                self.region_queues[task.region].put(task)

            for queue in self.region_queues:
                queue.join()

        # TODO: What happens if the above raises an exception
        self._close(threads)
        # TODO: Rename files

    def _close(self, threads):
        for queue in self.region_queues:
            for _ in range(self.n_clients_per_region):
                queue.put(None)
        for thread in threads:
            self.log.debug("Waiting for %s to close.", thread.name)
            thread.join()


def main():
    """Start the collection."""
    # The code needs to satisfy the following design goals
    #
    # - Each client should have no more than 1 job running in parallel
    common.init_logging(verbosity=2, name_thread=True)
    collector = Collector()
    collector.run()


# main()
