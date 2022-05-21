"""Class related to tracking progress for dataset collection."""
from typing import List
from dataclasses import dataclass, field
import numpy as np


def _empty_arr():
    return np.empty(0)


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
    completed: np.ndarray = field(init=False, default_factory=_empty_arr)
    #: The total number of sequential failures across all regions
    overall_failures: int = field(init=False, default=0)
    #: The number of sequential failures within a region
    failures: np.ndarray = field(init=False, default_factory=_empty_arr)

    def __post_init__(self):
        self.failures = np.zeros(self.n_regions, int)
        assert self.failures.shape == (self.n_regions, )
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
