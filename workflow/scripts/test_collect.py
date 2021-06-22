"""Tests for collect.py"""
# pylint: disable=invalid-name
from collect import ProgressTracker


def test_tracking():
    """It should track completion of the web-page."""
    tracker = ProgressTracker(n_instances=4, n_regions=2)
    assert tracker.remaining_regions() == [0, 1]

    assert not tracker.is_complete()
    tracker.success(region_id=0)
    assert not tracker.is_complete()
    assert tracker.remaining_regions() == [0, 1]

    tracker.success(region_id=0)
    assert not tracker.is_complete()
    assert tracker.remaining_regions() == [1]

    tracker.success(region_id=1)
    assert not tracker.is_complete()
    assert tracker.remaining_regions() == [1]

    tracker.success(region_id=1)
    assert tracker.is_complete()
    assert tracker.remaining_regions() == []


def test_sequential_failures_across_regions():
    """It should report the number of failures across regions."""
    tracker = ProgressTracker(n_instances=12, n_regions=3)

    tracker.failure(region_id=0)
    assert tracker.sequential_failures() == 1
    tracker.failure(region_id=1)
    assert tracker.sequential_failures() == 2
    tracker.failure(region_id=2)
    assert tracker.sequential_failures() == 3


def test_sequential_failures_within_regions():
    """It should report the number of failures max of the number of failures
    within or across regions.
    """
    tracker = ProgressTracker(n_instances=12, n_regions=3)

    tracker.failure(region_id=0)
    assert tracker.sequential_failures() == 1
    tracker.failure(region_id=1)
    assert tracker.sequential_failures() == 2
    tracker.failure(region_id=0)
    assert tracker.sequential_failures() == 3
    tracker.success(region_id=2)
    assert tracker.sequential_failures() == 2


def test_use_region():
    """It should return the correct region."""
    tracker = ProgressTracker(n_instances=1, n_regions=3)
    assert tracker.remaining_regions() == [0]

    tracker = ProgressTracker(n_instances=5, n_regions=3, use_only_region=2)
    assert tracker.remaining_regions() == [2]


def test_use_region_status():
    """It should return the correct region."""
    tracker = ProgressTracker(n_instances=1, n_regions=3, use_only_region=2)
    assert not tracker.is_complete()
    tracker.success(2)
    assert tracker.is_complete()
