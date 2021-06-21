"""Tests for collect.py"""
from pathlib import Path

from collect import Sample, Task


def test_sample_creates_tasks():
    """It should create a task per region."""
    sample = Sample(Path("A"), n_instances=10, n_regions=3)
    assert sample.get_tasks() == [Task(Path("A"), 0), Task(Path("A"), 1),
                                  Task(Path("A"), 2)]

    sample = Sample(Path("B"), n_instances=5, n_regions=1)
    assert sample.get_tasks() == [Task(Path("B"), 0), ]


def test_sample_is_complete():
    """It should track completion of the sample."""
    sample = Sample(Path("A"), n_instances=2, n_regions=2)

    assert not sample.is_complete()
    sample.task_succeeded(region_id=0)
    assert not sample.is_complete()
    sample.task_succeeded(region_id=0)
    assert not sample.is_complete()
    sample.task_succeeded(region_id=1)
    assert not sample.is_complete()
    sample.task_succeeded(region_id=1)
    assert sample.is_complete()
