"""Tests for the collectv2 module."""
from pathlib import Path
from asyncio import Queue
from typing import Tuple

import pytest

from common.collectv2 import TargetRunner, Collector

N_REGIONS: int = 3


def touch_target(_: Path, output_dir: Path, *args, **kwargs) -> bool:
    """Target used with the collector which just creates a touch file."""
    # pylint: disable=unused-argument
    (output_dir / "touch").touch(exist_ok=False)
    return True


def failing_target(infile, output_dir, region_id, client_id):  # pylint: disable=unused-argument
    """Returns false when client_id equals the region_id."""
    if client_id == region_id:
        (output_dir / "touch").write_text("failed")
        return False
    (output_dir / "touch").write_text("succeded")
    return True


class TargetError(RuntimeError):
    """An arbitrary error raised by the target."""


def exception_target(*args, **kwargs):
    """Target that always raises TargetError."""
    raise TargetError("This will always raise")


def create_queues(n_regions: int, n_clients: int):
    """Create n_region queues each with client_ids from 0 to n_clients."""
    region_queues: Tuple[Queue, ...] = tuple(Queue() for _ in range(n_regions))
    for queue in region_queues:
        for i in range(n_clients):
            queue.put_nowait(i)
    return region_queues


@pytest.mark.asyncio
async def test_collects_regions(tmp_path: Path):
    """It should collect the specified number of files across all regions."""
    n_instances = 5
    collector = TargetRunner(
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1)
    )
    is_success = await collector.run(n_instances)
    assert is_success
    assert (tmp_path / "region_id~0/status~success/run~0/touch").is_file()
    assert (tmp_path / "region_id~1/status~success/run~0/touch").is_file()
    assert (tmp_path / "region_id~2/status~success/run~0/touch").is_file()
    assert (tmp_path / "region_id~0/status~success/run~1/touch").is_file()
    assert (tmp_path / "region_id~1/status~success/run~1/touch").is_file()
    assert not (tmp_path / "region_id~2/status~success/run~1/touch").is_file()


@pytest.mark.asyncio
async def test_continues_collection(tmp_path: Path):
    """It should continue an already partially started collection."""
    n_instances = 8
    collector = TargetRunner(
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1)
    )

    all_files = [
        (tmp_path / f"region_id~{i}/status~success/run~{j}/touch")
        for j in range(4) for i in range(N_REGIONS)
    ]
    already_complete = all_files[:5]
    to_be_completed = all_files[5:n_instances]
    not_to_exist = all_files[n_instances:]

    for path in already_complete:
        path.parent.mkdir(parents=True)
        path.write_text("completed")

    is_success = await collector.run(n_instances)
    assert is_success

    # Check that the prior completed still contain the same text
    for path in already_complete:
        assert path.read_text() == "completed"
    # Check that the ones to be completed exist
    for path in to_be_completed:
        assert path.is_file()
    # Check that no more were created
    for path in not_to_exist:
        assert not path.exists()


@pytest.mark.asyncio
async def test_continues_with_failures(tmp_path: Path):
    """It should continue an already partially started collection."""
    n_instances = 9
    collector = TargetRunner(
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1)
    )

    all_files = [
        (tmp_path / f"region_id~{i}/status~success/run~0/touch")
        for i in range(N_REGIONS)
    ] + [
        (tmp_path / f"region_id~{i}/status~failure/run~1/touch")
        for i in range(N_REGIONS)
    ] + [
        (tmp_path / f"region_id~{i}/status~success/run~{j}/touch")
        for j in range(2, 4) for i in range(N_REGIONS)
    ]

    already_complete = all_files[:6]
    to_be_completed = all_files[6:n_instances+3]
    not_to_exist = all_files[n_instances+3:]

    for path in already_complete:
        path.parent.mkdir(parents=True)
        path.write_text("completed")

    is_success = await collector.run(n_instances)
    assert is_success

    # Check that the prior completed still contain the same text
    for path in already_complete:
        assert path.read_text() == "completed"
    # Check that the ones to be completed exist
    for path in to_be_completed:
        assert path.is_file()
    # Check that no more were created
    for path in not_to_exist:
        assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "n_failures,max_failures,should_fail", [(4, 5, False), (2, 2, True)]
)
async def test_too_many_prior_failures(
    tmp_path, n_failures, max_failures, should_fail
):
    """It should immediately raise if there have been more than max_failures
    with no successes.
    """
    n_instances = 1
    collector = TargetRunner(
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1)
    )

    for run_id in range(n_failures):
        path = tmp_path / f"region_id~0/status~failure/run~{run_id}/touch"
        path.parent.mkdir(parents=True)
        path.write_text("failed")

    is_success = await collector.run(n_instances, max_failures=max_failures)
    assert is_success == (not should_fail)


@pytest.mark.asyncio
async def test_already_complete(tmp_path: Path):
    """It should not run if already complete."""
    n_instances = 3
    collector = TargetRunner(
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1)
    )

    all_files = [
        (tmp_path / f"region_id~{i}/status~success/run~{j}/touch")
        for j in range(2) for i in range(N_REGIONS)
    ]
    already_complete = all_files[:n_instances]
    not_to_exist = all_files[n_instances:]

    for path in already_complete:
        path.parent.mkdir(parents=True)
        path.write_text("completed")

    is_success = await collector.run(n_instances)
    assert is_success

    # Check that the prior completed still contain the same text
    for path in already_complete:
        assert path.read_text() == "completed"
    for path in not_to_exist:
        assert not path.exists()


@pytest.mark.asyncio
async def test_cleanup_on_error(tmp_path):
    """It should cancel all tasks on error."""
    collector = TargetRunner(
        exception_target, Path(), tmp_path, create_queues(N_REGIONS, 1)
    )

    with pytest.raises(TargetError):
        await collector.run(10)


@pytest.mark.asyncio
async def test_success_with_failures(tmp_path):
    """Should successfully complete even with failures."""
    n_instances = 6
    collector = TargetRunner(
        failing_target, Path(), tmp_path, create_queues(N_REGIONS, 3)
    )

    is_success = await collector.run(n_instances)
    assert is_success


@pytest.mark.asyncio
async def test_should_abort_on_failures(tmp_path):
    """It should stop running after too many failures."""
    n_instances = 6
    # With 1 client, client_id=0, region_id=0 should always fail
    collector = TargetRunner(
        failing_target, Path(), tmp_path, create_queues(N_REGIONS, 1)
    )

    is_success = await collector.run(n_instances)
    assert not is_success


def create_inputs(path: Path, n_inputs: int):
    """Create a directory for inputs at path and a directory for outputs.

    Add n_inputs files to the input directory and return the
    (input, output) paths.
    """
    input_dir = path / "inputs"
    input_dir.mkdir()
    output_dir = path / "outputs"
    output_dir.mkdir()
    for i in range(n_inputs):
        (input_dir / f"{i}.json").touch()
    return input_dir, output_dir


def test_should_detect_prior(tmp_path):
    """It should identify prior monitored/unmonitored assignments and
    respect them.
    """
    indir, outdir = create_inputs(tmp_path, 10)
    (outdir.with_suffix(".wip") / "setting~monitored/9/").mkdir(parents=True)
    (outdir.with_suffix(".wip") / "setting~unmonitored/0/").mkdir(parents=True)

    collector = Collector(
        touch_target, indir, outdir, n_regions=N_REGIONS, n_instances=2,
        n_clients_per_region=1, n_monitored=5, n_unmonitored=5, max_failures=1
    )
    collector.init_runners()
    assert collector.is_monitored("9")
    assert not collector.is_monitored("0")


def test_should_assign_unmonitored_region(tmp_path):
    indir, outdir = create_inputs(tmp_path, 10)
    for sample, region_id, n_runs in [("1", 0, 2), ("3", 2, 1)]:
        for i in range(n_runs):
            (
                outdir.with_suffix(".wip") / "setting~unmonitored" / sample
                / f"region_id~{region_id}" / "status~success" / f"run~{i}"
            ).mkdir(parents=True)

    collector = Collector(
        touch_target, indir, outdir, n_regions=N_REGIONS, n_instances=2,
        n_clients_per_region=1, n_monitored=5, n_unmonitored=5, max_failures=1
    )
    collector.init_runners()
    assert not collector.is_monitored("1")
    assert not collector.is_monitored("3")

    assert collector.runners["1"][0].progress_.use_only_region == 0
    assert collector.runners["3"][0].progress_.use_only_region == 2


def test_should_init_runners(tmp_path):
    """It should create a runner for each required sample."""
    indir, outdir = create_inputs(tmp_path, 10)
    (outdir.with_suffix(".wip") / "setting~monitored/9/status~success/run~0")\
        .mkdir(parents=True)
    (outdir.with_suffix(".wip") / "setting~unmonitored/0/status~success/run~0")\
        .mkdir(parents=True)

    collector = Collector(
        touch_target, indir, outdir, n_regions=N_REGIONS, n_instances=2,
        n_clients_per_region=1, n_monitored=5, n_unmonitored=5, max_failures=1
    )
    collector.init_runners()

    for i in [1, 2, 3, 4, 9]:
        assert collector.is_monitored(str(i))
    for i in [0, 5, 6, 7, 8]:
        assert not collector.is_monitored(str(i))


# TODO Should be assigned to region which was already collected


# TODO: Need to balance the regions
# TODO: Balance the regions when reassigning failures
# TODO: Need to check sequential failures in a region with no successes
# TODO: Ensure that when resuming we pick monitored and unonitored samples
# correctly so that we dont waste already having a 100 samples on an
# unmonitored case
# TODO: Need to set the number of threads to use
