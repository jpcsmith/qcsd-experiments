"""Tests for the collectv2 module."""
import random
import threading
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


def failing_target(infile, output_dir, region_id, client_id):
    # pylint: disable=unused-argument
    """Returns false when client_id equals the region_id."""
    if client_id == region_id:
        (output_dir / "touch").write_text("failed")
        return False
    (output_dir / "touch").write_text("succeded")
    return True


def randomly_failing_target(chance: float, seed, raise_on_err=False):
    """A thread safe target which randomly fail with the specified chance."""
    assert 0 < chance < 1

    rng = random.Random(seed)
    lock = threading.Lock()

    # pylint: disable=unused-argument
    def _randomly_failing_target(infile, output_dir, region_id, client_id):
        with lock:
            value = rng.random()

        if value <= chance and raise_on_err:
            raise TargetError()
        if value <= chance:
            (output_dir / "touch").write_text("failed")
            return False
        (output_dir / "touch").write_text("succeded")
        return True

    return _randomly_failing_target


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
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1), delay=0
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

    collector = TargetRunner(
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1), delay=0
    )
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
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1), delay=0
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

    for run_id in range(n_failures):
        path = tmp_path / f"region_id~0/status~failure/run~{run_id}/touch"
        path.parent.mkdir(parents=True)
        path.write_text("failed")

    collector = TargetRunner(
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1), delay=0
    )
    is_success = await collector.run(n_instances, max_failures=max_failures)
    assert is_success == (not should_fail)


@pytest.mark.asyncio
async def test_too_many_region_failures(tmp_path):
    """It should fail due to too many failures in a single region."""
    n_failures = 3
    for (region_id, status, count) in [
        (0, "failure", n_failures), (1, "success", 4)
    ]:
        for run_id in range(count):
            path = (
                tmp_path / f"region_id~{region_id}" / f"status~{status}"
                / f"run~{run_id}" / "touch"
            )
            path.parent.mkdir(parents=True)
            path.write_text("failed")

    collector = TargetRunner(
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1), delay=0
    )
    is_success = await collector.run(10, max_failures=n_failures)
    assert not is_success


@pytest.mark.asyncio
async def test_already_complete(tmp_path: Path):
    """It should not run if already complete."""
    n_instances = 3

    all_files = [
        (tmp_path / f"region_id~{i}/status~success/run~{j}/touch")
        for j in range(2) for i in range(N_REGIONS)
    ]
    already_complete = all_files[:n_instances]
    not_to_exist = all_files[n_instances:]

    for path in already_complete:
        path.parent.mkdir(parents=True)
        path.write_text("completed")

    collector = TargetRunner(
        touch_target, Path(), tmp_path, create_queues(N_REGIONS, 1), delay=0
    )
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
        exception_target, Path(), tmp_path, create_queues(N_REGIONS, 1),
        delay=0
    )

    with pytest.raises(TargetError):
        await collector.run(10)


@pytest.mark.asyncio
async def test_success_with_failures(tmp_path):
    """Should successfully complete even with failures."""
    n_instances = 6
    collector = TargetRunner(
        failing_target, Path(), tmp_path, create_queues(N_REGIONS, 3), delay=0
    )

    is_success = await collector.run(n_instances)
    assert is_success


@pytest.mark.asyncio
async def test_should_abort_on_failures(tmp_path):
    """It should stop running after too many failures."""
    n_instances = 6
    # With 1 client, client_id=0, region_id=0 should always fail
    collector = TargetRunner(
        failing_target, Path(), tmp_path, create_queues(N_REGIONS, 1), delay=0
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
    """It should assign unmonitored regions to prior collections."""
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

    assert collector.get_region("1") == 0
    assert collector.get_region("3") == 2


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


@pytest.mark.asyncio
@pytest.mark.parametrize("n_inputs,n_monitored,n_unmonitored", [(20, 4, 6)])
async def test_runs_to_completion(
    tmp_path, n_inputs, n_monitored, n_unmonitored
):
    """It should successfully run a collection in spite of failures."""
    indir, outdir = create_inputs(tmp_path, n_inputs)
    collector = Collector(
        randomly_failing_target(0.1, seed=32), indir, outdir,
        n_regions=2, n_instances=20, n_clients_per_region=2,
        n_monitored=n_monitored, n_unmonitored=n_unmonitored, max_failures=2,
        delay=0.01
    )

    await collector.run()


@pytest.mark.asyncio
@pytest.mark.parametrize("n_inputs,n_monitored,n_unmonitored", [(20, 4, 6)])
async def test_runs_to_failure(
    tmp_path, n_inputs, n_monitored, n_unmonitored
):
    """It should successfully run a collection in spite of failures."""
    indir, outdir = create_inputs(tmp_path, n_inputs)
    collector = Collector(
        randomly_failing_target(0.1, seed=32, raise_on_err=True), indir, outdir,
        n_regions=2, n_instances=20, n_clients_per_region=2,
        n_monitored=n_monitored, n_unmonitored=n_unmonitored, max_failures=2,
        delay=0.01
    )

    with pytest.raises(TargetError):
        await collector.run()
