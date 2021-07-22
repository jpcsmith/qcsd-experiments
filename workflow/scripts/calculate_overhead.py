"""Create a CSV of bandwidth and latency overheads for the simulated
and collected traces.
"""
import logging
import itertools
import functools
from pathlib import Path
import multiprocessing
import multiprocessing.pool
from typing import Optional

import numpy as np
import pandas as pd
from lab import tracev2

import common

_LOGGER = logging.getLogger(__name__)


def main(
    input_,
    output, *,
    is_pad_only: bool,
    jobs: Optional[int] = None,
):
    """Calculate the overhead using multiple processes."""
    common.init_logging()
    _LOGGER.info("Using parameters: %s", locals())

    assert Path(input_).is_dir(), f"invalid path {input_}"
    directories = sorted([x.parent for x in Path(input_).glob("**/defended/")])
    _LOGGER.info("Found %d samples", len(directories))

    func = functools.partial(_calculate_overhead, is_pad_only=is_pad_only)
    jobs = jobs or (multiprocessing.cpu_count() or 4)
    if jobs > 1:
        chunksize = max(len(directories) // (jobs * 2), 1)
        with multiprocessing.pool.Pool(jobs) as pool:
            scores = list(
                pool.imap_unordered(func, directories, chunksize=chunksize)
            )
    else:
        # Run in the main process
        scores = [func(x) for x in directories]
    _LOGGER.info("Overhead calculation complete")

    pd.DataFrame.from_records(
        itertools.chain.from_iterable(scores)
    ).to_csv(output, header=True, index=False)


def _parse_duration(path):
    """Return the time taken to download all of the application's HTTP
    resources in ms, irrespective of any additional padding or traffic,
    as logged by the run.
    """
    tag = "[FlowShaper] Application complete after "  # xxx ms
    found = None
    with (path / "stdout.txt").open(mode="r") as stdout:
        found = next((line for line in stdout if line.startswith(tag)), None)
    assert found, f"Run never completed! {path}"

    # Parse the next word as an integer
    return int(found[len(tag):].split()[0])


def _calculate_overhead(dir_, *, is_pad_only: bool):
    control = tracev2.from_csv(dir_ / "undefended" / "trace.csv")
    defended = tracev2.from_csv(dir_ / "defended" / "trace.csv")
    schedule = tracev2.from_csv(dir_ / "defended" / "schedule.csv")

    undefended_size = np.sum(np.abs(control["size"]))
    defended_size = np.sum(np.abs(defended["size"]))
    simulated_size = np.sum(np.abs(schedule["size"]))
    # Add the undefended size as padding only defences only list the padding
    # in the schedule.
    if is_pad_only:
        simulated_size += undefended_size

    assert control["time"][0] == 0, "trace must start at 0s"
    # The trace should also already be sorted and in seconds
    undefended_ms = int(control["time"][-1] * 1000)
    defended_ms = _parse_duration(dir_ / "defended")

    return [
        {
            "sample": str(dir_),
            "overhead": "bandwidth",
            "setting": "collected",
            "value": (defended_size - undefended_size) / undefended_size
        },
        {
            "sample": str(dir_),
            "overhead": "bandwidth",
            "setting": "simulated",
            "value": (simulated_size - undefended_size) / undefended_size
        },
        {
            "sample": str(dir_),
            "overhead": "latency",
            "setting": "collected",
            "value": (defended_ms - undefended_ms) / undefended_ms
        },
    ]


if __name__ == "__main__":
    main(str(snakemake.input[0]), str(snakemake.output[0]), **snakemake.params,
         jobs=snakemake.threads)
