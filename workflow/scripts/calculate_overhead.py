#!/usr/bin/env python3
"""Usage: calculate_overhead.py [options] DEFENCE INPUT_DIR

Create a CSV of bandwidth and latency overheads for the simulated
and collected traces for the specified defence (front or tamaraw).

Options:
    --n-jobs <n>
        Use n processes to calculate the overhead for the various files
        found in the directory [default: 1].

    --tamaraw-config <json>
        A JSON dictionary with the configuration values for Tamaraw.
        Required if the defence is tamaraw.
"""
import json
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
from lab.defences import tamaraw

import common
from common.doceasy import doceasy, Use, Or

_LOGGER = logging.getLogger(__name__)


def main(
    input_dir: Path,
    defence: str,
    tamaraw_config: Optional[str],
    n_jobs: int
):
    """Calculate the overhead using multiple processes."""
    common.init_logging()
    _LOGGER.info("Using parameters: %s", locals())

    if defence == "tamaraw":
        if tamaraw_config is None:
            raise ValueError("Tamaraw configuration required.")
        tamaraw_config = json.loads(tamaraw_config)

    assert input_dir.is_dir(), f"invalid path {input_dir}"
    directories = sorted(
        [x.parent for x in Path(input_dir).glob("**/defended/")]
    )
    _LOGGER.info("Found %d samples", len(directories))

    func = functools.partial(
        _calculate_overhead, defence=defence, tamaraw_config=tamaraw_config)
    if n_jobs > 1:
        chunksize = max(len(directories) // (n_jobs * 2), 1)
        with multiprocessing.pool.Pool(n_jobs) as pool:
            scores = list(
                pool.imap_unordered(func, directories, chunksize=chunksize)
            )
    else:
        # Run in the main process
        scores = [func(x) for x in directories]
    _LOGGER.info("Overhead calculation complete")

    results = pd.DataFrame.from_records(itertools.chain.from_iterable(scores))
    print(results.to_csv(header=True, index=False), end="")


def _parse_duration(path):
    """Return the time taken to download all of the application's HTTP
    resources in ms, irrespective of any additional padding or traffic,
    as logged by the run.
    """
    tag = "[FlowShaper] Application complete after "  # xxx ms
    found = None
    with (path / "stdout.txt").open(mode="r") as stdout:
        found = [line for line in stdout if line.startswith(tag)][-1]
    assert found, f"Run never completed! {path}"

    # Parse the next word as an integer
    return int(found[len(tag):].split()[0])


def _calculate_overhead(dir_, *, defence: str, tamaraw_config):
    try:
        control = np.sort(tracev2.from_csv(dir_ / "undefended" / "trace.csv"))
        defended = np.sort(tracev2.from_csv(dir_ / "defended" / "trace.csv"))
        schedule = np.sort(tracev2.from_csv(dir_ / "defended" / "schedule.csv"))
    except Exception as err:
        raise ValueError(f"Error loading files in {dir_}") from err

    undefended_size = np.sum(np.abs(control["size"]))
    defended_size = np.sum(np.abs(defended["size"]))
    simulated_size = np.sum(np.abs(schedule["size"]))
    simulated_size_alt = None
    # Add the undefended size as padding only defences only list the padding
    # in the schedule.
    if defence == "front":
        simulated_size += undefended_size

    if defence == "tamaraw":
        tamaraw_trace = tamaraw.simulate(
            control,
            packet_size=tamaraw_config["packet_size"],
            rate_in=tamaraw_config["rate_in"] / 1000,
            rate_out=tamaraw_config["rate_out"] / 1000,
            pad_multiple=tamaraw_config["packet_multiple"],
        )
        simulated_size_alt = np.sum(np.abs(tamaraw_trace["size"]))

    assert control["time"][0] == 0, "trace must start at 0s"
    # The trace should also already be sorted and in seconds
    undefended_ms = int(control["time"][-1] * 1000)
    defended_ms = _parse_duration(dir_ / "defended")
    if defence == "front":
        simulated_ms = undefended_ms
    elif defence == "tamaraw":
        unpadded_tamaraw = tamaraw.simulate(
            control,
            packet_size=tamaraw_config["packet_size"],
            rate_in=tamaraw_config["rate_in"] / 1000,
            rate_out=tamaraw_config["rate_out"] / 1000,
            pad_multiple=1,
        )
        simulated_ms = int(unpadded_tamaraw["time"][-1] * 1000)
    else:
        raise ValueError(f"Unsupported defence: {defence}")

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
            "overhead": "bandwidth",
            "setting": "simulated-alt",
            "value": ((simulated_size_alt - undefended_size) / undefended_size
                      if simulated_size_alt is not None else None)
        },
        {
            "sample": str(dir_),
            "overhead": "latency",
            "setting": "collected",
            "value": (defended_ms - undefended_ms) / undefended_ms
        },
        {
            "sample": str(dir_),
            "overhead": "latency",
            "setting": "simulated",
            "value": (simulated_ms - undefended_ms) / undefended_ms
        },
    ]


if __name__ == "__main__":
    main(**doceasy(__doc__, {
        "DEFENCE": Or("tamaraw", "front"),
        "INPUT_DIR": Use(Path),
        "--n-jobs": Use(int),
        "--tamaraw-config": Or(None, str),
    }))
