#!/usr/bin/env python3
"""Usage: hide_sizes.py [options] STRATEGY INPATH OUTPATH

Round all incoming packet sizes in INPATH to a power of 2 or to 1500-bytes
and save the resulting HDF5 dataset to OUTPATH.

Options:
    --direction <dir>
        Round sizes for the specified direction. Either "in", "out"
        or "both" [default: both].
    --mtu <mtu>
        Max packet size to use [default: 1500].
"""
import math
import shutil
from pathlib import Path

import h5py
import numpy as np

from common.doceasy import doceasy, Use, Or


def main(
    inpath: Path, outpath: Path, strategy: str, *, direction: str, mtu: int
):
    """Copy the dataset then round incoming packets."""
    assert direction in ("in", "out", "both")
    shutil.copyfile(inpath, outpath)

    bins = _make_bins(strategy, mtu)
    with h5py.File(outpath, mode="r+") as h5file:
        for i, size_trace in enumerate(h5file["sizes"]):
            # Round all packets up according to the bins
            rounded = bins[np.digitize(np.abs(size_trace), bins, right=True)]

            # Create a mask to select only the incoming packets
            incoming_mask = (size_trace < 0)

            if direction in ("in", "both"):
                # Update the incoming packets with the rounded sizes
                size_trace[incoming_mask] = -rounded[incoming_mask]

            if direction in ("out", "both"):
                # Update the outgoing packets with the rounded sizes
                size_trace[~incoming_mask] = rounded[~incoming_mask]

            h5file["sizes"][i] = size_trace


def _make_bins(strategy: str, mtu: int) -> np.ndarray:
    assert mtu > 0, "MTU must be positive"

    if strategy == "padE":
        max_pow = int(math.log(mtu, 2))
        bins = [0] + [2**i for i in range(max_pow+1)]
    elif strategy == "pad500":
        max_factor = mtu // 500
        bins = [500*i for i in range(max_factor+1)]
    elif strategy == "padMTU":
        bins = [0, mtu]
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    if bins[-1] < mtu:
        bins.append(mtu)
    return np.asarray(bins)


if __name__ == "__main__":
    main(**doceasy(__doc__, {
        "INPATH": Use(Path),
        "OUTPATH": Use(Path),
        "STRATEGY": Or("padE", "pad500", "padMTU"),
        "--direction": Or("in", "out", "both"),
        "--mtu": Use(int),
    }))
