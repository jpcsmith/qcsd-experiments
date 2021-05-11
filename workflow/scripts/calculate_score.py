"""Calculate padding-only scores.
"""
import logging
import itertools
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean, cosine
import pyinform
import fastdtw

import common
from common import timeseries


_LOGGER = logging.getLogger(__name__)


def lag_timeseries(
    series_a: pd.Series, series_b: pd.Series, offsets, rate: str = "5ms"
) -> Tuple[int, pd.Series]:
    """Return a tuple of series_a offset by some amount of time that
    minimizes the euclidean distance between series_a and series_b,
    and the integer offset in milliseconds.

    The resampling rate is a frequency string such as '50ms'.
    """
    series_b = timeseries.resample(series_b, rate)

    best_offset = 0
    best_distance = np.inf

    for offset in offsets:
        shifted_index = series_a.index + pd.Timedelta(f"{offset}ms")

        # Put together in a dataframe to ensure they have the same length and
        # indices
        frame = pd.DataFrame({
            "shifted": timeseries.resample(
                pd.Series(series_a.values, index=shifted_index), rate),
            "other": series_b
        }).fillna(0)

        distance = np.linalg.norm(frame["shifted"] - frame["other"])
        if distance < best_distance:
            best_offset = offset
            best_distance = distance

    final_index = series_a.index + pd.Timedelta(f"{best_offset}ms")
    return (best_offset, pd.Series(series_a.values, index=final_index))


def main(input_, output, *, ts_offset, resample_rates):
    """Score how close a padding-only defended time series is to
    padding schedule.

    Params:
        input_["schedule"]: The filename of the CSV schedule
        input_["defence"]: The filename of the CSV details of the defence
            trace.
        output[0]: The file to write the resulting scores CSV
        ts_offset["min"], ts_offset["max"], ts_offset["inc"]:
            Range definition of the offsets to shift the defended
            timeseries by when comparing it to the schedule.
        resample_rates: List of frequencies, e.g, 5ms, 10ms, etc.,
            defining the rates to resample the timeseries at when
            when calculating the scores.
    """
    common.init_logging()

    chaff_ts = timeseries.from_trace(
        pd.read_csv(input_["defence"]), length_col="length_chaff"
    )
    schedule_ts = timeseries.from_csv(input_["schedule"])

    offsets = range(ts_offset["min"], ts_offset["max"], ts_offset["inc"])
    (offset, chaff_ts["in"]) = lag_timeseries(
        chaff_ts["in"], schedule_ts["in"], offsets=list(offsets)
    )
    _LOGGER.info("Using an incoming offset of %d ms", offset)

    results = []
    for rate, direction in itertools.product(resample_rates, ("in", "out")):
        series = pd.DataFrame({
            "a": timeseries.resample(chaff_ts[direction], rate),
            "b": timeseries.resample(schedule_ts[direction], rate),
        }).fillna(0)

        results.extend([
            (rate, "pearson", direction, pearsonr(series["a"], series["b"])[0]),
            (rate, "euclidean", direction, euclidean(series["a"], series["b"])),
            (rate, "dtw", direction, fastdtw.dtw(series["a"], series["b"])[0]),
            (rate, "cosine", direction, cosine(series["a"], series["b"])),
            (rate, "mutualinfo", direction,
             pyinform.mutualinfo.mutual_info(series["a"], series["b"]))
        ])

    pd.DataFrame(
        results, columns=["rate", "metric", "dir", "value"]
    ).to_csv(str(output[0]), header=True, index=False)


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, **snakemake.params) # type: ignore # noqa # pylint: disable=E0602
