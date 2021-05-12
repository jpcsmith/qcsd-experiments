"""Calculate padding-only scores.
"""
import logging
import itertools
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean, cosine
import pyinform
import fastdtw

import common
from common import timeseries, neqo


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


def simulate_with_lag(control, schedule, defended, offsets, rate="5ms"):
    defended_in = timeseries.resample(defended["in"], rate)
    simulated_out = control["out"].append(schedule["out"])

    best_offset = 0
    best_distance = np.inf
    best_simulated = None

    for offset in offsets:
        shifted_schedule = pd.Series(
            schedule["in"].values,
            index=(schedule["in"].index + pd.Timedelta(f"{offset}ms"))
        )
        simulated_in = control["in"].append(shifted_schedule)

        # Put together in a dataframe to ensure they have the same length and
        # indices
        frame = pd.DataFrame({
            "defended": defended_in,
            "simulated": timeseries.resample(simulated_in, rate),
        }).fillna(0)

        distance = euclidean(frame["defended"], frame["simulated"])
        if distance < best_distance:
            best_offset = offset
            best_distance = distance
            best_simulated = simulated_in

    assert best_simulated is not None
    return (
        best_offset,
        pd.DataFrame({"in": best_simulated, "out": simulated_out}).fillna(0)
    )


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
    assert len(input_["control"]) == len(input_["control_pcap"]) \
        == len(input_["defended"]) == len(input_["defended_pcap"]) \
        == len(input_["schedule"]), "unequal file lists"

    results = []
    for i in range(len(input_["control"])):
        directory = Path(input_["control"][i]).parent

        if (
            not neqo.is_run_successful(input_["control"][i])
            or not neqo.is_run_successful(input_["defended"][i])
        ):
            _LOGGER.info("Skipping unsuccessful run: %s", directory)

        schedule_ts = timeseries.from_csv(input_["schedule"][i])
        control_ts = timeseries.from_pcap(input_["control_pcap"][i])
        defended_ts = timeseries.from_pcap(input_["defended_pcap"][i])

        offsets = range(ts_offset["min"], ts_offset["max"], ts_offset["inc"])
        (offset, simulated_ts) = simulate_with_lag(
            control_ts, schedule_ts, defended_ts, offsets,
        )
        _LOGGER.info("Using an incoming offset of %d ms", offset)

        for rate, direction in itertools.product(resample_rates, ("in", "out")):
            series = pd.DataFrame({
                "a": timeseries.resample(defended_ts[direction], rate),
                "b": timeseries.resample(simulated_ts[direction], rate),
            }).fillna(0)

            results.append([
                rate,
                direction,
                pearsonr(series["a"], series["b"])[0],
                euclidean(series["a"], series["b"]),
                fastdtw.dtw(series["a"], series["b"])[0],
                cosine(series["a"], series["b"]),
                pyinform.mutualinfo.mutual_info(series["a"], series["b"])
            ])

    pd.DataFrame(results, columns=[
        "rate", "dir", "pearsonr", "euclidean", "dtw", "cosine", "mutual_info"
    ]).to_csv(output[0], header=True, index=False)


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, **snakemake.params) # type: ignore # noqa # pylint: disable=E0602
