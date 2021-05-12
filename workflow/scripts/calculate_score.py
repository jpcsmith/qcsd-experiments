"""Calculate padding-only scores.
"""
import logging
import itertools
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


def simulate_with_lag(control, schedule, defended, offsets, rate="5ms"):
    """Find the best offset such that the simulated trace formed by
    combining the control and schedule where the schedule is lagged
    the offset, has the lowest euclidean distance to the defended trace.
    """
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
    columns = ["sample", "rate", "dir", "pearsonr", "euclidean", "dtw",
               "cosine", "mutual_info"]

    if (
        not neqo.is_run_successful(input_["control"])
        or not neqo.is_run_successful(input_["defended"])
    ):
        _LOGGER.info("Skipping as run was unsuccessful")
        pd.DataFrame([], columns=columns).to_csv(output[0], index=False)
        return

    sample = Path(input_["control"]).parent.name

    schedule_ts = timeseries.from_csv(input_["schedule"])
    control_ts = timeseries.from_pcap(input_["control_pcap"])
    defended_ts = timeseries.from_pcap(input_["defended_pcap"])

    offsets = range(ts_offset["min"], ts_offset["max"], ts_offset["inc"])
    (offset, simulated_ts) = simulate_with_lag(
        control_ts, schedule_ts, defended_ts, offsets,
    )
    _LOGGER.info("Using an incoming offset of %d ms", offset)

    results = []
    for rate, direction in itertools.product(resample_rates, ("in", "out")):
        series = pd.DataFrame({
            "a": timeseries.resample(defended_ts[direction], rate),
            "b": timeseries.resample(simulated_ts[direction], rate),
        }).fillna(0)
        _LOGGER.info("Series length at rate %s: %d", rate, len(series))
        _LOGGER.info("Series summary at rate %s: %s", rate, series.describe())

        try:
            mutual_info = pyinform.mutualinfo.mutual_info(
                series["a"], series["b"]
            )
        except pyinform.error.InformError as err:
            mutual_info = np.NaN
            _LOGGER.error("Unable to compute mutalinfo: %s", err)

        results.append([
            sample,
            rate,
            direction,
            pearsonr(series["a"], series["b"])[0],
            euclidean(series["a"], series["b"]),
            fastdtw.fastdtw(series["a"], series["b"])[0],
            cosine(series["a"], series["b"]),
            mutual_info,
        ])

    pd.DataFrame(results, columns=columns).to_csv(output[0], index=False)


if __name__ == "__main__":
    main(snakemake.input, snakemake.output, **snakemake.params) # type: ignore # noqa # pylint: disable=E0602
