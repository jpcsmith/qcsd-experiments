"""Plot the fraction of successful samples for the excess MSD experiment.
"""
import logging
import itertools
from multiprocessing import pool
from pathlib import Path
from typing import List, Iterable, Tuple, Final, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

from common import pcap, timeseries


#: Possible offsets of the time-series to find the best match
_OFFSETS = list(range(0, 200, 5))

#: Resampling rate
_RATE: Final = "50ms"

_LOGGER = logging.getLogger(__name__)


def lagged_pearsonr(
    series_a: pd.Series,
    series_b: pd.Series,
    rate: str = _RATE,
) -> Tuple[float, float]:
    """Return the maximum pearson correlation coefficient between
    the resampled timeseries series_a and series_b, when series_a
    is lagged by various amounts.

    The resampling rate is a frequency string such as '50ms'.
    """
    series_b = timeseries.resample(series_b, rate)

    def _pearson_r(offset):
        shifted_index = series_a.index + pd.Timedelta(f"{offset}ms")

        # Put together in a dataframe to ensure they have the same length and
        # indices
        frame = pd.DataFrame({
            "shifted": timeseries.resample(
                pd.Series(series_a.values, index=shifted_index), rate),
            "other": series_b
        }).fillna(0)
        return stats.pearsonr(frame["shifted"], frame["other"])

    return max(
        (_pearson_r(offset) for offset in _OFFSETS), key=lambda coef: coef[0]
    )


def _compute_metrics(
    directory: Path,
    schedule: pd.DataFrame
) -> Dict[str, float]:
    """Compute various metrics from the trace in directory."""
    chaff_streams_file = directory.joinpath("chaff-stream-ids.txt")
    if not chaff_streams_file.is_file():
        chaff_streams_file = directory.joinpath("stdout.txt")

    data = pd.DataFrame(pcap.to_quic_lengths(
        str(directory.joinpath("trace.pcapng")), str(chaff_streams_file)))

    # This is the full data from the PCAP
    actual = timeseries.from_trace(data)
    # This is the application stream data + the schedule
    expected = (data[["time", "length_app_streams", "is_outgoing"]]
                .pipe(timeseries.from_trace, length_col="length_app_streams")
                .append(schedule)
                .groupby("time").sum()
                .sort_index())

    return {
        "pearson_in":
            lagged_pearsonr(actual["length_in"], expected["length_in"])[0],
        "pearson_out":
            lagged_pearsonr(actual["length_out"], expected["length_out"])[0],
    }


def _results(base_dir: str, excess_msd: int, ts):
    sample_id = 0

    while True:
        directory = Path(f"{base_dir}/{excess_msd}/{sample_id:04d}_00")
        stdout_file = directory.joinpath("stdout.txt")

        if not stdout_file.is_file():
            break

        success = ">> SUCCESS <<" in stdout_file.read_text()
        try:
            metrics = _compute_metrics(directory, ts) if success else {}
        except pcap.UndecryptedTraceError:
            _LOGGER.warning("Trace is still encrypted: %s", directory)
            metrics = {}

        yield {
            "excess_msd": excess_msd,
            "sample_id": sample_id,
            "success": success,
            **metrics,
        }

        sample_id += 1


def _mp_results(args) -> List:
    base_dir, excess_msd, ts = args
    return list(_results(base_dir, excess_msd, ts))

# ipdb> data[["excess_msd", "sample_id", "pearson_in", "pearson_out"]].dropna().set_index(["exces
# s_msd", "sample_id"]).rename_axis(columns="direction").stack().rename("pearson").reset_index()

def plot(base_dir: str, values: List[int], trend_path: str, heatmap_path: str):
    """Find the result files and create the plot."""
    ts = timeseries.from_csv("results/excess_msd/constant_1Mbps_10s_5ms.csv")

    results = itertools.chain.from_iterable(
        map(
            _mp_results, [(base_dir, excess_msd, ts) for excess_msd in values]),
    )
    data = pd.DataFrame(results)

    breakpoint()



if __name__ == "__main__":
    plot("results/excess_msd", [16, 2000, ], "", "")
    # plot(base_dir=snakemake.params["base_dir"],
    #      values=snakemake.params["values"],
    #      trend_path=snakemake.output["trend_path"],
    #      heatmap_path=snakemake.output["heatmap_path"])
