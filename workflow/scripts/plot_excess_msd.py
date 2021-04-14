"""Plot the fraction of successful samples for the excess MSD experiment.
"""
import logging
import itertools
from multiprocessing import pool
from pathlib import Path
from typing import List, Tuple, Final, Dict
from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


def _results(base_dir: str, excess_msd: int, sched_path: str):
    schedule = timeseries.from_csv(sched_path)

    sample_id = 0
    while True:
        directory = Path(f"{base_dir}/{excess_msd}/{sample_id:04d}_00")
        stdout_file = directory.joinpath("stdout.txt")

        if not stdout_file.is_file():
            break

        success = ">> SUCCESS <<" in stdout_file.read_text()
        try:
            metrics = _compute_metrics(directory, schedule) if success else {}
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
    base_dir, excess_msd, sched_path = args
    return list(_results(base_dir, excess_msd, sched_path))


def _plot_heatmap(data):
    """Plot a heatmap of the fraction of samples that succeeded with
    one excess MSD value but that failed with another, for every pair
    of excess MSD values.
    """
    table: Dict[int, Dict[int, int]] = defaultdict(dict)
    # Reshape to bool DF where rows are samples and cols are excess_msd values
    data = data.set_index(["sample_id", "excess_msd"]).unstack()["success"]

    # Count the number of samples that succeeded for msdA but failed for msdB
    for (msd1, msd2) in itertools.product(data.columns, repeat=2):
        table[msd1][msd2] = (data.loc[:, msd1] & ~data.loc[:, msd2]).sum()

    frame = pd.DataFrame.from_dict(table, orient="index")
    # Make the counts fractions of the total number of samples
    frame = (frame / len(data)).round(decimals=2)

    fig, axes = plt.subplots()
    sns.heatmap(frame.T, vmin=0, vmax=1, ax=axes, annot=True)
    axes.set_ylabel("Failed with excess MSD (bytes)")
    axes.set_xlabel("Succeeded with excess MSD (bytes)")

    return fig


def _plot_trend(data):
    """Plot the fraction of failures across all samples for each excess
    MSD factor.
    """
    data = data.groupby("excess_msd").agg({"success": ["sum", "size"]})
    data = ((data["success"]["sum"] / data["success"]["size"])
            .rename("success_rate").reset_index())

    fig, axes = plt.subplots()
    sns.pointplot(data=data, x="excess_msd", y="success_rate", ax=axes,
                  join=True)

    axes.set_ylim(0, 1.05)
    axes.set_ylabel("Successful fraction of samples")
    axes.set_xlabel("Excess MSD (bytes)")

    return fig


def _plot_pearson(data: pd.DataFrame):
    """Plot the pearson correlation correlation for the incoming and
    outgoing directions.
    """
    data = (data.melt(id_vars=["excess_msd", "sample_id"],
                      value_vars=["pearson_in", "pearson_out"],
                      var_name="direction",
                      value_name="pearsonr")
                .dropna())

    fig, axes = plt.subplots()
    sns.boxplot(data=data, x="excess_msd", y="pearsonr", hue="direction",
                ax=axes)

    axes.set_ylabel("Pearson Correlation Coefficient")
    axes.set_xlabel("Excess MSD (bytes)")

    return fig


def plot(
    base_dir: str,
    schedule_path: str,
    values: List[int],
    trend_path: str,
    heatmap_path: str,
    pearson_path: str,
):
    """Find the result files and create plots."""
    results = itertools.chain.from_iterable(
        pool.Pool().imap_unordered(
            _mp_results,
            [(base_dir, excess_msd, schedule_path) for excess_msd in values]
        )
    )
    data = pd.DataFrame(results)

    _plot_pearson(data).savefig(pearson_path, bbox_inches="tight", dpi=150)
    _plot_trend(data).savefig(trend_path, bbox_inches="tight", dpi=150)
    _plot_heatmap(data).savefig(heatmap_path, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    # plot(base_dir="results/excess_msd",
    #      schedule_path="results/excess_msd/constant_1Mbps_10s_5ms.csv",
    #      values=[16, 2000],
    #      trend_path="trend.png",
    #      heatmap_path="heatmap.png",
    #      pearson_path="pearson.png")
    plot(base_dir=snakemake.params["base_dir"],
         schedule_path=snakemake.input["schedule"],
         values=snakemake.params["values"],
         trend_path=snakemake.output["trend_path"],
         heatmap_path=snakemake.output["heatmap_path"],
         pearson_path=snakemake.output["pearson_path"])
