"""Plot the fraction of successful samples for the excess MSD experiment.
"""
import itertools
from multiprocessing import pool
from pathlib import Path
from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict


def _results(base_dir: str, excess_msd: int):
    sample_id = 0
    while True:
        path = Path(f"{base_dir}/{excess_msd}/{sample_id:04d}_00/stdout.txt")
        if not path.is_file():
            break

        yield {
            "excess_msd": excess_msd,
            "sample_id": sample_id,
            "success": ">> SUCCESS <<" in path.read_text()
        }
        sample_id += 1


def _mp_results(args) -> List:
    base_dir, excess_msd = args
    return list(_results(base_dir, excess_msd))


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


def plot(base_dir: str, values: List[int], trend_path: str, heatmap_path: str):
    """Find the result files and create the plot."""
    results = itertools.chain.from_iterable(
        pool.Pool().imap_unordered(
            _mp_results, [(base_dir, excess_msd) for excess_msd in values]))
    data = pd.DataFrame(results)

    _plot_trend(data).savefig(trend_path, bbox_inches="tight", dpi=150)
    _plot_heatmap(data).savefig(heatmap_path, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    plot(base_dir=snakemake.params["base_dir"],
         values=snakemake.params["values"],
         trend_path=snakemake.output["trend_path"],
         heatmap_path=snakemake.output["heatmap_path"])
