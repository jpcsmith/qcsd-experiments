"""Plot the fraction of successful samples for the excess MSD experiment.
"""
import itertools
from pathlib import Path
from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _results(base_dir: str, excess_msd: int):
    sample_id = 0
    while True:
        path = Path(f"{base_dir}/{excess_msd}/{sample_id:03d}_00/stdout.txt")
        if not path.is_file():
            break

        yield {
            "excess_msd": excess_msd,
            "sample_id": sample_id,
            "success": ">> SUCCESS <<" in path.read_text()
        }
        sample_id += 1


def plot(base_dir: str, values: List[int], output_path: str):
    """Find the result files and create the plot."""
    results = itertools.chain.from_iterable(
        [_results(base_dir, excess_msd) for excess_msd in values])
    data = pd.DataFrame(results)
    data = data.groupby("excess_msd").agg({"success": ["sum", "size"]})
    data = ((data["success"]["sum"] / data["success"]["size"])
            .rename("success_rate").reset_index())

    fig, axes = plt.subplots()

    sns.barplot(data=data, x="excess_msd", y="success_rate", ax=axes,
                palette="ch:.25")

    axes.set_ylim(0, 1.05)
    axes.set_ylabel("Successful fraction of samples")
    axes.set_xlabel("Excess MSD (bytes)")
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    plot(base_dir=snakemake.params["base_dir"],
         values=snakemake.params["values"],
         output_path=snakemake.output[0])
