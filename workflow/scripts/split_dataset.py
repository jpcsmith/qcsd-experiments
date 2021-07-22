"""Split a dataset and provide the indices of the splits."""
import json
from typing import List
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import common


def main(
    input_: str, output: List[str], n_folds: int, seed: int,
    validation_size: float
):
    """Split the dataset writing each the indices of each split to a file
    in output.
    """
    common.init_logging()

    with h5py.File(input_, mode="r") as h5in:
        labels = np.asarray(h5in["/labels"]["class"])

    rng = np.random.RandomState(seed)
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=rng)

    for (train_val_idx, test_idx), outfile in zip(
        splitter.split(np.zeros_like(labels), labels),
        output
    ):
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=validation_size, random_state=rng,
            stratify=labels[train_val_idx]
        )

        Path(outfile).write_text(
            json.dumps({
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
                "train-val": train_val_idx.tolist()
            }, indent=None, separators=(",", ":"))
        )


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(
        input_=str(snakemake.input[0]),
        output=list(snakemake.output),
        **snakemake.params,
    )
