import json
import logging

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import common


def main(
    input_: str, output: str, n_folds: int, seed: int, validation_size: float
):
    common.init_logging()

    with h5py.File(input_, mode="r") as h5in:
        labels = np.asarray(h5in["/labels"]["class"])

    rng = np.random.RandomState(seed)
    splitter = StratifiedKFold(n_folds)

    with open(output, mode="w") as outfile:
        for train_val_idx, test_idx in splitter.split(
            np.zeros_like(labels), labels
        ):
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=validation_size, random_state=rng,
                stratify=labels[train_val_idx]
            )

            json.dump({
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
                "train-val": train_val_idx.tolist()
            }, outfile, indent=None, separators=(",", ":"))
            outfile.write("\n")


if __name__ == "__main__":
    snakemake = globals().get("snakemake", None)
    main(
        input_=str(snakemake.input[0]),
        output=str(snakemake.output[0]),
        **snakemake.params,
    )
