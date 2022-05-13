#!/usr/bin/env python3
"""Usage: extract_kfp_features.py [options] INFILE

Extract features for the k-FP classifier from the datasets "sizes"
and "timestamps" in the HDF INFILE, and write them to stdout as a
CSV.
"""
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from lab.classifiers import kfingerprinting

from common import doceasy


def extract_features(infile: Path):
    """Perform the extraction."""
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
        level=logging.INFO)

    with h5py.File(infile, mode="r") as h5in:
        logging.info("Loading dataset...")
        sizes = np.asarray(h5in["sizes"], dtype=object)
        times = np.asarray(h5in["timestamps"], dtype=object)
        labels = np.asarray(h5in["labels"]["class"])

    # Extract time and size related features
    features = kfingerprinting.extract_features_sequence(
        sizes=sizes, timestamps=times, n_jobs=None
    )
    frame = pd.DataFrame(features, columns=kfingerprinting.ALL_DEFAULT_FEATURES)
    frame.insert(0, "y_true", labels)
    print(frame.to_csv(index=False, header=True), end="")


if __name__ == "__main__":
    extract_features(**doceasy.doceasy(__doc__, {
        "INFILE": doceasy.Use(Path),
    }))
