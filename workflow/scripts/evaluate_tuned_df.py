#!/usr/bin/env python3
"""Usage: evaluate_tuned_kfp.py [options] DATASET_PATH [OUTFILE]

Options:
    --verbose <val>
        Verbosity [default: 0]
"""
# pylint: disable=too-many-instance-attributes
import copy
import time
import logging
import dataclasses
from pathlib import Path
from typing import Optional, ClassVar

import h5py
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from lab.feature_extraction.trace import ensure_non_ragged
from lab.classifiers.dfnet import DeepFingerprintingClassifier
from lab.metrics import rprecision_score, recall_score
import tensorflow_addons as tfa

from common import doceasy
from common.doceasy import Use


def main(outfile, **kwargs):
    """Create and run the experiment and write the CSV results to outfile."""
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
        level=logging.INFO
    )

    (probabilities, y_true, classes) = Experiment(**kwargs).run()

    outfile.writerow(["y_true"] + list(classes))
    outfile.writerows(
        np.hstack((np.reshape(y_true, (-1, 1)), probabilities)))


@dataclasses.dataclass
class Experiment:
    """An experiment to evalute hyperparameter tuned deep-fingerprinting
    classifier.
    """
    # The path to the dataset of sizes and times
    dataset_path: Path

    # The fraction of samples to use for testing the final model
    test_size: float = 0.2

    # Number of folds used in the stratified k-fold cross validation to
    # select the hyperparameters
    # TODO: revert to 5
    n_folds: int = 2

    verbose: int = 0

    # Random seed for the experiment
    seed: int = 4511

    # The hyperparameters to tune for the DF classifier
    tuned_parameters: list = dataclasses.field(default_factory=lambda: [
        {
            "first_n_packets__kw_args": [{"n_packets": n_features}],
            # TODO: Revert to 30
            "dfnet__epochs": [10],
            "dfnet__metric": ["accuracy", "weighted_f1score"],
            "dfnet__n_features": [n_features]
        }
        # TODO: Revert to multiple options
        for n_features in [5000]
    ])

    # Other seeds which are chosen for different operations
    seeds_: Optional[dict] = None

    logger: ClassVar = logging.getLogger("Experiment")

    def run(self):
        """Perform hyperparameter tuning of the DeepFingerprinting
        classifier, then return the predictions of the best fitting
        classifier.
        """
        rng = np.random.default_rng(self.seed)
        self.seeds_ = {
            "train_test": rng.integers(10_000),
            "kfold_shuffle": rng.integers(10_000),
        }

        self.logger.info("Running %s", self)
        start = time.perf_counter()

        X, y = self.load_dataset()
        n_classes = len(np.unique(y))
        self.logger.info("Dataset shape=%s, n_classes=%d", X.shape, n_classes)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, shuffle=True,
            random_state=self.seeds_["train_test"]
        )

        grid_search = GridSearchCV(
            Pipeline([
                ("first_n_packets", FunctionTransformer(first_n_packets)),
                ("dfnet", DeepFingerprintingClassifier(
                    n_classes=n_classes, verbose=min(self.verbose, 1)
                ))
            ]),
            self.get_tuned_parameters(n_classes),
            cv=StratifiedKFold(
                self.n_folds, shuffle=True,
                random_state=self.seeds_["kfold_shuffle"]
            ),
            scoring=make_scorer(rf1_score),
            verbose=self.verbose
        )
        grid_search.fit(x_train, y_train)
        self.logger.info("grid search results = %s", grid_search.cv_results_)

        probabilities = grid_search.predict_proba(x_test)
        self.logger.info(
            "Experiment complete in %.2fs.", (time.perf_counter() - start))

        return (probabilities, y_test, grid_search.best_estimator_.classes_)

    def load_dataset(self):
        """Load the dataset and return the X and y samples as ndarrays.

        X is of shape (n, max_packets), and y of shape (n, )
        for n samples in the dataset.
        """
        with h5py.File(self.dataset_path, mode="r") as h5in:
            sizes = np.asarray(h5in["sizes"])
            classes = np.asarray(h5in["labels"]["class"][:])

        # TODO: Remove, this is for debugging purposes
        mask = (classes < 2)
        classes = classes[mask][9500:]
        sizes = sizes[mask][9500:]
        # TODO: Remove above

        max_packets = max(
            d["dfnet__n_features"] for d in self.tuned_parameters
        )[0]
        return ensure_non_ragged(sizes, dimension=max_packets), classes

    def get_tuned_parameters(self, n_classes: int):
        """Return the the parameters with any tags converted to their actual
        values.
        """
        params = copy.deepcopy(self.tuned_parameters)
        for entry in params:
            if (
                "dfnet__metric" in entry
                and "weighted_f1score" in entry["dfnet__metric"]
            ):
                idx = entry["dfnet__metric"].index("weighted_f1score")
                entry["dfnet__metric"][idx] = tfa.metrics.F1Score(
                    num_classes=n_classes, average="weighted"
                )
        return params


def first_n_packets(X, n_packets: int):
    """Return the first n_packets packets in the dataset."""
    return X[:, :n_packets]


def rf1_score(y_true, y_pred, *, negative_class=-1, ratio=20):
    """Compute the F1-score using the r-precisions with the specified ratio
    and recall.
    """
    precision = rprecision_score(
        y_true, y_pred, negative_class=negative_class, ratio=ratio
    )
    recall = recall_score(y_true, y_pred, negative_class=negative_class)
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, {
        "OUTFILE": doceasy.CsvFile(mode="w", default="-"),
        "DATASET_PATH": Use(Path),
        "--verbose": Use(int),
    }))
