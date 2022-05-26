#!/usr/bin/env python3
"""Usage: evaluate_tuned_df.py [options] DATASET_PATH [OUTFILE]

Perform hyperparameter turning on the Deep Fingerprinting classifier.
The data is read from an HDF5 file located at DATASET_PATH and the
probability predictions are written in CSV format to OUTFILE (defaults
to stdout).

Options:
    --hyperparams <val>
        Either the string 'tune' to perform hyperparameter tuning or a
        string of the format 'key1=value1,key2=value2' describing the
        hyperparameters to use [default: tune].
    --verbose <val>
        Set the verbosity of the gridsearch and classifier. A value of
        0 disables output whereas 3 produces all output [default: 0].
"""
# pylint: disable=too-many-instance-attributes
import time
import logging
import dataclasses
from pathlib import Path
from typing import Optional, ClassVar, Sequence, Union

import h5py
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from lab.feature_extraction.trace import ensure_non_ragged
from lab.classifiers import dfnet
from lab.metrics import rprecision_score, recall_score
import tensorflow

from common import doceasy
from common.doceasy import Use, Or, And

PAPER_EPOCHS: int = 30
MAX_RAND_SEED: int = 10_000

# Hyperparameters from the paper
DEFAULT_N_PACKETS: int = 5000
DEFAULT_LEARNING_RATE: float = 0.002
DEFAULT_EPOCHS: int = 30


def main(outfile, **kwargs):
    """Create and run the experiment and write the CSV results to outfile."""
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
        level=logging.INFO
    )
    # Use autoclustering for speedup
    tensorflow.config.optimizer.set_jit("autoclustering")

    (probabilities, y_true, classes) = Experiment(**kwargs).run()

    y_pred = classes[np.argmax(probabilities, axis=1)]
    score = rf1_score(y_true, y_pred)
    logging.info("r_20 f1-score = %.4g", score)

    outfile.writerow(["y_true"] + list(classes))
    outfile.writerows(
        np.hstack((np.reshape(y_true, (-1, 1)), probabilities)))


@dataclasses.dataclass
class Experiment:
    """An experiment to evalute hyperparameter tuned DF classifier.
    """
    # The path to the dataset of sizes and times
    dataset_path: Path

    # The fraction of samples to use for testing the final model
    test_size: float = 0.1

    # Number of folds used in the stratified k-fold cross validation
    n_folds: int = 3

    # Level of debugging output from sklearn and tensorflow
    verbose: int = 0

    # Random seed for the experiment
    seed: int = 114155

    # Hyperparams to use if not "tune"
    hyperparams: Union[str, dict] = "tune"

    # Hyperparameters to search
    n_packet_parameters: Sequence[int] = (5_000, 7_500, 10_000)
    tuned_parameters: dict = dataclasses.field(default_factory=lambda: {
        "epochs": [30],
        "learning_rate": [0.002],
    })

    # Other seeds which are chosen for different operations
    seeds_: Optional[dict] = None

    logger: ClassVar = logging.getLogger("Experiment")

    def run(self):
        """Run hyperparameter tuning for the DeepFingerprinting classifier
        and return the prediction probabilities for the best chosen
        classifier.
        """
        # Generate and set random seeds
        rng = np.random.default_rng(self.seed)
        self.seeds_ = {
            "train_test": rng.integers(MAX_RAND_SEED),
            "kfold_shuffle": rng.integers(MAX_RAND_SEED),
            "tensorflow": rng.integers(MAX_RAND_SEED),
        }
        tensorflow.random.set_seed(self.seeds_["tensorflow"])

        self.logger.info("Running %s", self)
        start = time.perf_counter()

        # Load the dataset
        X, y = self.load_dataset()
        n_classes = len(np.unique(y))
        self.logger.info("Dataset shape=%s, n_classes=%d", X.shape, n_classes)

        # Generate our training and final testing set
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, shuffle=True,
            random_state=self.seeds_["train_test"]
        )

        if self.hyperparams == "tune":
            self.logger.info("Performing hyperparameter tuning ...")
            # Tune other hyperparameters and fit the final estimator
            classifier = self.tune_hyperparameters(
                x_train, y_train, n_classes=n_classes
            )
        else:
            assert isinstance(self.hyperparams, dict)
            n_packets = self.hyperparams.get("n_packets", DEFAULT_N_PACKETS)
            learning_rate = self.hyperparams.get(
                "learning_rate", DEFAULT_LEARNING_RATE
            )
            epochs = self.hyperparams.get("epochs", DEFAULT_EPOCHS)
            self.logger.info(
                "Using n_packets=%s, learning_rate=%.3g, and epochs=%d",
                n_packets, learning_rate, epochs
            )

            x_train = first_n_packets(x_train, n_packets=n_packets)
            x_test = first_n_packets(x_test, n_packets=n_packets)

            classifier = dfnet.DeepFingerprintingClassifier(
                n_classes=n_classes, verbose=min(self.verbose, 1),
                n_features=n_packets, epochs=epochs, learning_rate=learning_rate
            )
            classifier.fit(x_train, y_train)

        # Predict the classes for the test set
        probabilities = classifier.predict_proba(x_test)
        self.logger.info(
            "Experiment complete in %.2fs.", (time.perf_counter() - start))

        return (probabilities, y_test, classifier.classes_)

    def tune_hyperparameters(self, x_train, y_train, *, n_classes):
        """Perform hyperparameter tuning on the learning rate."""
        assert self.seeds_ is not None, "seeds must be set"
        pipeline = Pipeline([
            ("first_n_packets", FunctionTransformer(first_n_packets)),
            ("dfnet", dfnet.DeepFingerprintingClassifier(
                n_classes=n_classes, verbose=min(self.verbose, 1),
            ))
        ])
        param_grid = [
            {
                "first_n_packets__kw_args": [{"n_packets": n_packets}],
                "dfnet__n_features": [n_packets],
                **{
                    f"dfnet__{key}": values
                    for key, values in self.tuned_parameters.items()
                }
            }
            for n_packets in self.n_packet_parameters
        ]
        cross_validation = StratifiedKFold(
            self.n_folds, shuffle=True,
            random_state=self.seeds_["kfold_shuffle"]
        )

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cross_validation, error_score="raise",
            scoring=make_scorer(rf1_score), verbose=self.verbose, refit=True,
        )
        grid_search.fit(x_train, y_train)

        self.logger.info("hyperparameter results = %s", grid_search.cv_results_)
        self.logger.info("hyperparameter best = %s", grid_search.best_params_)

        return grid_search.best_estimator_

    def load_dataset(self):
        """Load the features and classes from the dataset. Slices the
        packet features to the maximum evaluated.
        """
        max_packets = max(self.n_packet_parameters)
        with h5py.File(self.dataset_path, mode="r") as h5in:
            features = ensure_non_ragged(h5in["sizes"], dimension=max_packets)
            classes = np.asarray(h5in["labels"]["class"][:])
        return features, classes


def first_n_packets(features, *, n_packets: int):
    """Return the first n_packets packets along with the meta features."""
    return features[:, :n_packets]


def rf1_score(y_true, y_pred, *, negative_class=-1, ratio=20):
    """Compute the F1-score using the r-precisions with the specified ratio
    and recall.
    """
    precision = rprecision_score(
        y_true, y_pred, negative_class=negative_class, ratio=ratio,
        # If we're dividing by zero it means there were no true positives and
        # thus recall will be zero and the F1 score below will be zero.
        zero_division=1
    )
    recall = recall_score(y_true, y_pred, negative_class=negative_class)
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    main(**doceasy.doceasy(__doc__, {
        "OUTFILE": doceasy.CsvFile(mode="w", default="-"),
        "DATASET_PATH": Use(Path),
        "--verbose": Use(int),
        "--hyperparams": Or("tune", And(doceasy.Mapping(), {
            doceasy.Optional("n_packets"): Use(int),
            doceasy.Optional("epochs"): Use(int),
            doceasy.Optional("learning_rate"): Use(float),
        }))
    }))
