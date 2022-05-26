#!/usr/bin/env python3
"""Usage: evaluate_tuned_kfp.py [options] FEATURES_PATH [OUTFILE]

Perform hyperparameter turning on the k-FP classifier.

The features are read from the CSV file located at FEATURES_PATH and the
probability predictions are written in CSV format to OUTFILE (defaults to
stdout).

Options:
    --cv-results-path <path>
        Write the cross-validation results to a csv at <path>.
    --feature-importance <path>
        Save feature importances to specified file.
    --verbose <val>
        Set the verbosity of the gridsearch and classifier. A value of
        0 disables output whereas 3 produces all output [default: 0].
    --n-jobs <n>
        Use at most <n> jobs for training multiple instances of the random
        forest during the hyperparameter search [default: 1].
"""
# pylint: disable=too-many-instance-attributes
import time
import logging
import dataclasses
from pathlib import Path
from typing import Optional, ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from lab.metrics import rprecision_score, recall_score
from lab.classifiers import kfingerprinting

from common import doceasy
from common.doceasy import Use, Or

IDEAL_N_JOBS_KFP: int = 4
MAX_RAND_SEED: int = 10_000


def main(outfile, **kwargs):
    """Create and run the experiment and write the CSV results to outfile."""
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
        level=logging.INFO
    )

    (probabilities, y_true, classes) = Experiment(**kwargs).run()

    y_pred = classes[np.argmax(probabilities, axis=1)]
    score = rf1_score(y_true, y_pred)
    logging.info("r_20 f1-score = %.4g", score)

    outfile.writerow(["y_true"] + list(classes))
    outfile.writerows(
        np.hstack((np.reshape(y_true, (-1, 1)), probabilities)))


@dataclasses.dataclass
class Experiment:
    """An experiment to evalute hyperparameter tuned k-FP classifier.
    """
    # The path to the dataset of features
    features_path: Path

    # The output path for the cross-validation results
    cv_results_path: Optional[Path]

    # The output path for feature importances
    feature_importance: Optional[Path]

    # The fraction of samples to use for testing the final model
    test_size: float = 0.2

    # Number of folds used in the stratified k-fold cross validation
    n_folds: int = 3

    # Level of debugging output from sklearn and tensorflow
    verbose: int = 0

    # Total number of jobs to use in both the gridsearch and RF classifier
    n_jobs: int = 1

    # Random seed for the experiment
    seed: int = 3410

    # Hyperparameters to search
    tuned_parameters: dict = dataclasses.field(default_factory=lambda: {
        "n_neighbours": [2, 3, 6],
        "forest__n_estimators": [100, 150, 200, 250],
        "forest__max_features": ["sqrt", "log2", 20, 30],
        "forest__oob_score": [True, False],
        "forest__max_samples": [None, 0.5, 0.75, 0.9],
    })

    # Other seeds which are chosen for different operations
    seeds_: Optional[dict] = None

    # Dervied number of jobs
    n_jobs_: Optional[dict] = None

    logger: ClassVar = logging.getLogger("Experiment")

    def run(self):
        """Run hyperparameter tuning for the VarCNN classifier and return
        the prediction probabilities for the best chosen classifier.
        """
        # Generate and set random seeds
        rng = np.random.default_rng(self.seed)
        self.seeds_ = {
            "train_test": rng.integers(MAX_RAND_SEED),
            "kfold_shuffle": rng.integers(MAX_RAND_SEED),
            "kfp": rng.integers(MAX_RAND_SEED),
        }

        self.logger.info("Running %s", self)
        start = time.perf_counter()

        # Load the dataset
        X, y = self.load_dataset()
        self.logger.info(
            "Dataset shape=%s, n_classes=%d", X.shape, len(np.unique(y))
        )

        # Generate our training and final testing set
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, shuffle=True,
            random_state=self.seeds_["train_test"]
        )

        # Tune other hyperparameters and fit the final estimator
        (results, classifier) = self.tune_hyperparameters(x_train, y_train)
        if self.cv_results_path is not None:
            pd.DataFrame(results).to_csv(
                self.cv_results_path, header=True, index=False
            )

        if self.feature_importance is not None:
            pd.DataFrame({
                "feature": kfingerprinting.ALL_DEFAULT_FEATURES,
                "weight": classifier.forest_.feature_importances_
            }).to_csv(self.feature_importance, header=True, index=False)

        # Predict the classes for the test set
        probabilities = classifier.predict_proba(x_test)
        self.logger.info(
            "Experiment complete in %.2fs.", (time.perf_counter() - start))

        return (probabilities, y_test, classifier.classes_)

    def tune_hyperparameters(self, x_train, y_train):
        """Perform hyperparameter tuning."""
        assert self.seeds_ is not None, "seeds must be set"

        # Determine the number of jobs to use for the grid search and for kFP
        self.n_jobs_ = {"tune": 1, "kfp": self.n_jobs}
        if self.n_jobs > IDEAL_N_JOBS_KFP:
            self.n_jobs_["tune"] = max(1, self.n_jobs // IDEAL_N_JOBS_KFP)
            self.n_jobs_["kfp"] = IDEAL_N_JOBS_KFP
        self.logger.info("Using jobs: %s", self.n_jobs_)

        estimator = kfingerprinting.KFingerprintingClassifier(
            unknown_label=-1, random_state=self.seeds_["kfp"],
            n_jobs=self.n_jobs_["kfp"]
        )
        cross_validation = StratifiedKFold(
            self.n_folds, shuffle=True,
            random_state=self.seeds_["kfold_shuffle"]
        )

        grid_search = GridSearchCV(
            estimator, self.tuned_parameters, cv=cross_validation,
            error_score="raise", scoring=make_scorer(rf1_score),
            verbose=self.verbose, refit=True, n_jobs=self.n_jobs_["tune"],
        )
        grid_search.fit(x_train, y_train)

        self.logger.info("hyperparameter best = %s", grid_search.best_params_)

        return (grid_search.cv_results_, grid_search.best_estimator_)

    def load_dataset(self):
        """Load the features and classes from the dataset."""
        frame = pd.read_csv(self.features_path)
        assert frame.columns.get_loc("y_true") == 0, "y_true not first column?"

        classes = frame.iloc[:, 0].to_numpy()
        features = frame.iloc[:, 1:].to_numpy()

        return features, classes


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
        "FEATURES_PATH": Use(Path),
        "--cv-results-path": Or(None, Use(Path)),
        "--feature-importance": Or(None, Use(Path)),
        "--n-jobs": Use(int),
        "--verbose": Use(int),
    }))
