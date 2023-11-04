import argparse
import collections
import csv
import os
import pickle
from collections import Counter
from typing import List, Optional, Tuple

import femr
import femr.datasets
import lightgbm as ltb
import numpy as np
import scipy
import sklearn.linear_model
from scipy.sparse import issparse
from sklearn import metrics
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     PredefinedSplit)

XGB_PARAMS = {
    "max_depth": [3, 6, -1],
    "learning_rate": [0.02, 0.1, 0.5],
    "num_leaves": [10, 25, 100],
    "force_row_wise": [True],
    "num_threads": [4],
}

LR_PARAMS = {
    "C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6],
    "penalty": ["l2"],
}


def tune_hyperparams(
    X_train, y_train, X_val, y_val, model, params, num_threads: int = 1
):
    # In `test_fold`, -1 indicates that the corresponding sample is used for training, and a value >=0 indicates the test set.
    # We use `PredefinedSplit` to specify our custom validation split
    X = scipy.sparse.vstack([X_train, X_val])
    y = np.concatenate((y_train, y_val), axis=0)
    test_fold = -np.ones(X.shape[0])
    test_fold[X_train.shape[0] :] = 1
    clf = GridSearchCV(
        model,
        params,
        n_jobs=27,
        verbose=1,
        cv=PredefinedSplit(test_fold=test_fold),
        refit=True,
        scoring="roc_auc",
    )
    clf.fit(X, y)
    return clf


def main(args):
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE)

    print(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    print(f"Saving output to: {args.path_to_output_dir}")

    with open(os.path.join(args.path_to_output_dir, "features.pkl"), "rb") as f:
        features, patient_ids, label_values, label_times = pickle.load(f)

    train_patients = np.load(os.path.join(args.path_to_output_dir, 'train_patients.npy'))
    valid_patients = np.load(os.path.join(args.path_to_output_dir, 'valid_patients.npy'))
    test_patients = np.load(os.path.join(args.path_to_output_dir, 'test_patients.npy'))

    train_mask = np.isin(patient_ids, train_patients)
    valid_mask = np.isin(patient_ids, valid_patients)
    test_mask = np.isin(patient_ids, test_patients)

    print(f"Num train: {sum(train_mask)}")
    print(f"Num valid: {sum(valid_mask)}")
    print(f"Num test: {sum(test_mask)}")

    X_train = features[train_mask, :]
    X_valid = features[valid_mask, :]
    X_test = features[test_mask, :]

    y_train = label_values[train_mask]
    y_valid = label_values[valid_mask]
    y_test = label_values[test_mask]

    os.makedirs(args.path_to_output_dir, exist_ok=True)

    for model_name in ["logistic", "gbm"]:
        print(f"Working on {model_name}")

        if model_name == "gbm":
            model = tune_hyperparams(
                X_train,
                y_train,
                X_valid,
                y_valid,
                ltb.LGBMClassifier(),
                XGB_PARAMS,
                num_threads=args.num_threads,
            )
        elif model_name == "logistic":
            model = tune_hyperparams(
                X_train,
                y_train,
                X_valid,
                y_valid,
                sklearn.linear_model.LogisticRegression(),
                LR_PARAMS,
                num_threads=args.num_threads,
            )

        with open(
            os.path.join(args.path_to_output_dir, f"{model_name}_model.pkl"), "wb"
        ) as f:
            pickle.dump(model, f)

        proba = model.predict_proba(features)[:, 1]

        with open(
            os.path.join(args.path_to_output_dir, f"{model_name}_predictions.pkl"), "wb"
        ) as f:
            pickle.dump([proba, patient_ids, label_values, label_times], f)

        y_train_proba = proba[train_mask]
        y_valid_proba = proba[valid_mask]
        y_test_proba = proba[test_mask]

        train_auroc = metrics.roc_auc_score(y_train, y_train_proba)
        val_auroc = metrics.roc_auc_score(y_valid, y_valid_proba)
        test_auroc = metrics.roc_auc_score(y_test, y_test_proba)
        print(f"Train AUROC: {train_auroc}")
        print(f"Val AUROC: {val_auroc}")
        print(f"Test AUROC: {test_auroc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baselines logistic regression and lightgbm models"
    )
    parser.add_argument(
        "--path_to_database", required=True, type=str, help="Path to femr database"
    )
    parser.add_argument(
        "--path_to_output_dir",
        required=True,
        type=str,
        help="Path to save labeles and featurizers",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    args = parser.parse_args()
    main(args)
