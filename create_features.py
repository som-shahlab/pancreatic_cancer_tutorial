import argparse
import collections
import csv
import datetime
import json
import os
import pickle
import random
from typing import Any, Callable, List, Optional, Set, Tuple

import femr
import femr.featurizers
import femr.labelers
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr labeling")
    parser.add_argument(
        "--path_to_database", required=True, type=str, help="Path to femr database"
    )
    parser.add_argument(
        "--path_to_output_dir", required=True, type=str, help="Path to save labeles"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )

    args = parser.parse_args()

    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_OUTPUT_DIR = args.path_to_output_dir
    NUM_THREADS: int = args.num_threads

    print(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    print(f"Saving output to: {PATH_TO_OUTPUT_DIR}")
    print(f"# of threads: {NUM_THREADS}")

    labeled_patients = femr.labelers.load_labeled_patients(
        os.path.join(PATH_TO_OUTPUT_DIR, "subsample_labeled_patients.csv")
    )

    # Lets use both age and count featurizer
    age = femr.featurizers.AgeFeaturizer()
    count = femr.featurizers.CountFeaturizer(is_ontology_expansion=True)
    featurizer_age_count = femr.featurizers.FeaturizerList([age, count])

    # Preprocessing the featurizers, which includes processes such as normalizing age.
    print("Start | Preprocess featurizers")
    featurizer_age_count.preprocess_featurizers(
        PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS
    )
    with open(os.path.join(PATH_TO_OUTPUT_DIR, "featurizers.pkl"), "wb") as f:
        pickle.dump(featurizer_age_count, f)

    print("Finish | Preprocess featurizers")

    print("Start | Featurize patients")
    results = featurizer_age_count.featurize(
        PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS
    )
    with open(os.path.join(PATH_TO_OUTPUT_DIR, "features.pkl"), "wb") as f:
        pickle.dump(results, f)

    print("Finish | Featurize patients")
    feature_matrix, patient_ids, label_values, label_times = (
        results[0],
        results[1],
        results[2],
        results[3],
    )
    label_set, counts_per_label = np.unique(label_values, return_counts=True)
    print(
        "FeaturizedPatient stats:\n"
        f"feature_matrix={repr(feature_matrix)}\n"
        f"patient_ids={repr(patient_ids)}\n"
        f"label_values={repr(label_values)}\n"
        f"label_set={repr(label_set)}\n"
        f"counts_per_label={repr(counts_per_label)}\n"
        f"label_times={repr(label_times)}"
    )

    with open(os.path.join(PATH_TO_OUTPUT_DIR, "features_done.txt"), "w") as f:
        f.write("done")

    print("Done!")
