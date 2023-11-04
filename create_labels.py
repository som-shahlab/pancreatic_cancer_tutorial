import argparse
import collections
import csv
import datetime
import json
import os
import pickle
import random
from typing import Any, Callable, List, Optional, Set, Tuple

import femr.datasets
import femr.labelers
import femr.labelers.omop
import numpy as np


class PancreaticCancerLabeler(femr.labelers.TimeHorizonEventLabeler):
    def __init__(self, ontology, time_horizon):
        self.time_horizon = time_horizon
        self.cancer_codes = list(
            femr.labelers.omop.map_omop_concept_codes_to_femr_codes(
                #ontology, {"SNOMED/57054005"}, is_ontology_expansion=True
                ontology, {"SNOMED/363418001"}, is_ontology_expansion=True
            )
        )

        super().__init__()

    def get_prediction_times(self, patient):
        outpatient_visit_times = set()

        first_non_birth = None

        birth = patient.events[0].start.date()

        first_pad_code = None
        last_prediction = None

        for event in patient.events:
            if event.start.date() != birth and first_non_birth is None and event.start.year >= 2014: # year filter to require modern data
                first_non_birth = event.start

            if first_pad_code is None and event.code in self.cancer_codes:
                first_pad_code = event.start

            if (
                event.code == "Visit/OP"
                and first_non_birth is not None
                and (event.start - first_non_birth).days > 365
                and (last_prediction is None or (event.start - last_prediction).days > 365)
            ):
                last_prediction = event.start
                outpatient_visit_times.add(event.start) # - datetime.timedelta(days=1))

        return sorted([a for a in outpatient_visit_times if first_pad_code is None or a < first_pad_code])

    def get_time_horizon(self):
        return self.time_horizon

    def allow_same_time_labels(self):
        return False

    def get_outcome_times(self, patient):
        outcome_times = set()

        for event in patient.events:
            if event.code in self.cancer_codes:
                outcome_times.add(event.start)

        return sorted(list(outcome_times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FEMR labeling")
    parser.add_argument(
        "--path_to_database", required=True, type=str, help="Path to FEMR database"
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
    parser.add_argument(
        "--use_sample",
        action=argparse.BooleanOptionalAction,
        help="Label a sample instead of the whole database",
        default=False,
    )

    args = parser.parse_args()

    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_OUTPUT_DIR = args.path_to_output_dir
    NUM_THREADS: int = args.num_threads

    # Logging
    print(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    print(f"Saving output to: {PATH_TO_OUTPUT_DIR}")
    print(f"# of threads: {NUM_THREADS}")

    # create directories to save files
    PATH_TO_SAVE_LABELED_PATIENTS: str = os.path.join(
        PATH_TO_OUTPUT_DIR, "labeled_patients.csv"
    )
    PATH_TO_SAVE_SUBSAMPLE_LABELED_PATIENTS: str = os.path.join(
        PATH_TO_OUTPUT_DIR, "subsample_labeled_patients.csv"
    )
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Load PatientDatabase + Ontology
    print(f"Start | Load PatientDatabase")
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()
    print(f"Finish | Load PatientDatabase")

    labeler = PancreaticCancerLabeler(
        ontology,
        femr.labelers.TimeHorizon(
            datetime.timedelta(minutes=1), datetime.timedelta(days=365)
        ),
    )

    print(f"Start | Label")

    if args.use_sample:
        num_patients = 10_000
    else:
        num_patients = None

    labeled_patients = labeler.apply(
        path_to_patient_database=PATH_TO_PATIENT_DATABASE,
        num_threads=NUM_THREADS,
        num_patients=num_patients,
    )

    labeled_patients.save(PATH_TO_SAVE_LABELED_PATIENTS)
    print("Finish | Label patients")
    print(
        "LabeledPatient stats:\n"
        f"Total # of patients = {labeled_patients.get_num_patients()}\n"
        f"Total # of labels = {labeled_patients.get_num_labels()}\n"
        f"Total # of positives = {np.sum(labeled_patients.as_numpy_arrays()[1])}"
    )

    subsampled = femr.labelers.subsample_to_prevalence(labeled_patients, 0.2, seed=97)

    subsampled.save(PATH_TO_SAVE_SUBSAMPLE_LABELED_PATIENTS)
    print("Finish | Subsampled label patients")
    print(
        "Subsampled LabeledPatient stats:\n"
        f"Total # of patients = {subsampled.get_num_patients()}\n"
        f"Total # of labels = {subsampled.get_num_labels()}\n"
        f"Total # of positives = {np.sum(subsampled.as_numpy_arrays()[1])}"
    )

    print("Done!")

    patients = list(subsampled.keys())

    train_person_ids = set()

    with open(os.path.join(args.path_to_database, '..', 'train_person_ids.csv')) as f:
        reader = csv.DictReader(f)
        for line in reader:
            train_person_ids.add(int(line['person_id']))

    train_set = []
    valid_set = []
    test_set = []

    random.seed(34234)

    for patient in patients:
        if patient in train_person_ids:
            train_set.append(patient)
        else:
            if random.random() < 0.5:
                test_set.append(patient)
            else:
                valid_set.append(patient)

    np.save(os.path.join(PATH_TO_OUTPUT_DIR, 'train_patients.npy'), train_set)
    np.save(os.path.join(PATH_TO_OUTPUT_DIR, 'valid_patients.npy'), valid_set)
    np.save(os.path.join(PATH_TO_OUTPUT_DIR, 'test_patients.npy'), test_set)
