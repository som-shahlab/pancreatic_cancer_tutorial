import argparse
import femr.labelers
import pickle
import numpy as np
import os
import sklearn.linear_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr labeling")
    parser.add_argument(
        "--path_to_database", required=True, type=str, help="Path to femr database"
    )
    parser.add_argument(
        "--path_to_output_dir", required=True, type=str, help="Path to save labeles"
    )
    parser.add_argument(
        "--path_to_fm", required=True, type=str, help="Path to the foundation model"
    )

    args = parser.parse_args()

    labels = os.path.join(args.path_to_output_dir, "subsample_labeled_patients.csv")
    fm_name = os.path.basename(args.path_to_fm)
    print("Running: ", fm_name)
    fm_reprs_path = os.path.join(args.path_to_output_dir, fm_name + "_reprs.pkl")

    command = f"femr_compute_representations --data_path {args.path_to_database} --model_path {args.path_to_fm} --prediction_times_path {labels} {fm_reprs_path} --batch_size {1<<14}"
    os.system(command)
    
    labeled_patients = femr.labelers.load_labeled_patients(
        os.path.join(args.path_to_output_dir, "subsample_labeled_patients.csv")
    )

    label_pids, label_values, label_prediction_times = labeled_patients.as_numpy_arrays()
    ind = np.lexsort((label_pids, label_prediction_times))
    label_pids = label_pids[ind]
    label_values = label_values[ind]
    label_prediction_times = label_prediction_times[ind]

    with open(fm_reprs_path, 'rb') as f:
        fm_reprs = pickle.load(f)

    ind = np.lexsort((fm_reprs['patient_ids'], fm_reprs['prediction_times']))
    fm_reprs['patient_ids'] = fm_reprs['patient_ids'][ind]
    fm_reprs['prediction_times'] = fm_reprs['prediction_times'][ind]
    fm_reprs['representations'] = fm_reprs['representations'][ind, :]

    train_patients = np.load(os.path.join(args.path_to_output_dir, 'train_patients.npy'))
    valid_patients = np.load(os.path.join(args.path_to_output_dir, 'valid_patients.npy'))
    test_patients = np.load(os.path.join(args.path_to_output_dir, 'test_patients.npy'))

    test_mask = np.isin(label_pids, test_patients)
    train_val_mask = ~test_mask


    model = sklearn.linear_model.LogisticRegressionCV()
    model.fit(fm_reprs['representations'][train_val_mask, :], label_values[train_val_mask])

    with open(
        os.path.join(args.path_to_output_dir, f"{fm_name}_model.pkl"), "wb"
    ) as f:
        pickle.dump(model, f)

    proba = model.predict_proba(fm_reprs['representations'])[:, 1]

    with open(
        os.path.join(args.path_to_output_dir, f"{fm_name}_predictions.pkl"), "wb"
    ) as f:
        pickle.dump([proba, label_pids, label_values, label_prediction_times], f)
        
    y_test_proba = proba[test_mask]

    test_auroc = sklearn.metrics.roc_auc_score(label_values[test_mask], y_test_proba)
    print(f"Test AUROC: {test_auroc}")

    
