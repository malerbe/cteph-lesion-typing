"""
Create k patient-level stratified cross-validation splits from the patch dataset produced
by patch_extraction/lesion_patches_extraction.py.

All patches from the same suid (patient) are always assigned to the same fold, so no patient
ever appears in both train and validation for a given fold (no data-leakage)

This script is original because it is a deterministic splitting algorithm. The goal is to balance
the amount of patches from the minority class (here it is chronic):

Patients are processed from the heaviest (with more labels from minority class) to lightest.

author: Louca Malerba
"""

# Imports
import os
import json
import logging
import sys



##################
# Configuration
##################
PATH_TO_PATCH_DATASET = "/data/lmalerba/lesions_patches_dataset" # output of lesion_patches_extraction.py
OUTPUT_FILE = "/data/lmalerba/lesions_patches_dataset/cv_splits_3_0.2.json"

N_SPLITS = 3
TEST_RATIO = 0.2 # keep part of the dataset for a test split
CLASS_TO_IDX = {"acute": 0, "chronic": 1}
MINORITY_CLASS = "chronic"

##################
# Splitting
##################

def make_balanced_patient_splits(records, fold_weights):
    """
    Returns a list of n_splits lists of filenames builts by assigning whole patients to folds
    so that the monority class and the overall patch count both end up as evenly spread
    accross folds as possible, without ever splitting a patient into two folds
    """

    # 0. init
    assert sum(fold_weights) - 1.0 < 1e-6, "fold_weights must sum to 1"
    n_folds = len(fold_weights)


    # 1. Aggregate to patient level
    patient_files, patient_minority_count, patient_total_count = {}, {}, {}
    for filename, suid, class_name in records:
        if not suid in patient_files:
            patient_files[suid] = []
            patient_total_count[suid] = 0
            patient_minority_count[suid] = 0

        patient_files[suid].append(filename)
        patient_total_count[suid] += 1
        if class_name == MINORITY_CLASS:
            patient_minority_count[suid] += 1

    patients = list(patient_files.keys())

    total_minority = sum(patient_minority_count.values())
    total_patches = sum(patient_total_count.values())

    # 2. Compute what a "fair share" should be in the ideal case
    eps = 1e-9
    target_minority_per_fold = [max(total_minority*w, eps) for w in fold_weights]
    target_total_per_fold = [max(total_patches*w, eps) for w in fold_weights]

    # 3. Make splits
    # Process patients with more minority class first:
    patients.sort(key=lambda p: (patient_minority_count[p], patient_total_count[p], p), reverse=True)

    fold_minority_count = [0 for _ in range(n_folds)]
    fold_total_count = [0 for _ in range(n_folds)]
    folds = [[] for _ in range(n_folds)]

    for p in patients:
        # From all folds, select the emptiest one
        fold_id = min(range(n_folds),
                      key= lambda k:fold_minority_count[k]/target_minority_per_fold[k]
                      + fold_total_count[k]/target_total_per_fold[k],
                      )
        folds[fold_id].extend(patient_files[p])
        fold_minority_count[fold_id] += patient_minority_count[p]
        fold_total_count[fold_id] += patient_total_count[p]

    return folds

if __name__ == "__main__":
    assert 0 < TEST_RATIO < 1
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    # List patches
    images_dir = os.path.join(PATH_TO_PATCH_DATASET, "images")
    logging.info(f"listing patches in {images_dir}")

    records = []
    for filename in sorted(os.listdir(images_dir)):
        parts = filename.replace(".nii.gz", "").split("_")
        suid = parts[0]
        class_name = parts[-2]
        if class_name not in CLASS_TO_IDX:
            raise ValueError(f"Unknown class {class_name}")

        records.append((filename, suid, class_name))

    if not records:
        raise ValueError("no patches found")
    else:
        logging.info(f"Found {len(records)} samples accross {len({r[1] for r in records})} patients")

    # Make folds
    if TEST_RATIO > 0:
        cv_weights = (1.0 - TEST_RATIO) / N_SPLITS
        all_splits = make_balanced_patient_splits(records, [TEST_RATIO] + [cv_weights] * N_SPLITS)
        test_files, folds = all_splits[0], all_splits[1:]
    else:
        folds = make_balanced_patient_splits(records, [1.0 / N_SPLITS] * N_SPLITS)
        test_files = []

    # Save folds
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"n_splits": N_SPLITS,
                   "test_ratio": TEST_RATIO,
                   "class_to_idx": CLASS_TO_IDX,
                   "test": test_files,
                   "folds": folds}, f, indent=2)

    logging.info(f"Saved {N_SPLITS} splits to {OUTPUT_FILE}")

    logging.info("SUMMARIZE:")
    class_by_filename = {r[0]: r[2] for r in records}
    suid_by_filename = {r[0]: r[1] for r in records}

    if TEST_RATIO > 0:
        for i, fold in enumerate([test_files]):
            n_patients = len({suid_by_filename[f] for f in fold})
            n_minority = sum(1 for f in fold if class_by_filename[f] == MINORITY_CLASS)
            logging.info(f" TEST fold: {len(fold)} patches, {n_patients} patients, {n_minority} {MINORITY_CLASS}")

    for i, fold in enumerate(folds):
        n_patients = len({suid_by_filename[f] for f in fold})
        n_minority = sum(1 for f in fold if class_by_filename[f] == MINORITY_CLASS)
        logging.info(f" fold {i}: {len(fold)} patches, {n_patients} patients, {n_minority} {MINORITY_CLASS}")
                

