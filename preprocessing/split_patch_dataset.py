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
OUTPUT_FILE = "/data/lmalerba/lesions_patches_dataset/cv_splits_3.json"

N_SPLITS = 3
CLASS_TO_IDX = {"acute": 0, "chronic": 1}
MINORITY_CLASS = "chronic"

##################
# Splitting
##################

def make_balanced_patient_splits(records, n_splits=N_SPLITS):
    """
    Returns a list of n_splits lists of filenames builts by assigning whole patients to folds
    so that the monority class and the overall patch count both end up as evenly spread
    accross folds as possible, without ever splitting a patient into two folds
    """

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
    target_minority_per_fold = total_minority / n_splits
    target_total_per_fold = total_patches / n_splits

    # 3. Make splits
    # Process patients with more minority class first:
    patients.sort(key=lambda p: (patient_minority_count[p], patient_total_count[p]), reverse=True)

    fold_minority_count = [0 for _ in range(n_splits)]
    fold_total_count = [0 for _ in range(n_splits)]
    folds = [[] for _ in range(n_splits)]

    for p in patients:
        # From all folds, select the emptiest one
        fold_id = min(range(n_splits),
                      key= lambda k:fold_minority_count[k]/target_minority_per_fold
                      + fold_total_count[k]/target_total_per_fold,
                      )
        folds[fold_id].extend(patient_files[p])
        fold_minority_count[fold_id] += patient_minority_count[p]
        fold_total_count[fold_id] += patient_total_count[p]

    return folds

if __name__ == "__main__":
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
    folds = make_balanced_patient_splits(records)

    # Save folds
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"n_splits": N_SPLITS,
                   "class_to_idx": CLASS_TO_IDX,
                   "folds": folds}, f, indent=2)

    logging.info(f"Saved {N_SPLITS} splits to {OUTPUT_FILE}")

    logging.info("SUMMARIZE:")
    class_by_filename = {r[0]: r[2] for r in records}
    suid_by_filename = {r[0]: r[1] for r in records}

    for i, fold in enumerate(folds):
        n_patients = len({suid_by_filename[f] for f in fold})
        n_minority = sum(1 for f in fold if class_by_filename[f] == MINORITY_CLASS)
        logging.info(f" fold {i}: {len(fold)} patches, {n_patients} patients, {n_minority} {MINORITY_CLASS}")
                

