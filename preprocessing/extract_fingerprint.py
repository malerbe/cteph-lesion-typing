"""
Extract dataset fingerprint from whole images/masks niftis for CT normalization

Dataset fingerprint corresponds to get normalization parameters for the whole dataset
"""
# Standard imports
import os
import logging
import numpy as np
import sys

# Imports
import SimpleITK as sitk
import json
from tqdm import tqdm

##################
# Configuration
##################
INPUT_DATASET = "/data/lmalerba/augmented_RSPECT"
OUTPUT_FILE = "/data/lmalerba/augmented_RSPECT/dataset_fingerprint.json"

##################
# Fingerprint extraction
##################

def sample_foreground_intensities(img_array, mask_array, num_samples, seed = 24):
    """
    foreground: segmentation > 0

    For memory reasons, it is not possible for big datasets to take all foreground voxels.
    For this reason, num_samples voxels are sampled randomly from all foreground voxels in
    the image
    """

    assert img_array.shape == mask_array.shape

    random_state = np.random.RandomState(seed)
    foreground_mask = mask_array > 0
    foreground_voxels = img_array[foreground_mask]
    foreground_size = len(foreground_voxels)

    if foreground_size == 0:
        return np.array([])

    return random_state.choice(foreground_voxels, num_samples, replace=True)

def compute_ct_intensity_properties(pooled_foreground_intensities):
    """
    Turns the pooled foreground intensities concatenated accross every case of the dataset
    into the statistics CT normalization needs
    """
    percentiles = np.array((0.5, 50.0, 99.5))
    percentile_00_5, median, percentile_99_5 = np.percentile(pooled_foreground_intensities, percentiles)

    return {
        "mean": float(np.mean(pooled_foreground_intensities)),
        "median": float(median),
        "std": float(np.std(pooled_foreground_intensities)),
        "min": float(np.min(pooled_foreground_intensities)),
        "max": float(np.max(pooled_foreground_intensities)),
        "percentile_00_5": float(percentile_00_5),
        "percentile_99_5": float(percentile_99_5)
    }


def extract_fingerprint(dataset_directory, num_foreground_voxels_for_intensitystats = 10e7):
    """
    dataset_directory (str): path to a dataset formatted as two folder (images/masks) containing
        images and corresponding masks in nifti format 
    """

    # Read input folder and get a list of images/masks
    img_names = os.listdir(os.path.join(INPUT_DATASET, 'images'))
    masks_names = os.listdir(os.path.join(INPUT_DATASET, 'masks'))

    assert len(img_names) == len(masks_names), "Number of images and masks should be the same"

    per_case_samples = []

    num_samples_per_case = int(num_foreground_voxels_for_intensitystats // len(img_names))

    for img_name, mask_name in tqdm(zip(img_names, masks_names), total=len(img_names)):
        # Read nifti
        img_nifti = sitk.ReadImage(os.path.join(INPUT_DATASET, 'images', img_name))
        mask_nifti = sitk.ReadImage(os.path.join(INPUT_DATASET, 'masks', mask_name))

        img_array = sitk.GetArrayFromImage(img_nifti)
        mask_array = sitk.GetArrayFromImage(mask_nifti)

        # Extract foreground intensities
        per_case_samples.append(sample_foreground_intensities(img_array, mask_array, num_samples_per_case))

    # Keep only the non empty samples, and concatenate them for properties extraction
    non_empty = [s for s in per_case_samples if len(s) > 0]

    if not non_empty:
        raise ValueError("No foreground voxels found in any case")

    pooled = np.concatenate(non_empty)

    return compute_ct_intensity_properties(pooled)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    logging.info("Starting dataset fingerprint extraction...")
    # Extract fingerprint
    fingerprint = extract_fingerprint(INPUT_DATASET)

    # Print values and save
    logging.info("Fingerprint extracted. Values:")
    logging.info(fingerprint)

    logging.info(f"Saving fingerprint to {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(fingerprint, f)




