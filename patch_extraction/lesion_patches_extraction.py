"""
Extract a fixed size patch centered on lesions centroid from a image/mask
dataset in nifti format (as converted by RSPECT_dataset_conversion/RSPECT_to_seg.py).
Save the image patch and its corresponding mask patch as separate nii.gz files
"""

# Imports
import os
import sys
import numpy as np
import SimpleITK as sitk
import json

# Local imports
sys.path.append(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils")
import utils

##################
# Configuration
##################
PATH_TO_INPUT_DATASET = "/data/lmalerba/augmented_RSPECT"
OUTPUT_PATH = "/data/lmalerba/lesions_patches_dataset"

NORMALIZE = True # If set as True, CT normalization will be done and generated dataset will be normalized
PATH_TO_FINGERPRINT = "/data/lmalerba/augmented_RSPECT/dataset_fingerprint.json"

if NORMALIZE == True and PATH_TO_FINGERPRINT == None:
    raise ValueError("For normalization, dataset fingerprint is needed")


PATCH_SIZE_MM = 50.0 # patch size in mm
TARGET_SPACING = (0.8, 0.8, 0.8)
CLASS_LABELS = {1: "acute", 2: "chronic"}

def compute_axis_bounds(center, spacing_axis, patch_size_mm, dim_size):
    """Return (start, size) along one axis so the patch has a constant voxel size,
    shifting it (instead of clipping) to stay inside the volume when possible."""
    size_vox = max(1, round(patch_size_mm / spacing_axis))
    size_vox = min(size_vox, dim_size)

    start = center - size_vox // 2
    start = max(0, min(start, dim_size - size_vox))
    return start, size_vox


def get_lesion_classes(label_sitk, components_sitk, lesion_ids):
    """Majority mask value (1=acute, 2=chronic) found inside each connected component."""
    label_array = sitk.GetArrayFromImage(label_sitk)
    components_array = sitk.GetArrayFromImage(components_sitk)

    lesion_classes = {}
    for lesion_id in lesion_ids:
        component_mask = components_array == lesion_id
        values, counts = np.unique(label_array[component_mask], return_counts=True)
        lesion_classes[lesion_id] = int(values[np.argmax(counts)])
    return lesion_classes


def extract_patches_for_image(path_to_image, path_to_label, images_out_dir, masks_out_dir):
    base_name = os.path.basename(path_to_label).replace(".nii.gz", "")  # <StudyUID>_<SeriesUID>

    image_sitk = sitk.ReadImage(path_to_image)
    label_sitk = sitk.ReadImage(path_to_label)

    image_sitk = utils.resample_to_spacing(image_sitk, TARGET_SPACING, sitk.sitkLinear, default_pixel_value=-1000)
    label_sitk = utils.resample_to_spacing(label_sitk, TARGET_SPACING, sitk.sitkNearestNeighbor, default_pixel_value=0)

    binary_label = sitk.BinaryThreshold(label_sitk, lowerThreshold=1, insideValue=1, outsideValue=0)
    components_sitk = sitk.ConnectedComponent(binary_label)

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(components_sitk)
    lesion_ids = shape_stats.GetLabels()

    if not lesion_ids:
        return 0

    lesion_classes = get_lesion_classes(label_sitk, components_sitk, lesion_ids)

    spacing = image_sitk.GetSpacing()  # (x, y, z) mm per voxel
    volume_size = image_sitk.GetSize()  # (x, y, z)

    # read dataset fingerprint if normalization is needed
    fingerprint = None
    if NORMALIZE:
        with open(PATH_TO_FINGERPRINT, 'r', encoding='utf-8') as f:
            fingerprint = json.load(f)

        image_array = sitk.GetArrayFromImage(image_sitk)

        image_array = utils.ct_normalize(image_array,
                fingerprint["mean"],
                fingerprint["std"],
                fingerprint["percentile_00_5"], 
                fingerprint["percentile_99_5"])

        image_sitk_normalized = sitk.GetImageFromArray(image_array)
        image_sitk_normalized.CopyInformation(image_sitk)
        image_sitk = image_sitk_normalized

    saved_count = 0
    for lesion_id in lesion_ids:
        centroid_physical = shape_stats.GetCentroid(lesion_id)
        center_index = label_sitk.TransformPhysicalPointToIndex(centroid_physical)  # (x, y, z)

        start = [0, 0, 0]
        size = [0, 0, 0]
        for axis in range(3):
            start[axis], size[axis] = compute_axis_bounds(
                center_index[axis], spacing[axis], PATCH_SIZE_MM, volume_size[axis]
            )

        

        image_patch = sitk.RegionOfInterest(image_sitk, size=size, index=start)
        label_patch = sitk.RegionOfInterest(label_sitk, size=size, index=start)

        class_name = CLASS_LABELS.get(lesion_classes[lesion_id], f"class{lesion_classes[lesion_id]}")
        patch_name = f"{base_name}_{class_name}_{lesion_id}.nii.gz"

        sitk.WriteImage(image_patch, os.path.join(images_out_dir, patch_name))
        sitk.WriteImage(label_patch, os.path.join(masks_out_dir, patch_name))
        saved_count += 1

    return saved_count

if __name__ == "__main__":
    images_out_dir = os.path.join(OUTPUT_PATH, "images")
    masks_out_dir = os.path.join(OUTPUT_PATH, "masks")

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(masks_out_dir, exist_ok=True)

    assert not os.listdir(os.path.join(images_out_dir)), "Output folders are not empty. Please empty them before running this script."
    assert not os.listdir(os.path.join(masks_out_dir)), "Output folders are not empty. Please empty them before running this script."

    total_patches = 0
    total_images = 0
    
    labels_dir = os.path.join(PATH_TO_INPUT_DATASET, "masks")
    images_dir = os.path.join(PATH_TO_INPUT_DATASET, "images")

    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".nii.gz"):
            continue

        path_to_label = os.path.join(labels_dir, fname)
        path_to_image = os.path.join(images_dir, fname)

        # print(path_to_image)
        if not os.path.isfile(path_to_image):
            print(f"Skipping {fname}: matching image not found")
            continue

        saved_count = extract_patches_for_image(path_to_image, path_to_label, images_out_dir, masks_out_dir)
        total_images += 1
        total_patches += saved_count
        print(f"[{total_images}/{len(os.listdir(labels_dir))}] {fname}: {saved_count} patch(es) saved")

    print(f"Done. Processed {total_images} image(s), saved {total_patches} patch(es).")
