"""
RSPECT annotations are given as 2D bboxes on successive axial slices.

This script bridges the augmented RSPECT annotations and the RSNA RSPECT
dataset volumes to provide images and segmentation masks as niftis
"""

# Imports
import os
import sys
import warnings
import json
import pandas as pd
from scipy.ndimage import label
import numpy as np
import random
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

# Local imports
sys.path.append(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils")
import utils

##################
# Configuration
##################
PATH_TO_DATASET_IMGS = "/data/lmalerba/RSPECT_dataset/images" # path to dicom folders from the RSNA dataset (at least RSPECT reannotated subset)
PATH_TO_DATASET_AUGMENTED_LABELS = "/data/lmalerba/augmented_RSPECT/augmented_rspect.csv" 
OUTPUT_DIR = "/data/lmalerba/augmented_RSPECT"

# Make folders if necessary:
if not os.path.exists(os.path.join(OUTPUT_DIR, "images")):
    os.makedirs(os.path.join(OUTPUT_DIR, "images"))
if not os.path.exists(os.path.join(OUTPUT_DIR, "masks")):
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"))


##################
# Conversion script
##################
# List images paths:
augmented_dataset = pd.read_csv(PATH_TO_DATASET_AUGMENTED_LABELS)
couple_list = set(zip(augmented_dataset["StudyInstanceUID"], augmented_dataset["SeriesInstanceUID"]))
_list = [i for i in range(len(list(couple_list)))]
random.shuffle(_list)

for i in tqdm(range(len(_list))):
    k = _list[i]

    couple = list(couple_list)[k]

    # 1. Extract information from couple:
    StudyInstanceUID, SeriesInstanceUID = couple[0], couple[1]
    dcm_folder_path = os.path.join(PATH_TO_DATASET_IMGS, os.path.join(couple[0], couple[1]))

    # 2. Get image array and informations:
    img_array, res, vol_infos = utils.load_dicom_series(dcm_folder_path)
    # Adjust padding value:
    img_array[img_array < -1024] = -1024

    # 3. Get DCMs names :
    dicom_names = [os.path.basename(name).split(".")[0] for name in vol_infos['Filename']]

    # 4. Get the positions to know the extreme positions in mm :
    positions = np.array(vol_infos['ImagePositionPatient'])
    z_positions = positions[:, 2]
    min_z, max_z = min(z_positions), max(z_positions)

    # 5. Update dicom_names to keep only the slices that are in the boxes csv:
    dicom_names = [name for name in dicom_names if name in list(augmented_dataset[augmented_dataset["SeriesInstanceUID"] == SeriesInstanceUID]['SOPInstanceUID'])]

    # 6. Make a blank seg_array
    seg_array = np.zeros_like(img_array)

    # 7. Fill the seg_array with the segmentation:
    for dcm_file in dicom_names:
        # Read dcm:
        dcm = pydicom.dcmread(os.path.join(dcm_folder_path, dcm_file + ".dcm"))

        # Get its position:
        image_position = list(map(float, dcm.ImagePositionPatient))

        # Extract box from the augmented dataset for this dcm (i.e. for this slice)
        x = int(augmented_dataset[augmented_dataset["SOPInstanceUID"] == dcm_file]["x"].iloc[0])
        y = int(augmented_dataset[augmented_dataset["SOPInstanceUID"] == dcm_file]["y"].iloc[0])
        width = int(augmented_dataset[augmented_dataset["SOPInstanceUID"] == dcm_file]["width"].iloc[0])
        height = int(augmented_dataset[augmented_dataset["SOPInstanceUID"] == dcm_file]["height"].iloc[0])

        # Convert locations to a slice number for axis z:
        slice_pos_z = (max_z - image_position[2])/res[0]
        z_slice = int(slice_pos_z)

        # Make the segmentation array for this slice:
        mask_slice = np.zeros((img_array.shape[1], img_array.shape[2]))
        mask_slice[int(y):int(y)+int(height)+1, int(x):int(x)+int(width)+1] = 1
        seg_array[int(z_slice)] = mask_slice

    # 8. Save segmentation as nifti:
    seg_nifti = sitk.GetImageFromArray(seg_array)
    seg_nifti.SetOrigin(vol_infos['ImagePositionPatient'][-1])
    seg_nifti.SetDirection(np.array(vol_infos['ImageOrientationPatient'] + ['0', '0' ,'1']).astype(float))
    seg_nifti.SetSpacing((vol_infos['PixelSpacing'], vol_infos['PixelSpacing'], 1.0))

    save_path = os.path.join(OUTPUT_DIR, f'masks/{couple[0]}_{couple[1]}' + ".nii.gz")
    sitk.WriteImage(seg_nifti, save_path)  

    # 9. Save image as nifti:
    img_nifti = sitk.GetImageFromArray(img_array)
    img_nifti.SetOrigin(vol_infos['ImagePositionPatient'][-1])
    img_nifti.SetDirection(np.array(vol_infos['ImageOrientationPatient'] + ['0', '0' ,'1']).astype(float))
    img_nifti.SetSpacing((vol_infos['PixelSpacing'], vol_infos['PixelSpacing'], 1.0))   

    save_path = os.path.join(OUTPUT_DIR, f'images/{couple[0]}_{couple[1]}' + ".nii.gz")
    sitk.WriteImage(img_nifti, save_path)
    