"""
Extract a fixed size patches centered on the heart based on an image and a mask
of the hearts components extracted using TotalSegmentator
Save the image patch and its corresponding mask patch as separate nii.gz files

running total segmentator heartchambers_highres is needed. A helper script is available:
cteph-lesion-typing/utils/run_totalsegmentator.py

author: Louca Malerba
"""
# Imports
import os
import sys
import numpy as np
import SimpleITK as sitk
import json
import logging

# Local imports
sys.path.append(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/utils")
import utils

##################
# Configuration
##################
PATH_TO_INPUT_DATASET = "/data/lmalerba/augmented_RSPECT"
PATH_TO_TOTAL_SEGMENTATOR_OUT = "/data/lmalerba/total_segmentator_out/"
OUTPUT_PATH = "/data/lmalerba/lesions_patches_dataset_normalized"

NORMALIZE = True # If set as True, CT normalization will be done and generated dataset will be normalized
PATH_TO_FINGERPRINT = "/data/lmalerba/augmented_RSPECT/dataset_fingerprint.json"

if NORMALIZE == True and PATH_TO_FINGERPRINT == None:
    raise ValueError("For normalization, dataset fingerprint is needed")
else:
    logging.warning("Normalization is deactivated ! Make sure it really is what you want !")





