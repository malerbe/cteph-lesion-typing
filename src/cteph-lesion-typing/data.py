"""Dataset and dataloaders for 3D NIfTI crop classification."""

# Standard imports
import logging
from pathlib import Path
import os 
import numpy as np

# Imports
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from sklearn.model_selection import StratifiedShuffleSplit
from monai.transforms import (
    Compose,
    RandFlip,
    RandGaussianNoise,
    RandAffine,
    RandAdjustContrast,
    RandScaleIntensity,
)


######################################
# Configuration
######################################
CLASS_TO_IDX = {"acute": 0, "chronic": 1}

######################################
# Datasets
######################################
class PatchClassificationDataset(Dataset):
    """
    Dataset for classifying 3D nifti crops

    /!\ DATA IS ASSUMED TO BE PRE-NORMALIZED IF NORMALIZATION IS NEEDED /!\
    """
    def __init__(self, patch_dir, filenames, transform = None):
        self.filenames = filenames
        self.file_paths = []
        self.labels = []
        self.study_uids = [] # patient-level identificatin for train.val spliiting without data-leakage
        self.transform = transform

        for fname in sorted(self.filenames):
            parts = fname.replace(".nii.gz", "").split("_")
            class_name = parts[-2]
            if class_name not in CLASS_TO_IDX:
                raise ValueError(f"Unknown class {class_name}")
            
            self.file_paths.append(os.path.join(patch_dir, fname))
            self.labels.append(CLASS_TO_IDX[class_name])
            self.study_uids.append(parts[0])

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.file_paths, self.labels

    def __getitem__(self, idx):
        image = sitk.GetArrayFromImage(sitk.ReadImage(self.file_paths[idx])).astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)  # add channel dim
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
    
        return image_tensor, label_tensor

######################################################################
# Transforms
######################################################################

def get_transforms(transforms_name: str, train: bool = True):
    """Named augmentation presets"""

    if train:
        presets = {
            # ---- none ; no augmentation, just convert to tensor ----
            "none": Compose([]),
            # ---- safe ; spatial flips only ----
            "safe": Compose([
                RandFlip(prob=0.5, spatial_axis=0),  
                RandFlip(prob=0.5, spatial_axis=1),  
                RandFlip(prob=0.5, spatial_axis=2),  
            ]),
            # ---- moderate ; safe + light intensity ----
            "moderate": Compose([
                RandFlip(spatial_axis=0, prob=0.4),
                RandFlip(spatial_axis=1, prob=0.4),
                RandFlip(spatial_axis=2, prob=0.4),
                RandScaleIntensity(factors=0.1, prob=0.2),
                RandAdjustContrast(prob=0.2, gamma=(0.85, 1.15)),
            ]),
            # ---- standard ; moderate + affine ----
            "standard": Compose([
                RandFlip(spatial_axis=0, prob=0.5),
                RandFlip(spatial_axis=1, prob=0.5),
                RandFlip(spatial_axis=2, prob=0.5),
                RandAffine(
                    prob=0.3,
                    rotate_range=(0.15, 0.15, 0.15),  
                    scale_range=(0.05, 0.05, 0.05), 
                    padding_mode="border",
                ),
                RandScaleIntensity(factors=0.15, prob=0.25),
                RandAdjustContrast(prob=0.25, gamma=(0.8, 1.2)),
            ]),
            # ---- aggressive ; standard + strong noise and intensity ----
            "aggressive": Compose([
                RandFlip(spatial_axis=0, prob=0.5),
                RandFlip(spatial_axis=1, prob=0.5),
                RandFlip(spatial_axis=2, prob=0.5),
                RandAffine(
                    prob=0.4,
                    rotate_range=(0.26, 0.26, 0.26),   # more than standard
                    scale_range=(0.1, 0.1, 0.1),    # more than standard
                    padding_mode="border",
                ),
                RandGaussianNoise(prob=0.2, mean=0.0, std=0.05), # new
                RandScaleIntensity(factors=0.2, prob=0.3), # more than standard
                RandAdjustContrast(prob=0.3, gamma=(0.7, 1.3)), # more than standard
            ]),
        }

        if transforms_name not in presets:
            raise ValueError(
                f"Unknown transforms_name '{transforms_name}'. "
                f"Available: {list(presets.keys())}"
            )

        return presets[transforms_name]
    else:
        return Compose([])

######################################################################
# get_dataloaders and helpers
######################################################################
def get_dataloaders(data_config: dict, use_cuda: bool):
    """Main entry point to get dataloaders based on the provided config."""
    task = data_config["task"]
    if task == "classification":
        return _get_dataloaders_classification(data_config, use_cuda)
    else:
        raise ValueError(f"Unknown task '{task}' in data_config")

def _get_dataloaders_classification(data_config, use_cuda):
    """
    Build train and validation dataloaders from a patch classification dataset.

    The split is done at the **patient level** to avoid data leakage:
    all crops from a given patient are in the same split.
    """
    ######################################
    # Config
    ######################################
    dataset_dir = Path(data_config["dataset_dir"])
    valid_ratio = data_config.get("valid_ratio", 0.2)
    batch_size = data_config.get("batch_size", 1)
    num_workers = data_config.get("num_workers", 2)

    ######################################
    # Load splits
    ######################################
   

    