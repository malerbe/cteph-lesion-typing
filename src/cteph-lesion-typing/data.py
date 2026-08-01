"""
Dataset and dataloaders for 3D NIfTI crop classification.

Author: Louca Malerba
"""

# Standard imports
import logging
from pathlib import Path
import os 
import numpy as np
import json

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
            
            self.file_paths.append(os.path.join(patch_dir, "images", fname))
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
    splits_file = data_config["splits_file"]
    val_fold = data_config["val_fold"]
    val_fold -= 1 # We want to have a natural numbering in the config file

    ######################################
    # Load splits
    ######################################
    with open(splits_file, 'r', encoding='utf-8') as f:
        splits_json = json.load(f)

    folds = splits_json["folds"]
    n_splits = len(folds)

    assert (1 <= data_config["val_fold"] <= n_splits), "val fold must be between 1 and n_splits"

    valid_filenames = folds[val_fold]
    train_filenames = []
    for fold in range(n_splits):
        if fold != val_fold:
            train_filenames.extend(folds[fold])
    
    ######################################
    # Transforms
    ######################################
    transforms_name = data_config["transforms_name"] 
    train_transform = get_transforms(transforms_name, train=True)
    logging.info(f"  - Training augmentations ({transforms_name}): {train_transform}")

    ######################################
    # Datasets
    ######################################
    train_dataset = PatchClassificationDataset(patch_dir=dataset_dir, filenames=train_filenames, transform=train_transform)
    valid_dataset = PatchClassificationDataset(patch_dir=dataset_dir, filenames=valid_filenames) # no augmentation

    ######################################
    # Sampler
    ######################################
    # Weighted sampler: equal probability per class during training
    _, train_labels = train_dataset.get_labels()
    class_counts = np.bincount(train_labels, minlength=len(CLASS_TO_IDX)).astype(float)
    class_counts[class_counts == 0] = 1.0  # avoid division by zero
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    logging.info(f"  - WeightedRandomSampler: class_weights = {dict(enumerate(class_weights))}")

    ######################################
    # Dataloaders
    ######################################
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # replaces shuffle=True-
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )

    ######################################
    # Return dataloaders, input size, and num_classes
    ######################################
    # Get input_size from first sample
    first_sample = train_dataset[0]
    first_tensor = first_sample[0]
    input_size = tuple(first_tensor.shape)  # (1, D, H, W)

    return train_loader, valid_loader, input_size, len(CLASS_TO_IDX)

if __name__ == "__main__":
    """
    AI-GENERATED Sanity-check script:
    
    Sanity-check script: loads a real config.yaml (same format as used for
    training), builds the actual train/valid dataloaders exactly like
    main.py would, and dumps diagnostics + figures so you can eyeball
    whether everything is coherent before launching a real training run.

    Usage: python data.py config.yaml output_dir
    """
    import sys
    import yaml
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml output_dir")
        sys.exit(-1)

    logging.info("Loading {} configuration file".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

    ######################################
    # Build the real dataloaders, exactly like training would
    ######################################
    logging.info("Building dataloaders from config...")
    train_loader, valid_loader, input_size, num_classes = get_dataloaders(data_config, use_cuda=False)
    train_dataset = train_loader.dataset
    valid_dataset = valid_loader.dataset

    logging.info(f"input_size={input_size}, num_classes={num_classes}")
    logging.info(f"train: {len(train_dataset)} patches / valid: {len(valid_dataset)} patches")

    ######################################
    # 1) Patient-level leakage check
    ######################################
    train_suids = set(train_dataset.study_uids)
    valid_suids = set(valid_dataset.study_uids)
    leak = train_suids & valid_suids
    if leak:
        logging.error(f"/!\\ {len(leak)} patient(s) appear in BOTH train and valid: {sorted(leak)[:10]}")
    else:
        logging.info("OK - no patient overlap between train and valid")

    ######################################
    # 2) Label distribution per split
    ######################################
    for name, ds in [("train", train_dataset), ("valid", valid_dataset)]:
        counts = np.bincount(ds.labels, minlength=num_classes)
        dist = {idx_to_class[i]: int(c) for i, c in enumerate(counts)}
        logging.info(f"{name} label distribution: {dist}")

    ######################################
    # 3) Sampler behaviour (train dataloader only)
    ######################################
    sampled_labels = []
    for _, batch_labels in train_loader:
        sampled_labels.extend(batch_labels.tolist())
        if len(sampled_labels) >= 500:
            break
    sampled_counts = np.bincount(sampled_labels, minlength=num_classes)
    sampled_dist = {idx_to_class[i]: int(c) for i, c in enumerate(sampled_counts)}
    logging.info(f"sampler draws over first {len(sampled_labels)} train samples: {sampled_dist}")

    ######################################
    # 4) Patch shape consistency (catches DataLoader collate crashes early)
    ######################################
    shapes = set()
    n_shape_check = min(50, len(train_dataset))
    for i in range(n_shape_check):
        img, _ = valid_dataset[i] if i < len(valid_dataset) else train_dataset[i]
        shapes.add(tuple(img.shape))
    if len(shapes) > 1:
        logging.warning(f"/!\\ inconsistent patch shapes found: {shapes} "
                         f"-> batches with mixed shapes will crash at collate time")
    else:
        logging.info(f"OK - consistent patch shape over {n_shape_check} samples: {shapes}")

    ######################################
    # 5) Intensity stats (sanity check for normalization)
    ######################################
    n_intensity_check = min(20, len(valid_dataset))
    all_vals = np.concatenate([valid_dataset[i][0].numpy().ravel() for i in range(n_intensity_check)])
    logging.info(
        f"intensity stats over {n_intensity_check} raw valid samples: "
        f"min={all_vals.min():.2f}, max={all_vals.max():.2f}, "
        f"mean={all_vals.mean():.2f}, std={all_vals.std():.2f}"
    )
    if abs(all_vals.mean()) > 5 or not (0.1 < all_vals.std() < 10):
        logging.warning("/!\\ intensity stats look off for a normalized CT patch "
                         "(expected roughly mean~0, std~1) - double check normalization")

    ######################################
    # 6) Visual check: orthogonal slices + MIPs, raw vs. augmented
    ######################################
    def slices_and_mips(volume):
        """volume: (D, H, W) numpy array -> dict of 6 2D views."""
        d, h, w = volume.shape
        return {
            "axial slice": volume[d // 2, :, :],
            "coronal slice": volume[:, h // 2, :],
            "sagittal slice": volume[:, :, w // 2],
            "axial MIP": volume.max(axis=0),
            "coronal MIP": volume.max(axis=1),
            "sagittal MIP": volume.max(axis=2),
        }

    def plot_raw_vs_augmented(raw_vol, aug_vol, title, save_path):
        views_order = ["axial slice", "coronal slice", "sagittal slice", "axial MIP", "coronal MIP", "sagittal MIP"]
        raw_views = slices_and_mips(raw_vol)
        aug_views = slices_and_mips(aug_vol)

        fig, axes = plt.subplots(2, len(views_order), figsize=(3 * len(views_order), 6))
        vmin, vmax = raw_vol.min(), raw_vol.max()
        for col, key in enumerate(views_order):
            axes[0, col].imshow(raw_views[key], cmap="gray", vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f"raw - {key}", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(aug_views[key], cmap="gray", vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f"augmented - {key}", fontsize=9)
            axes[1, col].axis("off")

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def load_raw_tensor(file_path):
        """Bypasses any transform, reproducing exactly what __getitem__ does before the transform step."""
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(file_path))).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    n_examples_per_class = 3
    picked = {c: 0 for c in CLASS_TO_IDX}
    for i in range(len(train_dataset)):
        class_name = idx_to_class[train_dataset.labels[i]]
        if picked[class_name] >= n_examples_per_class:
            continue

        raw_tensor = load_raw_tensor(train_dataset.file_paths[i])
        aug_tensor = train_dataset.transform(raw_tensor.clone()) if train_dataset.transform is not None else raw_tensor

        raw_vol = raw_tensor[0].numpy()
        aug_vol = aug_tensor[0].numpy()

        fname = Path(train_dataset.file_paths[i]).name
        save_path = output_dir / f"{class_name}_{picked[class_name]}_{fname.replace('.nii.gz', '')}.png"
        plot_raw_vs_augmented(raw_vol, aug_vol, f"{class_name} - {fname}", save_path)
        logging.info(f"saved {save_path}")
        picked[class_name] += 1

        if all(v >= n_examples_per_class for v in picked.values()):
            break

    for class_name, n_found in picked.items():
        if n_found < n_examples_per_class:
            logging.warning(f"/!\\ only found {n_found}/{n_examples_per_class} '{class_name}' examples in train set")

    logging.info(f"Done. Figures and logs above are in {output_dir}")
