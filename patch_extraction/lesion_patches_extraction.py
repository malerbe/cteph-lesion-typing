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
`
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
