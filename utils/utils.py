import os
import warnings

import numpy as np
import pydicom


def load_dicom_series(series_dir):
    """Load a DICOM series directory using only pydicom.

    GDCM-based series discovery (sitk.ImageSeriesReader.GetGDCMSeriesIDs) finds
    no series on this anonymized dataset (invalid VRs make GDCM reject the
    files outright), so slices are read and sorted directly with pydicom
    instead.

    Returns:
        img_array: (z, y, x) numpy array, in Hounsfield units for CT.
        res: (res_z, res_y, res_x) voxel spacing in mm.
        vol_infos: dict with the per-slice header fields used downstream
            ("Filename", "ImagePositionPatient", "ImageOrientationPatient",
            "PixelSpacing").
    """
    filenames = [f for f in sorted(os.listdir(series_dir)) if not f.startswith(".")]
    if not filenames:
        raise ValueError(f"No DICOM files found in {series_dir}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")  # anonymized UIDs fail VR validation
        datasets = [(f, pydicom.dcmread(os.path.join(series_dir, f))) for f in filenames]

    datasets.sort(key=lambda item: float(item[1].ImagePositionPatient[2]))

    pixel_spacing = [float(v) for v in datasets[0][1].PixelSpacing]  # (row, column) mm
    orientation = [float(v) for v in datasets[0][1].ImageOrientationPatient]

    slices, positions = [], []
    for _, ds in datasets:
        if ds.file_meta.TransferSyntaxUID.is_compressed:
            ds.decompress()
        slices.append(pydicom.pixels.apply_rescale(ds.pixel_array, ds))
        positions.append(float(ds.ImagePositionPatient[2]))

    img_array = np.stack(slices).astype(np.int16)  # (z, y, x)
    img_array[img_array < -1024] = -1024

    # Slice spacing from the median inter-slice gap: more robust to occasional
    # missing/duplicate slices than trusting SliceThickness alone.
    z_diffs = np.abs(np.diff(positions))
    res_z = float(np.median(z_diffs)) if len(z_diffs) else float(getattr(datasets[0][1], "SliceThickness", 1.0))
    res = (res_z, pixel_spacing[0], pixel_spacing[1])

    vol_infos = {
        "Filename": [filename for filename, _ in datasets],
        "ImagePositionPatient": [[float(v) for v in ds.ImagePositionPatient] for _, ds in datasets],
        "ImageOrientationPatient": orientation,
        "PixelSpacing": pixel_spacing[0],
    }

    return img_array, res, vol_infos