"""
Run TotalSegmentator on a directory of NIfTI files, one file at a time.

Passing a directory directly to TotalSegmentator causes it to treat the folder
as a DICOM series and copy every file into /tmp via dicom2nifti, which can fill
up the /tmp partition. This script passes each .nii.gz file individually so
TotalSegmentator detects img_type="nifti" and skips that copy entirely.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input_dir", required=True, type=Path,
                        help="Directory containing .nii.gz files to segment.")
    parser.add_argument("-o", "--output_dir", required=True, type=Path,
                        help="Root output directory. One sub-folder per volume is created.")
    parser.add_argument("--task", default="total",
                        help="TotalSegmentator task (e.g. heartchambers_highres).")
    parser.add_argument("--roi_subset", nargs="+", default=None,
                        help="ROI subset passed to TotalSegmentator (e.g. heart). Only valid for task=total.")
    parser.add_argument("--fast", action="store_true",
                        help="Use the fast (low-res) model.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip volumes whose output folder already exists.")
    parser.add_argument("--totalseg_bin", default="TotalSegmentator",
                        help="Path or name of the TotalSegmentator binary.")
    parser.add_argument("--license_number", default=None,
                        help="License key for non-free tasks (e.g. heartchambers_highres).")
    return parser.parse_args()


def main():
    args = parse_args()

    nifti_files = sorted(args.input_dir.glob("*.nii.gz"))
    if not nifti_files:
        sys.exit(f"No .nii.gz files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, nii_path in enumerate(nifti_files, 1):
        vol_name = nii_path.name.replace(".nii.gz", "")
        out_path = args.output_dir / vol_name

        if args.skip_existing and out_path.exists():
            print(f"[{i}/{len(nifti_files)}] Skipping {vol_name} (already exists)")
            continue

        out_path.mkdir(parents=True, exist_ok=True)

        cmd = [args.totalseg_bin, "-i", str(nii_path), "-o", str(out_path),
               "--task", args.task]
        if args.roi_subset:
            cmd += ["--roi_subset"] + args.roi_subset
        if args.fast:
            cmd.append("--fast")
        if args.license_number:
            cmd += ["--license_number", args.license_number]

        print(f"[{i}/{len(nifti_files)}] {vol_name}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append(vol_name)
            print(f"  WARNING: TotalSegmentator failed for {vol_name}")

    if failed:
        print(f"\n{len(failed)} volume(s) failed:")
        for name in failed:
            print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
