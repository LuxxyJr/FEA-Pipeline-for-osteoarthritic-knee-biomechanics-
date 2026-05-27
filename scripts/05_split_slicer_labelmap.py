"""
Script 05: Split a Slicer-exported multi-label segmentation into binary masks.

Slicer may export:
    data/segmentations/<case>/Segmentation.nii
    data/segmentations/<case>/Segmentation.labels.csv

This script converts label values into:
    femur_left.nii.gz
    tibia_left.nii.gz

Run:
    python scripts/05_split_slicer_labelmap.py baseline
    python scripts/05_split_slicer_labelmap.py with_insole
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_labels(labels_path: Path) -> dict[int, str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing Slicer labels file: {labels_path}")

    labels: dict[int, str] = {}
    with labels_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row["LabelValue"])] = row["Name"]
    return labels


def split_case(case_name: str) -> None:
    case_dir = PROJECT_ROOT / "data" / "segmentations" / case_name
    labelmap_path = case_dir / "Segmentation.nii"
    labels_path = case_dir / "Segmentation.labels.csv"

    if not labelmap_path.exists():
        raise FileNotFoundError(f"Missing Slicer labelmap: {labelmap_path}")

    labels = load_labels(labels_path)
    image = sitk.ReadImage(str(labelmap_path))
    arr = sitk.GetArrayFromImage(image)

    print(f"Splitting {case_name}: {labelmap_path.relative_to(PROJECT_ROOT)}")
    print(f"Label values present in image: {sorted(int(v) for v in np.unique(arr))}")

    for label_value, name in sorted(labels.items()):
        if name not in {"femur_left", "tibia_left"}:
            print(f"  Skipping label {label_value}: {name}")
            continue

        output_path = case_dir / f"{name}.nii.gz"
        if label_value not in set(int(v) for v in np.unique(arr)):
            if output_path.exists():
                print(f"  Preserved existing {output_path.relative_to(PROJECT_ROOT)}; label {label_value} not present in export")
                continue
            print(f"  Warning: label {label_value} ({name}) not present and no existing output found")
            continue

        mask = (arr == label_value).astype(np.uint8)
        voxels = int(mask.sum())
        out = sitk.GetImageFromArray(mask)
        out.CopyInformation(image)

        sitk.WriteImage(out, str(output_path))
        print(f"  Wrote {output_path.relative_to(PROJECT_ROOT)} ({voxels} voxels)")

        if voxels == 0:
            print(f"  Warning: {name} is empty. That is OK only if you have not segmented it yet.")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"baseline", "with_insole"}:
        raise SystemExit("Usage: python scripts/05_split_slicer_labelmap.py baseline|with_insole")

    os.chdir(PROJECT_ROOT)
    split_case(sys.argv[1])


if __name__ == "__main__":
    main()
