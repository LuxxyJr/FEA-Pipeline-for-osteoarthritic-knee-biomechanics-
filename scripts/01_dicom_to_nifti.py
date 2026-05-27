"""
Script 01: Convert the four target DICOM series to NIfTI volumes.

Run from the project root:
    python scripts/01_dicom_to_nifti.py

Optionally set RAW_DICOM_DIR to the local DICOM folder.
"""

from __future__ import annotations

import os
from pathlib import Path

import pydicom
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "nifti"


def default_dicom_dir() -> Path:
    configured = os.environ.get("RAW_DICOM_DIR")
    if configured:
        return Path(configured)

    raw_root = PROJECT_ROOT / "data" / "raw"
    candidates = [p for p in raw_root.iterdir() if p.is_dir()] if raw_root.exists() else []
    if len(candidates) == 1:
        return candidates[0]
    return raw_root / "subject"


DICOM_DIR = default_dicom_dir()

EXPECTED_SLICES = 132
TARGET_SERIES = {
    "828116544": "series3_standing_30deg_WITHOUT_insole",
    "828117201": "series6_standing_30deg_WITH_insole",
    "828118030": "series9_standing_WITH_insole",
    "828118645": "series12_standing_BASELINE",
}


def read_series_metadata(dicom_path: Path) -> tuple[str, int, float]:
    """Return SeriesInstanceUID, InstanceNumber, and slice position for sorting."""
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
    series_uid = str(ds.SeriesInstanceUID)
    instance_number = int(getattr(ds, "InstanceNumber", -1))

    image_position = getattr(ds, "ImagePositionPatient", None)
    slice_position = float(image_position[2]) if image_position is not None else float(instance_number)
    return series_uid, instance_number, slice_position


def group_dicoms_by_series(dicom_dir: Path) -> dict[str, list[Path]]:
    all_files = sorted(dicom_dir.glob("*.dcm"))
    if not all_files:
        raise FileNotFoundError(f"No .dcm files found in {dicom_dir}")

    series_files: dict[str, list[Path]] = {}
    for dicom_path in all_files:
        series_uid, _, _ = read_series_metadata(dicom_path)
        series_files.setdefault(series_uid, []).append(dicom_path)

    print(f"Found {len(all_files)} DICOM files")
    print(f"Found {len(series_files)} series total")
    return series_files


def target_name_for_uid(series_uid: str) -> str | None:
    for uid_suffix, name in TARGET_SERIES.items():
        if uid_suffix in series_uid:
            return name
    return None


def sorted_dicom_paths(file_list: list[Path]) -> list[str]:
    decorated = []
    for path in file_list:
        _, instance_number, slice_position = read_series_metadata(path)
        decorated.append((slice_position, instance_number, path))

    decorated.sort(key=lambda item: (item[0], item[1], item[2].name))
    return [str(item[2]) for item in decorated]


def convert_series_to_nifti(file_list: list[Path], output_path: Path) -> None:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_dicom_paths(file_list))
    image = reader.Execute()

    print(f"  Volume size: {image.GetSize()}")
    print(f"  Voxel spacing: {image.GetSpacing()} mm")
    print(f"  Origin: {image.GetOrigin()}")

    sitk.WriteImage(image, str(output_path))
    if not output_path.exists():
        raise RuntimeError(f"Expected output was not created: {output_path}")
    print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    series_files = group_dicoms_by_series(DICOM_DIR)

    found_targets: dict[str, int] = {}
    converted = 0

    for series_uid, file_list in sorted(series_files.items()):
        matched_name = target_name_for_uid(series_uid)
        if matched_name is None:
            continue

        found_targets[matched_name] = len(file_list)
        print(f"\nConverting: {matched_name}")
        print(f"  Series UID: {series_uid}")
        print(f"  Files: {len(file_list)} slices")

        if len(file_list) != EXPECTED_SLICES:
            raise ValueError(
                f"{matched_name} has {len(file_list)} slices; expected {EXPECTED_SLICES}"
            )

        output_path = OUTPUT_DIR / f"{matched_name}.nii.gz"
        convert_series_to_nifti(file_list, output_path)
        converted += 1

    missing = sorted(set(TARGET_SERIES.values()) - set(found_targets))
    if missing:
        raise RuntimeError(f"Missing target series: {', '.join(missing)}")

    print(f"\nDone. Converted {converted}/4 target series.")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
