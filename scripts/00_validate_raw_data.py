"""
Script 00: Validate that the copied raw DICOM folder matches the guide.

Run from the project root:
    python scripts/00_validate_raw_data.py

Optionally set RAW_DICOM_DIR to the local DICOM folder.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

EXPECTED_TARGET_SERIES = {
    "828116544": 132,
    "828117201": 132,
    "828118030": 132,
    "828118645": 132,
}


def main() -> None:
    files = sorted(DICOM_DIR.glob("*.dcm"))
    print(f"Raw DICOM directory: {DICOM_DIR}")
    print(f"Total .dcm files: {len(files)}")

    if len(files) != 588:
        raise ValueError(f"Expected 588 DICOM files in this dataset, found {len(files)}")

    for series_suffix, expected_count in EXPECTED_TARGET_SERIES.items():
        series_files = sorted(DICOM_DIR.glob(f"*{series_suffix}*.dcm"))
        indices = sorted(int(path.stem.rsplit(".", 1)[1]) for path in series_files)
        missing = sorted(set(range(expected_count)) - set(indices))

        print(f"Series {series_suffix}: {len(series_files)} slices")
        if len(series_files) != expected_count or missing:
            raise ValueError(
                f"Series {series_suffix} failed validation. "
                f"Count={len(series_files)}, missing={missing}"
            )

    print("Raw data validation passed.")


if __name__ == "__main__":
    main()
