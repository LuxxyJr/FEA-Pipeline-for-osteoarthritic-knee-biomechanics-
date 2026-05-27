"""
Script 02: Resample all four NIfTI volumes to 0.82 mm isotropic spacing
and normalize intensities to [0, 1].

Run from the project root:
    python scripts/02_preprocess.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "nifti"
OUTPUT_DIR = PROJECT_ROOT / "data" / "nifti"
TARGET_SPACING = (0.82, 0.82, 0.82)

VOLUMES = [
    "series3_standing_30deg_WITHOUT_insole",
    "series6_standing_30deg_WITH_insole",
    "series9_standing_WITH_insole",
    "series12_standing_BASELINE",
]


def resample_to_isotropic(image: sitk.Image, target_spacing: tuple[float, float, float]) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def normalize_intensity(image: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    p2, p98 = np.percentile(arr, (2, 98))
    if np.isclose(p2, p98):
        raise ValueError("Cannot normalize image because p2 and p98 intensities are identical")

    arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
    result = sitk.GetImageFromArray(arr)
    result.CopyInformation(image)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for volume_name in VOLUMES:
        input_path = INPUT_DIR / f"{volume_name}.nii.gz"
        output_path = OUTPUT_DIR / f"{volume_name}_processed.nii.gz"

        if not input_path.exists():
            raise FileNotFoundError(f"Missing input NIfTI: {input_path}")

        print(f"\nProcessing: {volume_name}")
        image = sitk.ReadImage(str(input_path))
        print(f"  Input spacing: {image.GetSpacing()}, size: {image.GetSize()}")

        image = resample_to_isotropic(image, TARGET_SPACING)
        print(f"  Resampled spacing: {image.GetSpacing()}, size: {image.GetSize()}")

        image = normalize_intensity(image)
        sitk.WriteImage(image, str(output_path))

        if not output_path.exists():
            raise RuntimeError(f"Expected output was not created: {output_path}")
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

    print("\nDone. Created four processed NIfTI volumes.")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
