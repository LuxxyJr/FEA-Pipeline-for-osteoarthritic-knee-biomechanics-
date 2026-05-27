"""
Script 04b: Create robust overlay QC images for femur/tibia masks.

Run from the project root:
    python scripts/04b_segmentation_qc.py

Output:
    results/segmentation_qc_bones.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "segmentation_qc_bones.png"

CASES = {
    "baseline": PROJECT_ROOT / "data" / "nifti" / "series12_standing_BASELINE_processed.nii.gz",
    "with_insole": PROJECT_ROOT / "data" / "nifti" / "series9_standing_WITH_insole_processed.nii.gz",
}

MASKS = {
    "femur_left": (80, 220, 110),
    "tibia_left": (245, 210, 90),
}


def read_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_2d, (1, 99))
    if np.isclose(lo, hi):
        lo, hi = float(slice_2d.min()), float(slice_2d.max())
    out = np.clip((slice_2d - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def overlay_slice(mri: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], title: str) -> Image.Image:
    signal = mask.sum(axis=(1, 2))
    z = int(np.argmax(signal))
    voxels = int(mask.sum())

    base = normalize_slice(mri[z])
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)

    if voxels > 0:
        rgb[mask[z]] = 0.55 * rgb[mask[z]] + 0.45 * np.array(color, dtype=np.float32)

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((512, 512))
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), f"{title} z={z} vox={voxels}", fill=(255, 255, 255))
    return image


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tiles: list[Image.Image] = []
    print("Segmentation QC:")
    for case_name, mri_path in CASES.items():
        mri = read_array(mri_path)
        for mask_name, color in MASKS.items():
            mask_path = PROJECT_ROOT / "data" / "segmentations" / case_name / f"{mask_name}.nii.gz"
            mask = read_array(mask_path) > 0
            if mask.shape != mri.shape:
                raise ValueError(f"Shape mismatch for {case_name}/{mask_name}: MRI={mri.shape}, mask={mask.shape}")

            coords = np.argwhere(mask)
            voxels = int(mask.sum())
            if voxels == 0:
                print(f"  FAIL {case_name}/{mask_name}: empty mask")
            else:
                print(
                    f"  {case_name}/{mask_name}: voxels={voxels}, "
                    f"bbox_min={coords.min(axis=0).tolist()}, bbox_max={coords.max(axis=0).tolist()}"
                )
            tiles.append(overlay_slice(mri, mask, color, f"{case_name} {mask_name}"))

    canvas = Image.new("RGB", (1024, 1024), (0, 0, 0))
    positions = [(0, 0), (512, 0), (0, 512), (512, 512)]
    for tile, position in zip(tiles, positions):
        canvas.paste(tile, position)

    canvas.save(OUTPUT_PATH)
    print(f"\nSaved QC figure: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
