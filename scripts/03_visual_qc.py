"""
Script 03: Visual QC for the four processed NIfTI volumes.

Run from the project root:
    python scripts/03_visual_qc.py

Output:
    results/qc_processed_volumes.png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "nifti"
OUTPUT_PATH = PROJECT_ROOT / "results" / "qc_processed_volumes.png"

VOLUMES = [
    ("Series 3: 30deg without insole", "series3_standing_30deg_WITHOUT_insole_processed.nii.gz"),
    ("Series 6: 30deg with insole", "series6_standing_30deg_WITH_insole_processed.nii.gz"),
    ("Series 9: standing with insole", "series9_standing_WITH_insole_processed.nii.gz"),
    ("Series 12: standing baseline", "series12_standing_BASELINE_processed.nii.gz"),
]


def load_volume(path: Path) -> tuple[np.ndarray, sitk.Image]:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed NIfTI: {path}")
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return array, image


def middle_nonempty_slice(volume: np.ndarray) -> int:
    # SimpleITK arrays are z, y, x. Use summed intensity to avoid blank edge slices.
    per_slice_signal = volume.sum(axis=(1, 2))
    nonempty = np.flatnonzero(per_slice_signal > np.percentile(per_slice_signal, 10))
    if nonempty.size == 0:
        return volume.shape[0] // 2
    return int(nonempty[nonempty.size // 2])


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    loaded = []
    print("Processed volume QC:")
    for label, filename in VOLUMES:
        path = INPUT_DIR / filename
        volume, image = load_volume(path)
        loaded.append((label, volume))
        print(
            f"  {label}: shape z/y/x={volume.shape}, "
            f"spacing={image.GetSpacing()}, "
            f"min={volume.min():.3f}, max={volume.max():.3f}, mean={volume.mean():.3f}"
        )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="black")
    for col, (label, volume) in enumerate(loaded):
        z_mid = middle_nonempty_slice(volume)
        z_quarter = max(0, min(volume.shape[0] - 1, volume.shape[0] // 4))

        axes[0, col].imshow(volume[z_mid, :, :], cmap="gray", origin="lower")
        axes[0, col].set_title(label, color="white", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(volume[z_quarter, :, :], cmap="gray", origin="lower")
        axes[1, col].set_title(f"slice z={z_quarter}", color="white", fontsize=8)
        axes[1, col].axis("off")

    plt.tight_layout()
    fig.savefig(str(OUTPUT_PATH), dpi=180, bbox_inches="tight", facecolor="black")
    print(f"\nSaved QC figure: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
