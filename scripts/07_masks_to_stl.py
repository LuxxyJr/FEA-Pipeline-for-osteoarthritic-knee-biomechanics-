"""
Script 07: Convert femur/tibia binary masks into STL surface meshes.

Run:
    python scripts/07_masks_to_stl.py

Outputs:
    data/meshes/stl/baseline_femur_left.stl
    data/meshes/stl/baseline_tibia_left.stl
    data/meshes/stl/with_insole_femur_left.stl
    data/meshes/stl/with_insole_tibia_left.stl
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASK_ROOT = PROJECT_ROOT / "data" / "segmentations"
OUTPUT_DIR = PROJECT_ROOT / "data" / "meshes" / "stl"

CASES = ["baseline", "with_insole"]
MASKS = ["femur_left", "tibia_left"]


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n_labels = ndimage.label(mask)
    if n_labels == 0:
        raise ValueError("Mask is empty")
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_labels + 1))
    keep_label = int(np.argmax(sizes) + 1)
    return labeled == keep_label


def binary_to_mesh(mask_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(str(mask_path))
    mask = sitk.GetArrayFromImage(image) > 0
    mask = keep_largest_component(mask)

    # Gentle smoothing removes staircase noise from manual masks without changing gross anatomy.
    mask = ndimage.binary_closing(mask, iterations=1)
    mask = ndimage.binary_fill_holes(mask)

    spacing_xyz = image.GetSpacing()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    verts_zyx, faces, _normals, _values = measure.marching_cubes(
        mask.astype(np.uint8),
        level=0.5,
        spacing=spacing_zyx,
    )

    # Convert marching-cubes coordinates from z/y/x to x/y/z for STL.
    verts_xyz = verts_zyx[:, [2, 1, 0]]
    return verts_xyz.astype(np.float64), faces.astype(np.int64), spacing_xyz


def face_normal(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
    normal = np.cross(v2 - v1, v3 - v1)
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        return np.array([0.0, 0.0, 0.0])
    return normal / length


def write_ascii_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, solid_name: str) -> None:
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f"solid {solid_name}\n")
        for face in faces:
            v1, v2, v3 = vertices[face]
            normal = face_normal(v1, v2, v3)
            f.write(f"  facet normal {normal[0]:.8e} {normal[1]:.8e} {normal[2]:.8e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v1[0]:.8e} {v1[1]:.8e} {v1[2]:.8e}\n")
            f.write(f"      vertex {v2[0]:.8e} {v2[1]:.8e} {v2[2]:.8e}\n")
            f.write(f"      vertex {v3[0]:.8e} {v3[1]:.8e} {v3[2]:.8e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    total = 0.0
    for face in faces:
        v1, v2, v3 = vertices[face]
        total += 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    return float(total)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        for mask_name in MASKS:
            mask_path = MASK_ROOT / case / f"{mask_name}.nii.gz"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")

            output_path = OUTPUT_DIR / f"{case}_{mask_name}.stl"
            print(f"\nMeshing {case}/{mask_name}")
            vertices, faces, spacing = binary_to_mesh(mask_path)
            write_ascii_stl(output_path, vertices, faces, f"{case}_{mask_name}")

            bounds_min = vertices.min(axis=0)
            bounds_max = vertices.max(axis=0)
            print(f"  Vertices: {len(vertices)}")
            print(f"  Faces: {len(faces)}")
            print(f"  Surface area: {mesh_area(vertices, faces):.2f} mm^2")
            print(f"  Bounds min xyz: {[round(float(v), 2) for v in bounds_min]}")
            print(f"  Bounds max xyz: {[round(float(v), 2) for v in bounds_max]}")
            print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

            if len(vertices) < 100 or len(faces) < 100:
                raise ValueError(f"Mesh is suspiciously small: {output_path}")

    print("\nDone. STL meshes are ready for visual inspection/volumetric meshing.")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
