"""
Script 08: Create a simple QC preview of STL meshes.

Run:
    python scripts/08_mesh_qc_preview.py

Output:
    results/mesh_stl_qc.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STL_DIR = PROJECT_ROOT / "data" / "meshes" / "stl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "mesh_stl_qc.png"

MESHES = [
    ("baseline femur", STL_DIR / "baseline_femur_left.stl", (80, 220, 110)),
    ("baseline tibia", STL_DIR / "baseline_tibia_left.stl", (245, 210, 90)),
    ("with_insole femur", STL_DIR / "with_insole_femur_left.stl", (80, 220, 110)),
    ("with_insole tibia", STL_DIR / "with_insole_tibia_left.stl", (245, 210, 90)),
]


def read_ascii_stl_vertices(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing STL: {path}")

    vertices = []
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            _tag, x, y, z = stripped.split()
            vertices.append((float(x), float(y), float(z)))

    if not vertices:
        raise ValueError(f"No ASCII vertices found in {path}")
    return np.array(vertices, dtype=float)


def project_vertices(vertices: np.ndarray, width: int = 512, height: int = 512) -> Image.Image:
    # Project x/y points; z is ignored for a quick silhouette-style QC.
    xy = vertices[:, :2]
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    scale = min((width - 50) / (mx[0] - mn[0] + 1e-8), (height - 50) / (mx[1] - mn[1] + 1e-8))
    pts = (xy - mn) * scale + 25
    pts[:, 1] = height - pts[:, 1]

    image = Image.new("RGB", (width, height), (8, 8, 8))
    return image, pts.astype(int), mn, mx


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tiles = []

    print("STL mesh QC:")
    for label, path, color in MESHES:
        vertices = read_ascii_stl_vertices(path)
        image, pts, mn, mx = project_vertices(vertices)
        draw = ImageDraw.Draw(image)

        # Downsample plotted points for speed; there are many duplicate STL vertices.
        if len(pts) > 60000:
            pts = pts[:: max(1, len(pts) // 60000)]
        for x, y in pts:
            draw.point((int(x), int(y)), fill=color)

        draw.text((12, 12), label, fill=(255, 255, 255))
        draw.text((12, 30), f"vertices listed={len(vertices)}", fill=(220, 220, 220))
        tiles.append(image)

        print(
            f"  {label}: listed_vertices={len(vertices)}, "
            f"xy_min={[round(float(v), 2) for v in mn]}, xy_max={[round(float(v), 2) for v in mx]}"
        )

    canvas = Image.new("RGB", (1024, 1024), (0, 0, 0))
    positions = [(0, 0), (512, 0), (0, 512), (512, 512)]
    for tile, position in zip(tiles, positions):
        canvas.paste(tile, position)

    canvas.save(OUTPUT_PATH)
    print(f"\nSaved mesh QC: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
