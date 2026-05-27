"""
Script 09: Convert STL surface meshes into Gmsh tetrahedral volume meshes.

Run:
    python scripts/09_stl_to_gmsh.py

Outputs:
    data/meshes/gmsh/*.msh
"""

from __future__ import annotations

import os
from pathlib import Path

import gmsh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STL_DIR = PROJECT_ROOT / "data" / "meshes" / "stl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "meshes" / "gmsh"

MESHES = [
    "baseline_femur_left",
    "baseline_tibia_left",
    "with_insole_femur_left",
    "with_insole_tibia_left",
]


def generate_volume_mesh(name: str, surface_size: float = 3.0, bulk_size: float = 5.0) -> None:
    stl_path = STL_DIR / f"{name}.stl"
    output_path = OUTPUT_DIR / f"{name}.msh"
    if not stl_path.exists():
        raise FileNotFoundError(f"Missing STL: {stl_path}")

    print(f"\nGenerating volume mesh: {name}")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add(name)
        gmsh.merge(str(stl_path))

        # Robust STL path: use the STL as a discrete surface directly. The
        # classifySurfaces/createGeometry path can fail on manual masks with
        # "Wrong topology of boundary mesh for parametrization".
        gmsh.model.mesh.createTopology(makeSimplyConnected=True, exportDiscrete=True)

        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise RuntimeError(f"No surfaces found after importing STL: {stl_path}")

        surface_loop = gmsh.model.geo.addSurfaceLoop([tag for _dim, tag in surfaces])
        gmsh.model.geo.addVolume([surface_loop])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", surface_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", bulk_size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write(str(output_path))

        node_tags, _coords, _params = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _node_tags = gmsh.model.mesh.getElements(3)
        n_tets = sum(len(tags) for tags in elem_tags)
        print(f"  Nodes: {len(node_tags)}")
        print(f"  Volume elements: {n_tets}")
        print(f"  Saved: {output_path.relative_to(PROJECT_ROOT)}")

        if n_tets < 100:
            raise ValueError(f"Suspiciously small tetrahedral mesh: {output_path}")
    finally:
        gmsh.finalize()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in MESHES:
        generate_volume_mesh(name)
    print("\nDone. Gmsh volume meshes are ready.")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
