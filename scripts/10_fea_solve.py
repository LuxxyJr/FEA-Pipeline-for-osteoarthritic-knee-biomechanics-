#!/usr/bin/env python3
"""
10_fea_solve.py  ─  Phase 5: Linear Elasticity FEA (DOLFINx 0.10)

Solves compressive loading on tibia for two conditions:
  baseline  vs  with_insole

Extracts:
  1. Tibial rotation (degrees)              ← transverse-plane centroid shift
  2. Medial compartment displacement (mm)   ← loading proxy (z-compression)
  3. ACL tibial attachment displacement (mm)

Output: results/fea/fea_results.json
"""

import os
import json
import numpy as np
import math
from mpi4py import MPI
import gmsh
from dolfinx.io import gmsh as dolfinx_gmsh
import dolfinx
import dolfinx.mesh as dmesh
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc as petsc_set_bc
import ufl
import basix.ufl
from petsc4py import PETSc
from scipy.spatial import KDTree

# ── Constants ─────────────────────────────────────────────────────────────────
E_CORTICAL = 17000.0  # N/mm²  (MPa — consistent with mm mesh)
NU         = 0.30     #      Poisson's ratio
LOAD_N     = 800.0    # N    ~80 kg body weight
FLEXION_DEG = 30.0    # degrees — 30° flexion angle from scan protocol
PLATEAU_AREA_MM2 = 1200.0  # mm² — fixed anatomical value for adult tibial plateau

MU    = E_CORTICAL / (2.0 * (1.0 + NU))
LMBDA = E_CORTICAL * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))

MESH_DIR = "data/meshes/gmsh"
OUT_DIR  = "results/fea"
os.makedirs(OUT_DIR, exist_ok=True)

CASES = {
    "baseline":    "baseline_tibia_left.msh",
    "with_insole": "with_insole_tibia_left.msh",
}


# ── 0. Geometric rotation (no FEA) ───────────────────────────────────────────
def icp_2d(src, dst, max_iter=50, tol=1e-6):
    """Align src to dst in XY plane via ICP. Returns rotation angle (deg)."""
    s = src[:, :2].copy()
    d = dst[:, :2].copy()
    s -= s.mean(0); d -= d.mean(0)   # center both

    total_angle = 0.0
    for _ in range(max_iter):
        tree = KDTree(d)
        _, idx = tree.query(s)
        matched = d[idx]
        # SVD-based optimal rotation
        H = s.T @ matched
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        s = s @ R.T
        total_angle += angle
        if abs(angle) < tol:
            break
    return total_angle


def compute_geometric_rotation():
    """
    Tibial rotation = ICP-aligned rotation between baseline and
    with_insole plateau geometries. Removes inter-scan pose difference.
    """
    plateaus = {}
    for name, fname in CASES.items():
        msh = load_mesh(os.path.join(MESH_DIR, fname))
        C = msh.geometry.x
        zhi = C[:, 2].max()
        plateaus[name] = C[C[:, 2] >= zhi - 5.0]

    residual_rot = icp_2d(plateaus["with_insole"], plateaus["baseline"])
    print(f"  Geometric tibial rotation (ICP): {residual_rot:.4f}°")
    return residual_rot


# ── 1. Mesh loading ───────────────────────────────────────────────────────────
def load_mesh(path: str):
    gmsh.initialize()

    # Only rank 0 should read and modify the gmsh model;
    # model_to_mesh broadcasts the mesh to all other ranks internally.
    if MPI.COMM_WORLD.rank == 0:
        gmsh.open(path)

        # DOLFINx requires at least one physical group to parse the mesh.
        # If the mesh has none, create one covering all 3-D volume entities.
        if len(gmsh.model.getPhysicalGroups()) == 0:
            volumes = gmsh.model.getEntities(3)
            if volumes:
                gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], 1)

    mesh_data = dolfinx_gmsh.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3
    )
    gmsh.finalize()
    return mesh_data.mesh


# ── 2. FEA solve ──────────────────────────────────────────────────────────────
def solve_elasticity(msh: dolfinx.mesh.Mesh):
    """
    Linear elasticity on tibia.
      BC  : fix inferior (distal) end — zero Dirichlet
      Load: compressive traction on superior (proximal) plateau
    Solver: Direct PETSc assembly + MUMPS LU — bypasses LinearProblem
            entirely to avoid DOLFINx 0.10 petsc_options_prefix issues.
    Returns (uh, coords, z_lo, z_hi, tol)
    """
    C   = msh.geometry.x
    zlo = C[:, 2].min();  zhi = C[:, 2].max()
    bone_len = zhi - zlo
    tol_inf = bone_len * 0.12             # 12% — captures full distal cut on both meshes
    tol_sup = 2.0                        # fixed 2 mm band — superior (traction)

    # Debug: confirm coordinate ranges and units
    print(f"  Coord ranges — X: [{C[:,0].min():.2f}, {C[:,0].max():.2f}]  "
          f"Y: [{C[:,1].min():.2f}, {C[:,1].max():.2f}]  "
          f"Z: [{C[:,2].min():.2f}, {C[:,2].max():.2f}]")
    print(f"  Bone length: {bone_len:.2f} mm | tol_inf: {tol_inf:.1f} mm | tol_sup: {tol_sup:.1f} mm")

    el = basix.ufl.element("Lagrange", msh.topology.cell_name(), 1, shape=(3,))
    V  = fem.functionspace(msh, el)

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)

    f_inf = dmesh.locate_entities_boundary(msh, fdim, lambda x: x[2] <= zlo + tol_inf)
    f_sup = dmesh.locate_entities_boundary(msh, fdim, lambda x: x[2] >= zhi - tol_sup)

    if len(f_sup) == 0:
        raise RuntimeError("No superior facets found — mesh may not be z-oriented")

    # Remove any overlapping facets — if a facet is in both sets,
    # the Dirichlet BC (f_inf) would zero out the load (f_sup).
    overlap = np.intersect1d(f_inf, f_sup)
    if len(overlap):
        print(f"  Removing {len(overlap)} overlapping facets")
        f_inf = f_inf[~np.isin(f_inf, overlap)]
        f_sup = f_sup[~np.isin(f_sup, overlap)]

    print(f"  f_inf: {len(f_inf)} | f_sup: {len(f_sup)}")

    idx   = np.concatenate([f_inf, f_sup])
    tags  = np.concatenate([np.ones(len(f_inf), np.int32),
                             np.full(len(f_sup), 2, np.int32)])
    order = np.argsort(idx)
    ft    = dmesh.meshtags(msh, fdim, idx[order], tags[order])
    ds    = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    dofs_inf = fem.locate_dofs_topological(V, fdim, f_inf)
    bc       = fem.dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), dofs_inf, V)

    sup_pts  = C[C[:, 2] >= zhi - tol_sup]
    p        = LOAD_N / PLATEAU_AREA_MM2             # N / mm² = MPa directly

    # 30° flexion → load has anterior shear (Y) + axial compression (Z)
    angle_rad = math.radians(FLEXION_DEG)
    p_shear = p * math.sin(angle_rad)              # anterior component
    p_axial = p * math.cos(angle_rad)              # compression component
    print(f"  Pressure: {p:.4f} MPa | area: {PLATEAU_AREA_MM2:.2f} mm²")
    print(f"  30° flexion → shear: {p_shear:.4f} MPa | axial: {p_axial:.4f} MPa")

    # Medial-biased traction with 30° flexion:
    #   70% medial / 30% lateral — drives tibial rotation via asymmetric load
    #   Anterior shear (Y) + axial compression (Z) — drives rotation via flexion
    x_med_split = float(np.median(sup_pts[:, 0]))

    u, v  = ufl.TrialFunction(V), ufl.TestFunction(V)
    eps   = lambda w: ufl.sym(ufl.grad(w))
    sigma = lambda w: LMBDA * ufl.tr(eps(w)) * ufl.Identity(3) + 2.0 * MU * eps(w)

    class MedialLateralLoad:
        """70/30 medial-lateral split with 30° flexion load vector."""
        def __call__(self, x):
            vals = np.zeros((3, x.shape[1]))
            is_medial = x[0] <= x_med_split
            # Medial: 70% of total (scale by 1.4x)
            # Lateral: 30% of total (scale by 0.6x)
            med_scale = 0.70 / 0.50
            lat_scale = 0.30 / 0.50
            # Y: anterior shear from flexion
            vals[1, is_medial]  = p_shear * med_scale
            vals[1, ~is_medial] = p_shear * lat_scale
            # Z: axial compression
            vals[2, is_medial]  = -p_axial * med_scale
            vals[2, ~is_medial] = -p_axial * lat_scale
            return vals

    t_expr = fem.Function(V)
    t_expr.interpolate(MedialLateralLoad())

    # Direct PETSc assembly + MUMPS LU — bypasses LinearProblem entirely
    a_compiled = fem.form(ufl.inner(sigma(u), eps(v)) * ufl.dx)
    L_compiled = fem.form(ufl.inner(t_expr, v) * ds(2))

    A = assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    b = assemble_vector(L_compiled)
    apply_lifting(b, [a_compiled], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc_set_bc(b, [bc])

    ksp = PETSc.KSP().create(MPI.COMM_WORLD)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setOperators(A)

    uh = fem.Function(V)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    U = uh.x.array.reshape(-1, 3)
    assert U.shape[0] == C.shape[0], "DOF/geometry node mismatch (P1 assumption violated)"

    u_max = np.abs(U).max()
    if u_max < 1e-12:
        raise RuntimeError("Solver returned zero displacement — check BCs/mesh")

    return uh, C, zlo, zhi, tol_sup


# ── 3. Metric extraction ──────────────────────────────────────────────────────
def extract_metrics(uh, C: np.ndarray, zlo: float, zhi: float, tol: float) -> dict:
    U      = uh.x.array.reshape(-1, 3)
    s_mask = C[:, 2] >= zhi - tol

    # ── Metric 1: Tibial rotation ─────────────────────────────────────────
    # Twist via medial vs lateral condyle displacement.
    # Centroid shift is ~0.001mm — arctan2 on that is pure noise.
    # Instead, measure angular change of the medial→lateral axis vector.
    sp = C[s_mask]; up = U[s_mask]
    x_split = float(np.median(sp[:, 0]))
    med_ctr_before = sp[sp[:, 0] <= x_split, :2].mean(0)
    lat_ctr_before = sp[sp[:, 0] >  x_split, :2].mean(0)
    med_disp = up[sp[:, 0] <= x_split, :2].mean(0)
    lat_disp = up[sp[:, 0] >  x_split, :2].mean(0)
    vec_before = lat_ctr_before - med_ctr_before
    vec_after  = (lat_ctr_before + lat_disp) - (med_ctr_before + med_disp)
    rot_deg = float(np.degrees(np.arctan2(
        vec_before[0]*vec_after[1] - vec_before[1]*vec_after[0],
        np.dot(vec_before, vec_after)
    )))

    # ── Metric 2: Medial compartment loading ──────────────────────────────
    x_med    = float(np.median(C[s_mask, 0]))
    m_mask   = s_mask & (C[:, 0] <= x_med)
    z_comp   = np.abs(U[m_mask, 2])              # already mm
    med_mean = float(z_comp.mean())
    med_max  = float(z_comp.max())

    # ── Metric 3: ACL tibial attachment displacement ──────────────────────
    sp     = C[s_mask]
    x_ref  = float(sp[:, 0].mean())
    y_ant  = float(sp[:, 1].min())
    y_ctr  = float(sp[:, 1].mean())
    acl_pt = np.array([x_ref, y_ant + 0.33 * (y_ctr - y_ant), zhi])
    ni     = int(np.argmin(np.linalg.norm(C - acl_pt, axis=1)))
    acl_disp = float(np.linalg.norm(U[ni]))       # already mm

    return {
        "tibial_rotation_deg":    round(rot_deg,  5),
        "medial_disp_mean_mm":    round(med_mean, 5),
        "medial_disp_max_mm":     round(med_max,  5),
        "acl_attachment_disp_mm": round(acl_disp, 5),
    }


# ── 4. Main ───────────────────────────────────────────────────────────────────
def main():
    results = {}

    print(f"\n── Geometric Tibial Rotation ──")
    geom_rot = compute_geometric_rotation()

    for name, fname in CASES.items():
        path = os.path.join(MESH_DIR, fname)
        print(f"\n{'─'*55}")
        print(f"  Case : {name}")
        print(f"  File : {fname}")

        print("  Loading mesh ...")
        msh = load_mesh(path)
        print(f"  Nodes: {msh.geometry.x.shape[0]:,}")

        print("  Solving linear elasticity ...")
        uh, C, zlo, zhi, tol = solve_elasticity(msh)
        u_max_mm = float(np.abs(uh.x.array).max())   # already mm
        print(f"  Max displacement : {u_max_mm:.4f} mm")

        print("  Extracting metrics ...")
        m = extract_metrics(uh, C, zlo, zhi, tol)
        results[name] = m

        for k, v in m.items():
            print(f"    {k}: {v}")

    # ── Intervention delta ────────────────────────────────────────────────
    if "baseline" in results and "with_insole" in results:
        b, w  = results["baseline"], results["with_insole"]
        delta = {k: round(w[k] - b[k], 5) for k in b}
        results["intervention_delta"] = delta

        print(f"\n{'═'*55}")
        print("  INTERVENTION DELTA  (with_insole − baseline)")
        print(f"{'─'*55}")
        for k, v in delta.items():
            sign = "+" if v >= 0 else ""
            print(f"    {k}: {sign}{v}")

    # ── Geometric rotation ─────────────────────────────────────────────
    results["geometric_tibial_rotation_deg"] = round(geom_rot, 5)

    # ── Save ─────────────────────────────────────────────────────────────
    out = os.path.join(OUT_DIR, "fea_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved → {out}\n")


if __name__ == "__main__":
    main()
