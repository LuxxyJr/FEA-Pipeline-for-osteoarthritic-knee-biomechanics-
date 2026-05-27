# FEA Pipeline for Osteoarthritic Knee Biomechanics

A reproducible finite element analysis workflow for studying knee biomechanics under baseline and insole-assisted standing conditions from MRI-derived bone geometry. The pipeline converts DICOM image series into processed NIfTI volumes, supports manual femur/tibia segmentation, performs rigid registration, generates STL and tetrahedral Gmsh meshes, and runs a preliminary linear-elastic tibial loading model.

This project is designed as a research pipeline scaffold: raw medical images and derived volumetric/mesh assets are kept local, while scripts and lightweight numeric summaries are version-controlled for reproducibility.

---

## Project Overview

The workflow is organized into sequential stages:

**Image preparation** - Validates local DICOM series, converts target acquisitions to NIfTI, preprocesses intensities, and creates visual QC outputs.

**Bone segmentation** - Uses manual 3D Slicer segmentation for femur and tibia masks after automatic segmentation proved unsuitable for knee-only MRI.

**Registration and kinematics** - Registers with-insole bone masks to baseline masks and estimates tibia motion after removing femur/global pose.

**Mesh generation** - Converts binary masks into STL surfaces and tetrahedral Gmsh volume meshes for downstream simulation.

**FEA prototype** - Runs a simplified DOLFINx linear-elastic tibia loading model and exports displacement-based proxy metrics.

### Key Features

- **Privacy-preserving repo:** excludes raw DICOM, NIfTI, STL, MSH, and MRI QC images
- **Stepwise scripts:** numbered scripts from validation through FEA
- **Manual-segmentation support:** Slicer handoff and split-labelmap workflow
- **Registration outputs:** femur/tibia transforms and relative tibial rotation summary
- **Mesh pipeline:** marching-cubes STL generation and robust Gmsh tetrahedral meshing
- **FEA prototype:** DOLFINx-based linear elasticity solver for reproducible displacement proxies

---

## Architecture & Methods

### 1. Data Conversion and Preprocessing

The early pipeline validates the expected local scan set, groups DICOM slices by series UID suffix, converts selected scans to NIfTI, and applies preprocessing for downstream segmentation and registration.

| Script | Purpose |
|---|---|
| `scripts/00_validate_raw_data.py` | Validate local DICOM slice counts |
| `scripts/01_dicom_to_nifti.py` | Convert target DICOM series to NIfTI |
| `scripts/02_preprocess.py` | Normalize/preprocess converted volumes |
| `scripts/03_visual_qc.py` | Generate local processed-volume QC render |

Raw image paths are local-only. Set `RAW_DICOM_DIR` if your DICOM folder is not under `data/raw/`.

### 2. Segmentation

Automatic TotalSegmentator masks were not reliable for this knee-only MRI workflow, so the accepted path uses manual femur and tibia segmentation in 3D Slicer.

| Script | Purpose |
|---|---|
| `scripts/04a_auto_bone_segmentation.py` | Experimental automatic segmentation attempt |
| `scripts/04b_segmentation_qc.py` | Validate femur/tibia masks and create local QC |
| `scripts/05_split_slicer_labelmap.py` | Split exported Slicer labelmaps into per-bone masks |

Manual instructions are documented in `docs/manual_segmentation_slicer.md`.

### 3. Registration and Kinematics

Rigid registration is performed separately for femur and tibia masks. Femur motion is treated as the global pose component, and tibia motion relative to femur is reported as the insole-associated kinematic difference.

Current summary from `results/registration/rigid_registration_summary.json`:

| Metric | Value |
|---|---:|
| Femur moving-to-fixed rotation Z-like component | `5.6637 deg` |
| Tibia moving-to-fixed rotation Z-like component | `-3.5568 deg` |
| Relative tibia-vs-femur rotation Z-like component | `-9.2089 deg` |
| Relative flexion/extension-like component | `-3.3961 deg` |
| Relative varus/valgus-like component | `2.7636 deg` |

Interpretation should remain cautious: these are image-registration-derived kinematic estimates, not direct instrumented gait measurements.

### 4. Mesh Generation

Binary masks are converted to smooth STL surfaces and then tetrahedralized with Gmsh.

| Script | Purpose |
|---|---|
| `scripts/07_masks_to_stl.py` | Convert femur/tibia masks to STL surfaces |
| `scripts/08_mesh_qc_preview.py` | Generate local STL QC preview |
| `scripts/09_stl_to_gmsh.py` | Convert STL surfaces to `.msh` tetrahedral volume meshes |

Mesh files are excluded from git because they are derived medical geometry and can be large.

### 5. FEA Prototype

The current FEA model applies a simplified compressive/shear load to the tibial plateau using a linear-elastic cortical-bone material assumption. It reports displacement-based proxy metrics rather than validated clinical stress, contact pressure, or ligament strain.

Current summary from `results/fea/fea_results.json`:

| Metric | Baseline | With Insole | Delta |
|---|---:|---:|---:|
| Load-response tibial twist proxy (deg) | `0.00141` | `-0.00212` | `-0.00353` |
| Medial displacement mean (mm) | `0.00356` | `0.00212` | `-0.00144` |
| Medial displacement max (mm) | `0.00658` | `0.00676` | `0.00018` |
| Proxy ACL attachment displacement (mm) | `0.00710` | `0.00157` | `-0.00553` |

The registration-derived relative tibial rotation is the stronger kinematic result. The FEA outputs should be presented as prototype displacement proxies until cartilage, contact, ligament geometry, and boundary conditions are modeled more fully.

---

## Project Structure

```text
.
├── README.md
├── docs/
│   └── manual_segmentation_slicer.md
├── scripts/
│   ├── 00_validate_raw_data.py
│   ├── 01_dicom_to_nifti.py
│   ├── 02_preprocess.py
│   ├── 03_visual_qc.py
│   ├── 04a_auto_bone_segmentation.py
│   ├── 04b_segmentation_qc.py
│   ├── 05_split_slicer_labelmap.py
│   ├── 06_registration.py
│   ├── 07_masks_to_stl.py
│   ├── 08_mesh_qc_preview.py
│   ├── 09_stl_to_gmsh.py
│   └── 10_fea_solve.py
├── results/
│   ├── fea/
│   │   └── fea_results.json
│   └── registration/
│       ├── femur_left_with_insole_to_baseline.tfm
│       ├── tibia_left_with_insole_to_baseline.tfm
│       └── rigid_registration_summary.json
└── data/
    └── local only, not tracked
```

---

## Installation & Setup

Create a local Python/conda environment with the scientific imaging, meshing, and FEA packages needed by the scripts. Environment files are intentionally not tracked because local solver stacks and paths vary across machines.

Core packages used by the pipeline:

- `SimpleITK`
- `pydicom`
- `nibabel`
- `numpy`
- `scipy`
- `scikit-image`
- `matplotlib`
- `Pillow`
- `gmsh`
- `dolfinx`
- `mpi4py`
- `petsc4py`
- `basix`
- `ufl`

---

## Usage

Run from the project root.

### Phase 1: Validate and Convert Local DICOM Data

```bash
python scripts/00_validate_raw_data.py
python scripts/01_dicom_to_nifti.py
python scripts/02_preprocess.py
python scripts/03_visual_qc.py
```

### Phase 2: Manual Segmentation

Follow:

```text
docs/manual_segmentation_slicer.md
```

Then split and QC exported masks:

```bash
python scripts/05_split_slicer_labelmap.py
python scripts/04b_segmentation_qc.py
```

### Phase 3: Registration

```bash
python scripts/06_registration.py
```

Outputs:

- `results/registration/rigid_registration_summary.json`
- `results/registration/*.tfm`

### Phase 4: Mesh Generation

```bash
python scripts/07_masks_to_stl.py
python scripts/08_mesh_qc_preview.py
python scripts/09_stl_to_gmsh.py
```

### Phase 5: FEA Prototype

```bash
python scripts/10_fea_solve.py
```

Output:

- `results/fea/fea_results.json`

---

## Privacy and Data Governance

This repository intentionally excludes:

- raw DICOM files
- NIfTI volumes
- segmentation masks
- STL surface meshes
- Gmsh volume meshes
- MRI-derived PNG QC renders
- local environment files
- Python caches and editor files

Only source code, documentation, and lightweight numeric/text summaries are tracked.

---

## Current Limitations

- Manual segmentation quality controls downstream accuracy.
- The FEA model is a prototype linear-elastic tibia-only simulation.
- Cartilage, menisci, ligaments, contact mechanics, and subject-specific material properties are not yet modeled.
- Reported ACL-related values are proxy displacements near an estimated tibial attachment point, not ligament strain.
- Contact pressure and stress-map folders are placeholders unless separately generated.

---

## Roadmap

- Compute loaded plateau surface area directly from mesh facets.
- Export VTK/XDMF displacement fields for visual FEA review.
- Add cartilage/contact geometry when segmentation is available.
- Replace proxy ACL displacement with explicit ligament attachment modeling.
- Add automated non-sensitive result report generation.

---

## License

See `LICENSE`.

---

## Maintainer

Maintained as an academic research project on knee biomechanics and finite element analysis.
