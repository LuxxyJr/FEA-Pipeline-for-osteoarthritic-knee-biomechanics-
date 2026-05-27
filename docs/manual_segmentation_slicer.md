# Manual Segmentation Handoff: Knee Bones

Automatic TotalSegmentator masks failed on this knee-only MRI, producing empty femur/tibia masks. Do not use those masks for registration, meshing, or FEA.

## Required Corrected Masks

Create these four files:

```text
data/segmentations/baseline/femur_left.nii.gz
data/segmentations/baseline/tibia_left.nii.gz
data/segmentations/with_insole/femur_left.nii.gz
data/segmentations/with_insole/tibia_left.nii.gz
```

## MRI Inputs

Use these processed volumes in 3D Slicer:

```text
data/nifti/series12_standing_BASELINE_processed.nii.gz
data/nifti/series9_standing_WITH_insole_processed.nii.gz
```

## Slicer Workflow

1. Open 3D Slicer.
2. Load one processed NIfTI volume.
3. Open `Segment Editor`.
4. Create two segments: `femur_left` and `tibia_left`.
5. Start with threshold/paint tools, then manually correct every relevant slice.
6. Keep masks binary and aligned to the source volume geometry.
7. Export each segment as NIfTI using the exact filenames above.
8. Repeat for the second scan.

## Acceptance Check

After exporting corrected masks, run:

```bash
python scripts/04b_segmentation_qc.py
```

Accept only if:

- Femur overlay sits on the upper bone.
- Tibia overlay sits on the lower bone.
- No mask is empty or tiny.
- No large mask spill into muscle/background.
