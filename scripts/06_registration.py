"""
Script 06: Rigid registration between Series 9 and Series 12 bone masks.

Registers with-insole masks to baseline masks for femur and tibia separately,
then reports the tibia motion after removing femur/global motion.

Run:
    python scripts/06_registration.py

Outputs:
    results/registration/rigid_registration_summary.json
    results/registration/registration_checkerboard.png
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_MRI = PROJECT_ROOT / "data" / "nifti" / "series12_standing_BASELINE_processed.nii.gz"
INSOLE_MRI = PROJECT_ROOT / "data" / "nifti" / "series9_standing_WITH_insole_processed.nii.gz"

MASKS = {
    "femur_left": (
        PROJECT_ROOT / "data" / "segmentations" / "baseline" / "femur_left.nii.gz",
        PROJECT_ROOT / "data" / "segmentations" / "with_insole" / "femur_left.nii.gz",
    ),
    "tibia_left": (
        PROJECT_ROOT / "data" / "segmentations" / "baseline" / "tibia_left.nii.gz",
        PROJECT_ROOT / "data" / "segmentations" / "with_insole" / "tibia_left.nii.gz",
    ),
}

OUTPUT_DIR = PROJECT_ROOT / "results" / "registration"
SUMMARY_PATH = OUTPUT_DIR / "rigid_registration_summary.json"
QC_PATH = OUTPUT_DIR / "registration_checkerboard.png"


def read_mask(path: Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing mask: {path}")
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(image)
    if int((arr > 0).sum()) == 0:
        raise ValueError(f"Mask is empty: {path}")
    return sitk.Cast(image > 0, sitk.sitkFloat32)


def mask_to_distance(mask: sitk.Image) -> sitk.Image:
    """Convert a binary mask to a signed distance map for smoother registration."""
    distance = sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask > 0, sitk.sitkUInt8),
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return sitk.Cast(distance, sitk.sitkFloat32)


def rotation_matrix_to_euler_xyz(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return XYZ Euler angles in degrees for a 3x3 rotation matrix."""
    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(matrix[2, 1], matrix[2, 2])
        y = math.atan2(-matrix[2, 0], sy)
        z = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        x = math.atan2(-matrix[1, 2], matrix[1, 1])
        y = math.atan2(-matrix[2, 0], sy)
        z = 0.0

    return tuple(math.degrees(v) for v in (x, y, z))


def transform_matrix(transform: sitk.Transform) -> np.ndarray:
    return np.array(transform.GetMatrix(), dtype=float).reshape(3, 3)


def transform_translation(transform: sitk.Transform) -> list[float]:
    return [float(v) for v in transform.GetTranslation()]


def as_euler3d(transform: sitk.Transform) -> sitk.Euler3DTransform:
    """Unwrap SimpleITK registration outputs into an Euler3DTransform."""
    if isinstance(transform, sitk.Euler3DTransform):
        return transform

    if transform.GetName() == "CompositeTransform":
        composite = sitk.CompositeTransform(transform)
        if composite.GetNumberOfTransforms() == 0:
            raise ValueError("Registration returned an empty CompositeTransform")
        return as_euler3d(composite.GetBackTransform())

    return sitk.Euler3DTransform(transform)


def register_binary_masks(fixed: sitk.Image, moving: sitk.Image, name: str) -> sitk.Euler3DTransform:
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    fixed_distance = mask_to_distance(fixed)
    moving_distance = mask_to_distance(moving)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.50, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-5,
        numberOfIterations=500,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)

    print(f"\nRegistering {name}: with_insole -> baseline")
    final_transform = registration.Execute(fixed_distance, moving_distance)
    final = as_euler3d(final_transform)
    print(f"  Final metric: {registration.GetMetricValue():.6f}")
    print(f"  Stop: {registration.GetOptimizerStopConditionDescription()}")
    print(f"  Translation mm: {transform_translation(final)}")
    print(f"  Euler XYZ deg: {rotation_matrix_to_euler_xyz(transform_matrix(final))}")
    return final


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice_2d, (1, 99))
    return (np.clip((slice_2d - lo) / (hi - lo + 1e-8), 0.0, 1.0) * 255).astype(np.uint8)


def make_checkerboard_qc(femur_transform: sitk.Transform) -> None:
    fixed = sitk.ReadImage(str(BASELINE_MRI), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(INSOLE_MRI), sitk.sitkFloat32)
    moved = sitk.Resample(moving, fixed, femur_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

    fixed_arr = sitk.GetArrayFromImage(fixed)
    moved_arr = sitk.GetArrayFromImage(moved)
    z = fixed_arr.shape[0] // 2
    a = normalize_slice(fixed_arr[z])
    b = normalize_slice(moved_arr[z])

    block = 32
    checker = a.copy()
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            if ((x // block) + (y // block)) % 2:
                checker[y, x] = b[y, x]

    image = Image.fromarray(checker).convert("RGB").resize((768, 768))
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), "Checkerboard: baseline vs femur-registered with_insole", fill=(255, 255, 255))
    image.save(QC_PATH)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transforms: dict[str, sitk.Euler3DTransform] = {}
    summary: dict[str, object] = {}

    for name, (fixed_path, moving_path) in MASKS.items():
        fixed = read_mask(fixed_path)
        moving = read_mask(moving_path)
        transform = register_binary_masks(fixed, moving, name)
        transforms[name] = transform
        transform_path = OUTPUT_DIR / f"{name}_with_insole_to_baseline.tfm"
        sitk.WriteTransform(transform, str(transform_path))

        matrix = transform_matrix(transform)
        summary[name] = {
            "moving_to_fixed_translation_mm": transform_translation(transform),
            "moving_to_fixed_euler_xyz_deg": list(rotation_matrix_to_euler_xyz(matrix)),
            "moving_to_fixed_rotation_matrix": matrix.tolist(),
            "transform_file": str(transform_path.relative_to(PROJECT_ROOT)),
        }

    femur_r = transform_matrix(transforms["femur_left"])
    tibia_r = transform_matrix(transforms["tibia_left"])
    relative_r = femur_r.T @ tibia_r
    relative_xyz = rotation_matrix_to_euler_xyz(relative_r)

    summary["tibia_relative_to_femur_insole_effect"] = {
        "relative_euler_xyz_deg": list(relative_xyz),
        "interpretation": {
            "x_deg": "flexion_extension_like_axis",
            "y_deg": "varus_valgus_like_axis",
            "z_deg": "internal_external_tibial_rotation_like_axis",
        },
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_checkerboard_qc(transforms["femur_left"])

    print(f"\nSaved summary: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved checkerboard QC: {QC_PATH.relative_to(PROJECT_ROOT)}")
    print("\nRelative tibia-vs-femur rotation XYZ deg:", relative_xyz)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
