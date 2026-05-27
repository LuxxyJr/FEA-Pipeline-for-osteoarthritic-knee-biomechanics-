"""
Script 04a: Attempt automatic femur/tibia segmentation for the two main cases.

Run from the project root after activating the medical imaging environment:
    python scripts/04a_auto_bone_segmentation.py

Important:
    This script intentionally processes only the primary comparison pair:
    - Series 12: standing baseline
    - Series 9: standing with insole

Always inspect the masks with scripts/04b_segmentation_qc.py before using them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES = {
    "baseline": PROJECT_ROOT / "data" / "nifti" / "series12_standing_BASELINE_processed.nii.gz",
    "with_insole": PROJECT_ROOT / "data" / "nifti" / "series9_standing_WITH_insole_processed.nii.gz",
}

# Keep this explicit. TotalSegmentator's MR appendicular-bone model names tibia
# as "tibia" without left/right. Because this scan volume contains one knee, we
# save that output as tibia_left.nii.gz to keep downstream filenames stable.
OUTPUT_MASKS = ["femur_left", "tibia_left"]

SEGMENTATION_JOBS = [
    {
        "task": "total_mr",
        "roi_subset": ["femur_left"],
        "fast": True,
        "output_map": {"femur_left.nii.gz": "femur_left.nii.gz"},
    },
    {
        "task": "appendicular_bones_mr",
        "roi_subset": None,
        "fast": False,
        "output_map": {
            "tibia.nii.gz": "tibia_left.nii.gz",
            "patella.nii.gz": "patella.nii.gz",
            "fibula.nii.gz": "fibula.nii.gz",
        },
    },
]


def totalsgmentator_command() -> str:
    command = shutil.which("TotalSegmentator") or shutil.which("totalsegmentator")
    if command is None:
        raise FileNotFoundError(
            "Could not find TotalSegmentator on PATH. Install/activate the environment "
            "that contains TotalSegmentator before running this script."
        )
    return command


def validate_inputs() -> None:
    for case_name, input_path in CASES.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Missing processed NIfTI for {case_name}: {input_path}")


def run_job(command: str, case_name: str, input_path: Path, job: dict[str, object]) -> None:
    output_dir = PROJECT_ROOT / "data" / "segmentations" / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        command,
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "--task",
        str(job["task"]),
    ]
    if job["roi_subset"] is not None:
        cmd.extend(["--roi_subset", *list(job["roi_subset"])])
    if job["fast"]:
        cmd.append("--fast")

    print(f"\nRunning TotalSegmentator for {case_name}: task={job['task']}")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"TotalSegmentator failed for {case_name}, task={job['task']} "
            f"with exit code {result.returncode}"
        )

    output_map = dict(job["output_map"])
    for source_name, final_name in output_map.items():
        source_path = output_dir / source_name
        final_path = output_dir / final_name
        if not source_path.exists():
            print(f"  Warning: expected optional mask was not created: {source_name}")
            continue
        if source_path != final_path:
            shutil.copy2(source_path, final_path)
            print(f"  Copied {source_name} -> {final_name}")


def run_case(command: str, case_name: str, input_path: Path) -> None:
    output_dir = PROJECT_ROOT / "data" / "segmentations" / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for mask_name in ["femur_left", "tibia_left", "tibia", "patella", "fibula"]:
        stale_path = output_dir / f"{mask_name}.nii.gz"
        if stale_path.exists():
            stale_path.unlink()

    for job in SEGMENTATION_JOBS:
        run_job(command, case_name, input_path, job)

    missing_required = []
    for mask_name in OUTPUT_MASKS:
        mask_path = output_dir / f"{mask_name}.nii.gz"
        if not mask_path.exists():
            missing_required.append(mask_name)

    if missing_required:
        raise RuntimeError(
            f"Missing required final masks for {case_name}: {', '.join(missing_required)}. "
            "This means automatic segmentation is not sufficient for this scan."
        )

    print(f"Saved masks to: {output_dir.relative_to(PROJECT_ROOT)}")


def main() -> None:
    validate_inputs()
    command = totalsgmentator_command()

    print(f"Using segmentation command: {command}")
    print(f"Final expected masks: {', '.join(OUTPUT_MASKS)}")
    for case_name, input_path in CASES.items():
        run_case(command, case_name, input_path)

    print("\nDone. Now run: python scripts/04b_segmentation_qc.py")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
