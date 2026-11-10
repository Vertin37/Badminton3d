"""End-to-end VDA depth, trajectory repair, and continuous foot contact.

The input video must already have synchronized stable 2-D pose and a GVHMR
court bundle, exactly like the existing project pipeline. No file names,
player IDs, frame count, or test video are hard-coded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_VDA_PYTHON = Path(r"D:\miniconda\envs\vda_win\python.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metric VDA + full-sequence multi-cue GVHMR trajectory repair")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose-csv", type=Path, required=True)
    parser.add_argument("--gvhmr-bundle", type=Path, required=True)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vda-python", type=Path, default=DEFAULT_VDA_PYTHON)
    parser.add_argument("--person-depth-csv", type=Path, help="Reuse existing observations and skip VDA inference")
    parser.add_argument("--depth-start-frame", type=int, default=1)
    parser.add_argument("--depth-end-frame", type=int)
    parser.add_argument("--fusion-start-frame", type=int, default=1)
    parser.add_argument("--fusion-end-frame", type=int)
    parser.add_argument("--bundle-start-frame", type=int, default=1)
    parser.add_argument("--vda-input-size", type=int, default=392)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT, check=True)


def run(args: argparse.Namespace) -> None:
    inputs = [args.video, args.pose_csv, args.gvhmr_bundle, args.scene_metadata]
    if args.person_depth_csv is not None:
        inputs.append(args.person_depth_csv)
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    if output_dir in {path.resolve().parent for path in inputs}:
        raise ValueError("Output directory must not be an input directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_csv = args.person_depth_csv.resolve() if args.person_depth_csv is not None else None
    depth_output = output_dir / "depth"
    if depth_csv is None:
        if not args.vda_python.exists():
            raise FileNotFoundError(args.vda_python)
        vda_command = [
            str(args.vda_python.resolve()),
            str(PROJECT / "work" / "video_depth_anything" / "run_vda_metric_depth.py"),
            "--video",
            str(args.video.resolve()),
            "--output-dir",
            str(depth_output),
            "--start-frame",
            str(args.depth_start_frame),
            "--input-size",
            str(args.vda_input_size),
        ]
        if args.depth_end_frame is not None:
            vda_command += ["--end-frame", str(args.depth_end_frame)]
        _run(vda_command)

        depth_csv = depth_output / "person_depth.csv"
        _run(
            [
                sys.executable,
                str(PROJECT / "work" / "depth_estimation" / "extract_person_depth.py"),
                "--pose-csv",
                str(args.pose_csv.resolve()),
                "--depth-raw",
                str(depth_output / "depth_raw.npy"),
                "--depth-quality",
                str(depth_output / "depth_quality.json"),
                "--output-csv",
                str(depth_csv),
            ]
        )

    fusion_output = output_dir / "fusion"
    fusion_command = [
        sys.executable,
        str(PROJECT / "work" / "trajectory_fusion" / "run_multicue_trajectory_fusion.py"),
        "--video",
        str(args.video.resolve()),
        "--pose-csv",
        str(args.pose_csv.resolve()),
        "--gvhmr-bundle",
        str(args.gvhmr_bundle.resolve()),
        "--scene-metadata",
        str(args.scene_metadata.resolve()),
        "--person-depth-csv",
        str(depth_csv),
        "--output-dir",
        str(fusion_output),
        "--start-frame",
        str(args.fusion_start_frame),
        "--bundle-start-frame",
        str(args.bundle_start_frame),
    ]
    if args.fusion_end_frame is not None:
        fusion_command += ["--end-frame", str(args.fusion_end_frame)]
    _run(fusion_command)

    fusion_bundle = fusion_output / "multicue_trajectory_fused_smplx_court_animation.npz"
    foot_output = output_dir / "foot_contact"
    _run(
        [
            sys.executable,
            str(PROJECT / "work" / "foot_contact" / "run_foot_contact_refinement.py"),
            "--bundle",
            str(fusion_bundle),
            "--scene-metadata",
            str(args.scene_metadata.resolve()),
            "--output-dir",
            str(foot_output),
        ]
    )

    final_bundle = foot_output / "foot_contact_refined_smplx_court_animation.npz"

    manifest = {
        "source_video": str(args.video.resolve()),
        "pose_csv": str(args.pose_csv.resolve()),
        "gvhmr_bundle": str(args.gvhmr_bundle.resolve()),
        "scene_metadata": str(args.scene_metadata.resolve()),
        "person_depth_csv": str(depth_csv),
        "depth_backend": "existing CSV" if args.person_depth_csv is not None else "official Metric Video Depth Anything Small",
        "depth_frame_range": [args.depth_start_frame, args.depth_end_frame],
        "fusion_frame_range": [args.fusion_start_frame, args.fusion_end_frame],
        "trajectory_fusion_bundle": str(fusion_bundle),
        "final_bundle": str(final_bundle),
        "comparison_metrics": str(fusion_output / "comparison_metrics.json"),
        "foot_contact_metrics": str(foot_output / "foot_contact_metrics.json"),
        "safety": {
            "inputs_overwritten": False,
            "remote_git_operations": False,
        },
    }
    (output_dir / "pipeline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
