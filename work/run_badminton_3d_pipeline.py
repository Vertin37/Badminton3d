"""One-command Windows pipeline from a badminton video to the final 3-D bundle.

The program orchestrates the project's existing, tested stages without any
per-video manual edits: RTMLib tracking, court calibration, two independent
official GVHMR reconstructions, court-coordinate fusion, short-window Metric
Video Depth Anything, multi-cue trajectory repair, and support-foot refinement.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BADMINTON_PYTHON = Path(r"D:\miniconda\envs\badminton3d\python.exe")
DEFAULT_GVHMR_PYTHON = Path(r"D:\miniconda\envs\gvhmr_win\python.exe")
DEFAULT_VDA_PYTHON = Path(r"D:\miniconda\envs\vda_win\python.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic video -> stable two-player SMPL-X court animation"
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to outputs/<video-stem>_automatic_3d",
    )
    parser.add_argument("--badminton-python", type=Path, default=DEFAULT_BADMINTON_PYTHON)
    parser.add_argument("--gvhmr-python", type=Path, default=DEFAULT_GVHMR_PYTHON)
    parser.add_argument("--vda-python", type=Path, default=DEFAULT_VDA_PYTHON)
    parser.add_argument("--f-mm", type=int, default=24)
    parser.add_argument(
        "--moving-camera",
        action="store_true",
        help="Compute SimpleVO instead of using the default fixed-camera GVHMR mode",
    )
    parser.add_argument("--reference-frame0", type=int)
    parser.add_argument(
        "--depth-window-frames",
        type=int,
        default=75,
        help="Automatic high-motion VDA window; 0 processes the full video",
    )
    parser.add_argument("--vda-input-size", type=int, default=392)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete stage outputs recorded under an existing output directory",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _video_info(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    info = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if info["frames"] < 3 or not np.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError(f"Invalid video metadata: {info}")
    return info


def _ready(paths: Iterable[Path]) -> bool:
    paths = list(paths)
    return bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _court_calibration_available(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchor = payload.get("video_mapping_anchor") or {}
    if anchor.get("court_mapping", anchor.get("pole_mapped_court")) is not None:
        return True
    return any(
        item.get("court_mapping", item.get("pole_mapped_court")) is not None
        for item in payload.get("frames", [])
    )


def _court_calibration_rms(path: Path, focal_mm: int) -> float:
    """Measure whether the exported court corners form a usable camera pose."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("frames", []))
    if payload.get("video_mapping_anchor") is not None:
        records.append(payload["video_mapping_anchor"])
    records = [
        item
        for item in records
        if item.get("court_mapping", item.get("pole_mapped_court")) is not None
    ]
    if not records:
        return float("inf")
    target = int(payload.get("video_mapping_stats", {}).get("anchor_frame", records[0]["frame_index"]))
    record = min(records, key=lambda item: abs(int(item["frame_index"]) - target))
    mapped = record.get("court_mapping", record.get("pole_mapped_court"))
    object_points = np.c_[np.asarray(mapped["world_corners"], dtype=np.float64), np.zeros(4)]
    mapped_diagnostics = mapped.get("diagnostics", {})
    image_points = np.asarray(
        mapped_diagnostics.get("camera_pnp_image_corners", mapped["image_corners"]),
        dtype=np.float64,
    )
    if object_points.shape != (4, 3) or image_points.shape != (4, 2):
        return float("inf")
    detected_focal = mapped_diagnostics.get("approximate_camera_focal_px")
    if detected_focal is None:
        diagonal_px = float(np.hypot(payload["width"], payload["height"]))
        focal_px = diagonal_px * float(focal_mm) / float(np.hypot(24.0, 36.0))
    else:
        focal_px = float(detected_focal)
    k = np.array(
        [
            [focal_px, 0.0, 0.5 * float(payload["width"])],
            [0.0, focal_px, 0.5 * float(payload["height"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            k,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return float("inf")
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, None)
    except (cv2.error, ValueError, TypeError):
        return float("inf")
    error = projected.reshape(-1, 2) - image_points
    return float(np.sqrt(np.mean(np.sum(error * error, axis=1))))


def _court_calibration_valid(path: Path, focal_mm: int, max_rms_px: float = 40.0) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        int(payload.get("court_mapping_schema", 0)) >= 2
        and _court_calibration_available(path)
        and _court_calibration_rms(path, focal_mm) <= max_rms_px
    )


def _write_status(path: Path, status: dict) -> None:
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_stage(
    name: str,
    command: list[str],
    expected: list[Path],
    status: dict,
    status_path: Path,
    resume: bool,
) -> None:
    if resume and _ready(expected):
        print(f"[SKIP] {name}: complete outputs already exist", flush=True)
        status["stages"][name] = {
            "state": "reused",
            "completed_at": _utc_now(),
            "outputs": [str(path) for path in expected],
        }
        _write_status(status_path, status)
        return

    print(f"[RUN] {name}", flush=True)
    status["current_stage"] = name
    status["stages"][name] = {
        "state": "running",
        "started_at": _utc_now(),
        "command": command,
    }
    _write_status(status_path, status)
    try:
        subprocess.run(command, cwd=PROJECT, check=True)
    except BaseException as exc:
        status["stages"][name]["state"] = "failed"
        status["stages"][name]["failed_at"] = _utc_now()
        status["stages"][name]["error"] = f"{type(exc).__name__}: {exc}"
        status["state"] = "failed"
        _write_status(status_path, status)
        raise
    if not _ready(expected):
        raise RuntimeError(f"Stage {name} completed without expected outputs: {expected}")
    status["stages"][name] = {
        "state": "complete",
        "completed_at": _utc_now(),
        "outputs": [str(path) for path in expected],
    }
    _write_status(status_path, status)


def _select_depth_window(pose_csv: Path, frame_count: int, requested: int) -> tuple[int, int, dict]:
    if requested <= 0 or requested >= frame_count:
        return 1, frame_count, {"method": "full_video", "window_frames": frame_count}
    window = min(frame_count, max(50, min(100, requested)))
    points: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    with pose_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                keypoint = int(row["keypoint_id"])
                if keypoint not in (11, 12):
                    continue
                frame = int(row["frame_id"])
                player = int(row["player_id"])
                confidence = float(row["confidence"])
                x, y = float(row["x"]), float(row["y"])
                scale = float(row.get("body_scale", 100.0))
            except (KeyError, TypeError, ValueError):
                continue
            if player not in (0, 1) or confidence < 0.2 or not np.isfinite([x, y, scale]).all():
                continue
            points.setdefault((frame, player), []).append((x, y, confidence, max(scale, 20.0)))

    centers: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    for key, values in points.items():
        array = np.asarray(values, dtype=np.float64)
        centers[key] = (np.average(array[:, :2], axis=0, weights=array[:, 2]), float(np.median(array[:, 3])))

    motion = np.zeros(frame_count, dtype=np.float64)
    coverage = np.zeros(frame_count, dtype=np.float64)
    for frame in range(1, frame_count + 1):
        for player in (0, 1):
            current = centers.get((frame, player))
            previous = centers.get((frame - 1, player))
            if current is not None:
                coverage[frame - 1] += 0.5
            if current is None or previous is None:
                continue
            scale = max(0.5 * (current[1] + previous[1]), 20.0)
            motion[frame - 1] += float(np.linalg.norm(current[0] - previous[0]) / scale)
    positive = motion[motion > 0]
    if positive.size:
        motion = np.minimum(motion, np.percentile(positive, 95))
    score = np.convolve(motion, np.ones(window), mode="valid")
    window_coverage = np.convolve(coverage, np.ones(window) / window, mode="valid")
    score *= 0.25 + 0.75 * window_coverage
    start0 = int(np.argmax(score)) if score.size else max(0, (frame_count - window) // 2)
    return (
        start0 + 1,
        start0 + window,
        {
            "method": "highest_robust_normalized_pelvis_motion",
            "window_frames": window,
            "mean_player_coverage": float(window_coverage[start0]) if window_coverage.size else 0.0,
            "motion_score": float(score[start0]) if score.size else 0.0,
        },
    )


def run(args: argparse.Namespace) -> None:
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT / "outputs" / f"{video.stem}_automatic_3d").resolve()
    )
    interpreters = [args.badminton_python.resolve(), args.gvhmr_python.resolve(), args.vda_python.resolve()]
    for interpreter in interpreters:
        if not interpreter.is_file():
            raise FileNotFoundError(interpreter)
    if output_dir == video.parent:
        raise ValueError("Output directory must not be the input video directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    info = _video_info(video)
    frame_count = int(info["frames"])
    fps = float(info["fps"])
    reference_frame0 = (
        int(args.reference_frame0) if args.reference_frame0 is not None else (frame_count - 1) // 2
    )
    if not 0 <= reference_frame0 < frame_count:
        raise ValueError(f"Reference frame {reference_frame0} is outside 0..{frame_count - 1}")

    status_path = output_dir / "pipeline_status.json"
    status = {
        "state": "running",
        "started_at": _utc_now(),
        "input_video": str(video),
        "output_dir": str(output_dir),
        "video": info,
        "configuration": {
            "fixed_camera": not args.moving_camera,
            "f_mm": args.f_mm,
            "reference_frame0": reference_frame0,
            "depth_window_frames": args.depth_window_frames,
            "vda_input_size": args.vda_input_size,
        },
        "current_stage": None,
        "stages": {},
    }
    _write_status(status_path, status)

    pose_dir = output_dir / "01_pose"
    pose_csv = pose_dir / "pose_data_stable.csv"
    _run_stage(
        "pose_tracking",
        [
            str(args.badminton_python.resolve()),
            str(PROJECT / "run_pose_test.py"),
            "--video",
            str(video),
            "--output-dir",
            str(pose_dir),
            "--device",
            "cuda",
            "--no-display",
        ],
        [pose_csv, pose_dir / "temporal_filter_stats.json"],
        status,
        status_path,
        args.resume,
    )

    court_dir = output_dir / "02_court"
    court_tracking_json = court_dir / "court_detection_lines.json"
    court_track_csv = court_dir / "player_pole_geometry.csv"
    _run_stage(
        "court_calibration",
        [
            str(args.badminton_python.resolve()),
            str(PROJECT / "run_court_detection.py"),
            "--video",
            str(video),
            "--output-dir",
            str(court_dir),
            "--keyframes",
            "3",
            "--pose-csv",
            str(pose_csv),
            "--no-labels",
            "--no-edge-preview",
            "--all-frames",
            "--video-output",
            str(court_dir / "court_detection_video.mp4"),
            "--reference-frame",
            str(reference_frame0),
            "--pole-mapping",
        ] + ([] if args.moving_camera else ["--static-camera"]),
        [court_tracking_json, court_track_csv, court_dir / "court_detection_video.mp4"],
        status,
        status_path,
        args.resume,
    )

    if not _court_calibration_valid(court_tracking_json, args.f_mm):
        validated_court_dir = output_dir / "02_court_validated_v2"
        validated_json = validated_court_dir / "court_detection_lines.json"
        validated_track_csv = validated_court_dir / "player_pole_geometry.csv"
        _run_stage(
            "court_calibration_validated_retry",
            [
                str(args.badminton_python.resolve()),
                str(PROJECT / "run_court_detection.py"),
                "--video",
                str(video),
                "--output-dir",
                str(validated_court_dir),
                "--keyframes",
                "3",
                "--pose-csv",
                str(pose_csv),
                "--no-labels",
                "--no-edge-preview",
                "--all-frames",
                "--video-output",
                str(validated_court_dir / "court_detection_video.mp4"),
                "--reference-frame",
                str(reference_frame0),
                "--pole-mapping",
            ] + ([] if args.moving_camera else ["--static-camera"]),
            [validated_json, validated_track_csv, validated_court_dir / "court_detection_video.mp4"],
            status,
            status_path,
            args.resume,
        )
        court_dir = validated_court_dir
        court_tracking_json = validated_json
        court_track_csv = validated_track_csv

    court_json = court_tracking_json
    if not _court_calibration_valid(court_tracking_json, args.f_mm):
        tracking_payload = json.loads(court_tracking_json.read_text(encoding="utf-8"))
        detected_anchor = tracking_payload.get("video_mapping_stats", {}).get("anchor_frame")
        if detected_anchor is None:
            raise RuntimeError("Court tracking produced no calibration frame")
        anchor_dir = output_dir / "02_court_anchor_validated_v2"
        anchor_json = anchor_dir / "court_detection_lines.json"
        _run_stage(
            "court_anchor_export",
            [
                str(args.badminton_python.resolve()),
                str(PROJECT / "run_court_detection.py"),
                "--video",
                str(video),
                "--output-dir",
                str(anchor_dir),
                "--frame",
                str(int(detected_anchor)),
                "--pose-csv",
                str(pose_csv),
                "--no-labels",
                "--no-edge-preview",
                "--pole-mapping",
            ],
            [anchor_json],
            status,
            status_path,
            args.resume,
        )
        if not _court_calibration_valid(anchor_json, args.f_mm):
            rms = _court_calibration_rms(anchor_json, args.f_mm)
            raise RuntimeError(
                f"Detected court anchor {detected_anchor} failed calibration validation "
                f"(reprojection RMS {rms:.2f}px)"
            )
        court_json = anchor_json

    native_results: list[Path] = []
    for player in (0, 1):
        cache_parent = output_dir / "03_gvhmr_cache" / f"p{player}"
        cache_root = cache_parent / video.stem
        preprocess = cache_root / "preprocess"
        adapter_command = [
            str(args.gvhmr_python.resolve()),
            str(PROJECT / "work" / "rtmlib_to_gvhmr_adapter.py"),
            "--csv",
            str(pose_csv),
            "--video",
            str(video),
            "--clip",
            str(video),
            "--output-root",
            str(cache_parent),
            "--player-id",
            str(player),
            "--start-frame0",
            "0",
            "--end-frame0",
            str(frame_count),
            "--extract-features",
            "--f-mm",
            str(args.f_mm),
        ]
        cache_expected = [preprocess / "bbx.pt", preprocess / "vitpose.pt", preprocess / "vit_features.pt"]
        if args.moving_camera:
            adapter_command.append("--compute-simplevo")
            cache_expected.append(preprocess / "slam_results.pt")
        _run_stage(
            f"gvhmr_cache_p{player}",
            adapter_command,
            cache_expected,
            status,
            status_path,
            args.resume,
        )

        native_dir = output_dir / "04_gvhmr_native" / f"p{player}"
        native_result = native_dir / "1_hmr4d_results_native.pt"
        native_command = [
            str(args.gvhmr_python.resolve()),
            str(PROJECT / "work" / "run_gvhmr_native_mesh_slice.py"),
            "--video",
            str(video),
            "--cache-root",
            str(cache_root),
            "--output-dir",
            str(native_dir),
            "--start-frame0",
            "0",
            "--end-frame0",
            str(frame_count),
            "--f-mm",
            str(args.f_mm),
        ]
        if not args.moving_camera:
            native_command.append("--static-camera")
        _run_stage(
            f"gvhmr_native_p{player}",
            native_command,
            [native_result, native_dir / "2_smplx_mesh_global.npz"],
            status,
            status_path,
            args.resume,
        )
        native_results.append(native_result)

    bundle_dir = output_dir / "05_court_bundle_v3"
    bundle = bundle_dir / "two_player_smplx_court_animation.npz"
    scene_metadata = bundle_dir / "scene_metadata.json"
    bundle_command = [
        str(args.gvhmr_python.resolve()),
        str(PROJECT / "work" / "build_gvhmr_two_player_court_bundle.py"),
        "--p0-result",
        str(native_results[0]),
        "--p1-result",
        str(native_results[1]),
        "--court-json",
        str(court_json),
        "--court-track-csv",
        str(court_track_csv),
        "--output-dir",
        str(bundle_dir),
        "--source-start-frame0",
        "0",
        "--f-mm",
        str(args.f_mm),
        "--fps",
        str(fps),
    ]
    if args.reference_frame0 is not None:
        bundle_command += ["--anchor-frame0", str(reference_frame0)]
    _run_stage(
        "court_bundle",
        bundle_command,
        [bundle, scene_metadata],
        status,
        status_path,
        args.resume,
    )

    depth_start, depth_end, depth_selection = _select_depth_window(
        pose_csv, frame_count, args.depth_window_frames
    )
    status["depth_selection"] = {
        "start_frame_1based": depth_start,
        "end_frame_1based": depth_end,
        **depth_selection,
    }
    _write_status(status_path, status)

    final_dir = output_dir / "06_final_v5"
    final_bundle = final_dir / "foot_contact" / "foot_contact_refined_smplx_court_animation.npz"
    _run_stage(
        "depth_trajectory_and_foot_refinement",
        [
            str(args.gvhmr_python.resolve()),
            str(PROJECT / "work" / "video_depth_anything" / "run_vda_multicue_pipeline.py"),
            "--video",
            str(video),
            "--pose-csv",
            str(pose_csv),
            "--gvhmr-bundle",
            str(bundle),
            "--scene-metadata",
            str(scene_metadata),
            "--output-dir",
            str(final_dir),
            "--vda-python",
            str(args.vda_python.resolve()),
            "--depth-start-frame",
            str(depth_start),
            "--depth-end-frame",
            str(depth_end),
            "--fusion-start-frame",
            "1",
            "--fusion-end-frame",
            str(frame_count),
            "--bundle-start-frame",
            "1",
            "--vda-input-size",
            str(args.vda_input_size),
        ],
        [
            final_bundle,
            final_dir / "fusion" / "comparison_metrics.json",
            final_dir / "foot_contact" / "foot_contact_metrics.json",
        ],
        status,
        status_path,
        args.resume,
    )

    manifest = {
        "input_video": str(video),
        "output_dir": str(output_dir),
        "video": info,
        "pose_csv": str(pose_csv),
        "court_json": str(court_json),
        "court_tracking_json": str(court_tracking_json),
        "court_track_csv": str(court_track_csv),
        "native_results": [str(path) for path in native_results],
        "initial_court_bundle": str(bundle),
        "depth_selection": status["depth_selection"],
        "final_bundle": str(final_bundle),
        "viewer_command": [
            str(args.gvhmr_python.resolve()),
            str(PROJECT / "work" / "view_gvhmr_two_player_court.py"),
            "--bundle",
            str(final_bundle),
        ],
        "manual_per_video_corrections": False,
        "remote_git_operations": False,
    }
    manifest_path = output_dir / "automatic_pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    status["state"] = "complete"
    status["current_stage"] = None
    status["completed_at"] = _utc_now()
    status["manifest"] = str(manifest_path)
    _write_status(status_path, status)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
