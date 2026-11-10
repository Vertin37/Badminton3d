"""Apply automatic depth-guided drift constraints to a GVHMR court bundle.

This is the production entry point. It accepts arbitrary video, pose CSV,
GVHMR bundle, and scene calibration paths; no frame range or player identity is
hard-coded. Depth inference and trajectory correction run automatically.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.pose_biomechanics.depth_constraint import (  # noqa: E402
    DepthConstraintConfig,
    camera_depth_delta_to_court_xy,
    clip_court_xy_correction,
    constrain_camera_depth,
    discover_player_ids,
    trajectory_metrics,
)


DEFAULT_VENDOR = PROJECT / "work" / "depth_estimation" / "vendor"
TORSO_KEYPOINTS = (5, 6, 11, 12)
TORSO_POLYGON_ORDER = (5, 6, 12, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constrain short-time GVHMR camera-distance drift with monocular depth.",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose-csv", type=Path, help="Stable COCO/WholeBody pose CSV; required for depth inference")
    parser.add_argument("--gvhmr-bundle", type=Path, required=True)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--person-depth-csv", type=Path, help="Reuse observations and skip GPU depth inference")
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional legacy Ultralytics depth checkpoint; prefer an externally generated --person-depth-csv",
    )
    parser.add_argument("--vendor-dir", type=Path, default=DEFAULT_VENDOR, help="Optional legacy Ultralytics vendor")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--start-frame", type=int, default=1, help="Video frame, 1-based inclusive")
    parser.add_argument("--end-frame", type=int, help="Video frame, 1-based inclusive; defaults to all overlapping frames")
    parser.add_argument(
        "--bundle-start-frame",
        type=int,
        default=1,
        help="Video frame corresponding to bundle array index 0",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save-raw-depth", action="store_true")
    parser.add_argument("--max-correction-m", type=float)
    parser.add_argument("--max-correction-speed-mps", type=float)
    parser.add_argument("--max-unexplained-depth-speed-mps", type=float)
    parser.add_argument("--depth-weight", type=float)
    parser.add_argument("--gvhmr-weight", type=float)
    return parser.parse_args()


def _validated_config(args: argparse.Namespace) -> DepthConstraintConfig:
    config = DepthConstraintConfig()
    overrides = {
        "max_correction_m": args.max_correction_m,
        "max_correction_speed_mps": args.max_correction_speed_mps,
        "max_unexplained_depth_speed_mps": args.max_unexplained_depth_speed_mps,
        "depth_weight": args.depth_weight,
        "gvhmr_weight": args.gvhmr_weight,
    }
    values = {key: value for key, value in overrides.items() if value is not None}
    if values:
        config = replace(config, **values)
    if config.max_correction_m <= 0 or config.max_correction_speed_mps <= 0:
        raise ValueError("Correction limits must be positive")
    if config.depth_weight <= 0 or config.gvhmr_weight <= 0:
        raise ValueError("Depth and GVHMR weights must be positive")
    return config


def _video_info(video: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    info = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if info["frames"] <= 0 or info["fps"] <= 0 or info["width"] <= 0 or info["height"] <= 0:
        raise ValueError(f"Invalid video metadata: {info}")
    return info


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}. "
            "Choose a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics").mkdir(parents=True, exist_ok=True)


def _pose_lookup(pose: pd.DataFrame) -> dict[tuple[int, int], pd.DataFrame]:
    return {
        (int(frame_id), int(player_id)): group
        for (frame_id, player_id), group in pose.groupby(["frame_id", "player_id"], sort=False)
    }


def _torso_region(group: pd.DataFrame, width: int, height: int) -> tuple[tuple[int, int, int, int], np.ndarray, float, int]:
    """Build a torso/pelvis mask from COCO shoulder and hip keypoints."""
    points: dict[int, tuple[float, float, float]] = {}
    indexed = group[group.keypoint_id.isin(TORSO_KEYPOINTS)].set_index("keypoint_id")
    for keypoint_id in TORSO_KEYPOINTS:
        if keypoint_id not in indexed.index:
            continue
        row = indexed.loc[keypoint_id]
        x, y = float(row.x), float(row.y)
        confidence = float(np.clip(row.confidence, 0.0, 1.0))
        if np.isfinite(x) and np.isfinite(y) and confidence >= 0.2:
            points[keypoint_id] = (x, y, confidence)

    scale_values = group.body_scale.to_numpy(dtype=float) if "body_scale" in group else np.array([])
    scale_values = scale_values[np.isfinite(scale_values) & (scale_values > 1)]
    body_scale = float(np.median(scale_values)) if scale_values.size else 100.0
    mask = np.zeros((height, width), dtype=np.uint8)
    if all(keypoint_id in points for keypoint_id in TORSO_POLYGON_ORDER):
        polygon = np.asarray([[points[key][0], points[key][1]] for key in TORSO_POLYGON_ORDER], dtype=np.float64)
        center = polygon.mean(axis=0)
        polygon = center + (polygon - center) * np.array([1.15, 1.10])
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        integer_polygon = np.round(polygon).astype(np.int32)
        cv2.fillConvexPoly(mask, integer_polygon, 1)
        x1, y1 = integer_polygon.min(axis=0)
        x2, y2 = integer_polygon.max(axis=0) + 1
    elif len(points) >= 2:
        coordinates = np.asarray([[value[0], value[1]] for value in points.values()], dtype=float)
        x_span = max(float(np.ptp(coordinates[:, 0])), 0.35 * body_scale, 20.0)
        y_span = max(float(np.ptp(coordinates[:, 1])), 0.45 * body_scale, 30.0)
        x1 = int(np.clip(np.floor(coordinates[:, 0].min() - 0.18 * x_span), 0, width - 1))
        x2 = int(np.clip(np.ceil(coordinates[:, 0].max() + 0.18 * x_span), x1 + 1, width))
        y1 = int(np.clip(np.floor(coordinates[:, 1].min() - 0.15 * y_span), 0, height - 1))
        y2 = int(np.clip(np.ceil(coordinates[:, 1].max() + 0.18 * y_span), y1 + 1, height))
        mask[y1:y2, x1:x2] = 1
    else:
        return (0, 0, 0, 0), mask, 0.0, len(points)

    keypoint_confidence = float(
        np.mean([value[2] for value in points.values()]) * len(points) / len(TORSO_KEYPOINTS)
    )
    return (int(x1), int(y1), int(x2), int(y2)), mask, keypoint_confidence, len(points)


def _depth_row(
    frame_id: int,
    player_id: int,
    pose_group: pd.DataFrame | None,
    depth_map: np.ndarray,
) -> dict[str, Any]:
    height, width = depth_map.shape
    if pose_group is None or pose_group.empty:
        box, mask, kp_conf, keypoint_count = (0, 0, 0, 0), np.zeros_like(depth_map, dtype=np.uint8), 0.0, 0
    else:
        box, mask, kp_conf, keypoint_count = _torso_region(pose_group, width, height)
    valid = (mask > 0) & np.isfinite(depth_map) & (depth_map > 0)
    values = depth_map[valid]
    region_pixels = int(np.count_nonzero(mask))
    valid_ratio = float(values.size / region_pixels) if region_pixels else 0.0
    if values.size:
        q05, q25, q75, q95 = np.percentile(values, [5, 25, 75, 95])
        trimmed = values[(values >= q05) & (values <= q95)]
        median = float(np.median(values))
        trimmed_mean = float(np.mean(trimmed if trimmed.size else values))
        iqr = float(q75 - q25)
    else:
        median = trimmed_mean = iqr = float("nan")
    dispersion_quality = (
        float(np.clip(1.0 - iqr / max(median, 1.0), 0.0, 1.0)) if np.isfinite(median) else 0.0
    )
    roi_confidence = float(kp_conf * valid_ratio * dispersion_quality)
    x1, y1, x2, y2 = box
    return {
        "frame_id": frame_id,
        "player_id": player_id,
        "roi_type": "torso_pelvis_polygon",
        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2,
        "valid_pixel_count": int(values.size),
        "valid_ratio": valid_ratio,
        "depth_median": median,
        "depth_trimmed_mean": trimmed_mean,
        "depth_iqr": iqr,
        "keypoint_confidence": kp_conf,
        "valid_body_keypoint_count": keypoint_count,
        "roi_confidence": roi_confidence,
        "occlusion_or_missing_flag": bool(keypoint_count < 3 or valid_ratio < 0.5),
    }


def _depth_tensor_to_numpy(result: Any, width: int, height: int) -> np.ndarray:
    depth_object = getattr(result, "depth", None)
    if depth_object is None:
        raise RuntimeError("Ultralytics result does not contain result.depth")
    data = getattr(depth_object, "data", depth_object)
    if hasattr(data, "detach"):
        depth = data.detach().float().cpu().numpy()
    else:
        depth = np.asarray(data, dtype=np.float32)
    depth = np.squeeze(depth).astype(np.float32, copy=False)
    if depth.ndim != 2:
        raise RuntimeError(f"Unexpected depth output shape: {depth.shape}")
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    return depth


def infer_person_depth(
    args: argparse.Namespace,
    pose: pd.DataFrame,
    player_ids: list[int],
    frame_ids: np.ndarray,
    video_info: dict[str, float | int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.pose_csv is None:
        raise ValueError("--pose-csv is required unless --person-depth-csv is supplied")
    if args.vendor_dir.exists():
        sys.path.insert(0, str(args.vendor_dir.resolve()))
    yolo_config_dir = args.output_dir / "ultralytics_config"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    # Keep Ultralytics' mutable settings beside this run instead of polluting
    # the repository root or the user's global profile.
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir.resolve())
    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(args.model))
    if getattr(model, "task", None) != "depth":
        raise RuntimeError(f"Model task is not depth: {getattr(model, 'task', None)!r}")
    capture = cv2.VideoCapture(str(args.video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_ids[0]) - 1)
    width, height = int(video_info["width"]), int(video_info["height"])
    preview = None
    if not args.no_preview:
        preview = cv2.VideoWriter(
            str(args.output_dir / "depth_preview.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(video_info["fps"]),
            (width * 2, height),
        )
        if not preview.isOpened():
            capture.release()
            raise RuntimeError("Could not create depth_preview.mp4")
    raw_depth = None
    if args.save_raw_depth:
        raw_depth = np.lib.format.open_memmap(
            args.output_dir / "depth_raw.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(frame_ids), height, width),
        )
    lookup = _pose_lookup(pose)
    rows: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for offset, frame_id in enumerate(frame_ids):
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read video frame {frame_id}")
                prediction = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    device=args.device,
                    batch=1,
                    verbose=False,
                    stream=False,
                )
                if not prediction:
                    raise RuntimeError(f"No depth result for frame {frame_id}")
                depth_map = _depth_tensor_to_numpy(prediction[0], width, height)
                if raw_depth is not None:
                    raw_depth[offset] = depth_map
                for player_id in player_ids:
                    rows.append(_depth_row(int(frame_id), player_id, lookup.get((int(frame_id), player_id)), depth_map))
                if preview is not None:
                    finite = np.isfinite(depth_map) & (depth_map > 0)
                    values = depth_map[finite]
                    if values.size:
                        lower, upper = np.percentile(values, [2, 98])
                    else:
                        lower, upper = 0.0, 1.0
                    scaled = np.clip((depth_map - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
                    depth_color = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    overlay = cv2.addWeighted(frame, 0.45, depth_color, 0.55, 0.0)
                    cv2.putText(
                        overlay,
                        f"frame {frame_id} depth p2={lower:.2f} p98={upper:.2f}",
                        (18, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    preview.write(np.hstack([frame, overlay]))
                if (offset + 1) % 50 == 0 or offset + 1 == len(frame_ids):
                    print(f"Depth inference: {offset + 1}/{len(frame_ids)} frames")
    finally:
        capture.release()
        if preview is not None:
            preview.release()
        if raw_depth is not None:
            raw_depth.flush()
            raw_depth._mmap.close()

    table = pd.DataFrame(rows).sort_values(["frame_id", "player_id"])
    runtime = {
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_peak_allocated_mb": int(torch.cuda.max_memory_allocated() / 1024**2)
        if torch.cuda.is_available()
        else None,
        "model": str(args.model.resolve()),
        "device": args.device,
        "imgsz": args.imgsz,
    }
    return table, runtime


def _quality_from_observations(group: pd.DataFrame) -> np.ndarray:
    observed = group.depth_median.to_numpy(dtype=float)
    valid_ratio = np.clip(group.valid_ratio.to_numpy(dtype=float), 0.0, 1.0)
    roi_confidence = np.clip(group.roi_confidence.to_numpy(dtype=float), 0.0, 1.0)
    iqr = np.maximum(group.depth_iqr.to_numpy(dtype=float), 0.0)
    dispersion = np.clip(1.0 - iqr / np.maximum(observed, 1.0), 0.0, 1.0)
    quality = roi_confidence * valid_ratio * dispersion
    quality[~np.isfinite(observed)] = 0.0
    return quality


def _inside_court_ratio(points: np.ndarray, width: float, length: float) -> float:
    inside = (np.abs(points[:, 0]) <= width / 2.0) & (np.abs(points[:, 1]) <= length / 2.0)
    return float(np.mean(inside))


def _ground_consistency(
    baseline_vertices: np.ndarray,
    corrected_vertices: np.ndarray,
    foot_stationary_confidence: np.ndarray,
) -> dict[str, Any]:
    baseline_min_z = np.min(baseline_vertices[:, :, 2], axis=1)
    corrected_min_z = np.min(corrected_vertices[:, :, 2], axis=1)
    contact = foot_stationary_confidence >= 0.6
    if not np.any(contact):
        contact = np.ones(len(baseline_min_z), dtype=bool)
    return {
        "contact_frame_count": int(np.count_nonzero(contact)),
        "baseline_contact_median_min_vertex_z_m": float(np.median(baseline_min_z[contact])),
        "corrected_contact_median_min_vertex_z_m": float(np.median(corrected_min_z[contact])),
        "max_abs_min_vertex_z_change_m": float(np.max(np.abs(corrected_min_z - baseline_min_z))),
        "exactly_preserved": bool(np.array_equal(corrected_min_z, baseline_min_z)),
    }


def _write_diagnostic_plot(diagnostics: pd.DataFrame, player_ids: list[int], output_path: Path) -> None:
    figure, axes = plt.subplots(len(player_ids), 2, figsize=(14, max(4, 3.8 * len(player_ids))), squeeze=False)
    for row_index, player_id in enumerate(player_ids):
        group = diagnostics[diagnostics.player_id == player_id].sort_values("frame_id")
        frames = group.frame_id.to_numpy()
        axes[row_index, 0].plot(frames, group.baseline_camera_depth_m, "--", label="GVHMR baseline")
        axes[row_index, 0].plot(frames, group.aligned_target_camera_depth_m, ":", label="Depth target")
        axes[row_index, 0].plot(frames, group.corrected_camera_depth_m, "-", label="Corrected")
        axes[row_index, 0].set_title(f"Player {player_id}: camera distance")
        axes[row_index, 0].set_ylabel("camera Z (m)")
        axes[row_index, 0].legend(fontsize=8)
        axes[row_index, 1].plot(frames, group.jump_score, label="jump")
        axes[row_index, 1].plot(frames, group.travel_score, label="depth travel")
        axes[row_index, 1].plot(frames, group.drift_score, label="drift")
        axes[row_index, 1].set_ylim(-0.02, 1.02)
        axes[row_index, 1].set_title(f"Player {player_id}: continuous motion scores")
        axes[row_index, 1].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("video frame")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    config = _validated_config(args)
    for path in (args.video, args.gvhmr_bundle, args.scene_metadata):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.person_depth_csv is None and (args.pose_csv is None or not args.pose_csv.exists()):
        raise FileNotFoundError("A valid --pose-csv is required for automatic depth inference")
    if args.person_depth_csv is not None and not args.person_depth_csv.exists():
        raise FileNotFoundError(args.person_depth_csv)
    if args.person_depth_csv is None and args.model is None:
        raise ValueError(
            "The legacy YOLO depth model is no longer bundled. Generate person depth with "
            "work/video_depth_anything/run_vda_multicue_pipeline.py, or pass --person-depth-csv."
        )
    if args.person_depth_csv is None and not args.model.exists():
        raise FileNotFoundError(args.model)
    output_resolved = args.output_dir.resolve()
    input_parents = {
        args.video.resolve().parent,
        args.gvhmr_bundle.resolve().parent,
        args.scene_metadata.resolve().parent,
    }
    if args.pose_csv is not None:
        input_parents.add(args.pose_csv.resolve().parent)
    if args.person_depth_csv is not None:
        input_parents.add(args.person_depth_csv.resolve().parent)
    if output_resolved in input_parents:
        raise ValueError("--output-dir must be a separate directory, not an input directory")
    _prepare_output(args.output_dir)

    video_info = _video_info(args.video)
    source_bundle = np.load(args.gvhmr_bundle, allow_pickle=False)
    player_ids = discover_player_ids(source_bundle.files)
    if not player_ids:
        raise ValueError("GVHMR bundle contains no pN_root_court + pN_vertices_court player pairs")
    bundle_frames = len(source_bundle[f"p{player_ids[0]}_root_court"])
    for player_id in player_ids:
        if len(source_bundle[f"p{player_id}_root_court"]) != bundle_frames:
            raise ValueError("Player root frame counts differ in the GVHMR bundle")
    bundle_end_frame = args.bundle_start_frame + bundle_frames - 1
    overlap_end = min(int(video_info["frames"]), bundle_end_frame)
    if args.pose_csv is not None and args.pose_csv.exists():
        pose = pd.read_csv(args.pose_csv)
        required_pose_columns = {"frame_id", "player_id", "keypoint_id", "x", "y", "confidence"}
        missing_pose_columns = required_pose_columns - set(pose.columns)
        if missing_pose_columns:
            raise ValueError(f"Pose CSV is missing columns: {sorted(missing_pose_columns)}")
        overlap_end = min(overlap_end, int(pose.frame_id.max()))
    else:
        pose = pd.DataFrame()
    end_frame = args.end_frame if args.end_frame is not None else overlap_end
    if args.start_frame < 1 or end_frame < args.start_frame or end_frame > overlap_end:
        raise ValueError(
            f"Invalid frame range {args.start_frame}..{end_frame}; overlapping input range ends at {overlap_end}"
        )
    frame_ids = np.arange(args.start_frame, end_frame + 1, dtype=int)
    bundle_indices = frame_ids - args.bundle_start_frame
    if np.any(bundle_indices < 0) or np.any(bundle_indices >= bundle_frames):
        raise ValueError("Selected video frames do not map into the GVHMR bundle")

    if args.person_depth_csv is not None:
        person_depth = pd.read_csv(args.person_depth_csv)
        required_depth_columns = {
            "frame_id",
            "player_id",
            "depth_median",
            "depth_iqr",
            "valid_ratio",
            "roi_confidence",
        }
        missing_depth_columns = required_depth_columns - set(person_depth.columns)
        if missing_depth_columns:
            raise ValueError(f"Person depth CSV is missing columns: {sorted(missing_depth_columns)}")
        runtime = {
            "depth_inference_skipped": True,
            "reused_person_depth_csv": str(args.person_depth_csv.resolve()),
        }
    else:
        pose = pose[pose.frame_id.isin(frame_ids) & pose.player_id.isin(player_ids)].copy()
        person_depth, runtime = infer_person_depth(args, pose, player_ids, frame_ids, video_info)
    person_depth = person_depth[
        person_depth.frame_id.isin(frame_ids) & person_depth.player_id.isin(player_ids)
    ].copy()
    person_depth.to_csv(args.output_dir / "person_depth.csv", index=False, float_format="%.8f")

    metadata = json.loads(args.scene_metadata.read_text(encoding="utf-8"))
    calibration = metadata.get("calibration", {})
    rotation = np.asarray(calibration.get("court_to_camera_R"), dtype=np.float64)
    translation = np.asarray(calibration.get("court_to_camera_t"), dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("scene_metadata.json lacks a valid court_to_camera_R/t calibration")
    court_width = float(np.asarray(source_bundle["court_width_m"]).reshape(()))
    court_length = float(np.asarray(source_bundle["court_length_m"]).reshape(()))
    bundle_fps = float(np.asarray(source_bundle["fps"]).reshape(()))
    fps = float(video_info["fps"])
    if abs(bundle_fps - fps) > 0.1:
        raise ValueError(f"Video FPS {fps} and bundle FPS {bundle_fps} differ")

    output_arrays = {key: np.asarray(source_bundle[key]) for key in source_bundle.files}
    diagnostics_rows: list[dict[str, Any]] = []
    player_metrics: dict[str, Any] = {}
    for player_id in player_ids:
        group = (
            person_depth[person_depth.player_id == player_id]
            .set_index("frame_id")
            .reindex(frame_ids)
            .reset_index()
        )
        group["player_id"] = player_id
        for column, default in {
            "depth_median": np.nan,
            "depth_iqr": np.nan,
            "valid_ratio": 0.0,
            "roi_confidence": 0.0,
        }.items():
            if column not in group:
                group[column] = default
            group[column] = group[column].fillna(default) if np.isfinite(default) else group[column]
        quality = _quality_from_observations(group)
        observed = group.depth_median.to_numpy(dtype=float)
        baseline_root_full = np.asarray(source_bundle[f"p{player_id}_root_court"], dtype=np.float32)
        baseline_vertices_full = np.asarray(source_bundle[f"p{player_id}_vertices_court"], dtype=np.float32)
        baseline_root = np.asarray(baseline_root_full[bundle_indices], dtype=np.float64)
        baseline_vertices = np.asarray(baseline_vertices_full[bundle_indices], dtype=np.float64)
        baseline_camera = baseline_root @ rotation.T + translation
        foot_key = f"p{player_id}_foot_static_confidence"
        if foot_key in source_bundle.files:
            foot_stationary = np.asarray(source_bundle[foot_key][bundle_indices], dtype=np.float64)
            if foot_stationary.ndim > 1:
                foot_stationary = np.max(foot_stationary, axis=tuple(range(1, foot_stationary.ndim)))
            foot_stationary = np.clip(
                np.nan_to_num(foot_stationary, nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
        else:
            foot_stationary = np.zeros(len(frame_ids), dtype=np.float64)

        valid_depth_count = int(
            np.count_nonzero(np.isfinite(observed) & (observed > 0) & (quality > 0))
        )
        result = None
        skip_reason = None
        if len(frame_ids) < 4:
            skip_reason = "fewer_than_four_frames"
        elif valid_depth_count < config.min_valid_observations:
            skip_reason = (
                f"insufficient_valid_depth_observations:{valid_depth_count}"
                f"<{config.min_valid_observations}"
            )
        if skip_reason is None:
            result = constrain_camera_depth(
                baseline_camera[:, 2],
                observed,
                quality,
                baseline_root[:, 2],
                fps,
                config,
                foot_stationary_confidence=foot_stationary,
            )
            correction_xy = camera_depth_delta_to_court_xy(result.applied_camera_depth_delta_m, rotation)
            correction_xy = clip_court_xy_correction(
                baseline_root[:, :2],
                correction_xy,
                court_width,
                court_length,
                config.court_margin_m,
            )
        else:
            correction_xy = np.zeros((len(frame_ids), 2), dtype=np.float64)
        corrected_root_full = baseline_root_full.copy()
        corrected_vertices_full = baseline_vertices_full.copy()
        corrected_root_full[bundle_indices, :2] += correction_xy.astype(np.float32)
        corrected_vertices_full[bundle_indices, :, :2] += correction_xy[:, None, :].astype(np.float32)
        output_arrays[f"p{player_id}_root_court"] = corrected_root_full
        output_arrays[f"p{player_id}_vertices_court"] = corrected_vertices_full

        corrected_root = np.asarray(corrected_root_full[bundle_indices], dtype=np.float64)
        corrected_vertices = np.asarray(corrected_vertices_full[bundle_indices], dtype=np.float64)
        corrected_camera = corrected_root @ rotation.T + translation
        if result is not None:
            metrics = trajectory_metrics(
                baseline_root,
                corrected_root,
                baseline_camera[:, 2],
                corrected_camera[:, 2],
                result.aligned_target_camera_depth_m,
                fps,
            )
            unexplained_threshold = config.max_unexplained_depth_speed_mps / fps
            target_steps = np.diff(result.aligned_target_camera_depth_m)
            baseline_steps = np.diff(baseline_camera[:, 2])
            corrected_steps = np.diff(corrected_camera[:, 2])
            baseline_spikes = np.abs(baseline_steps - target_steps) > unexplained_threshold
            corrected_spikes = np.abs(corrected_steps - target_steps) > unexplained_threshold
            metrics.update(
                {
                    "status": "applied",
                    "affine_scale": result.affine_scale,
                    "affine_offset_m": result.affine_offset_m,
                    "affine_residual_scale_m": result.affine_residual_scale_m,
                    "valid_depth_observations": valid_depth_count,
                    "outlier_observations": int(np.count_nonzero(result.observation_outlier)),
                    "short_drift_spike_count_before": int(np.count_nonzero(baseline_spikes)),
                    "short_drift_spike_count_after": int(np.count_nonzero(corrected_spikes)),
                    "max_abs_applied_camera_depth_delta_m": float(
                        np.max(np.abs(corrected_camera[:, 2] - baseline_camera[:, 2]))
                    ),
                    "max_jump_score": float(np.max(result.jump_score)),
                    "max_travel_score": float(np.max(result.travel_score)),
                    "max_drift_score": float(np.max(result.drift_score)),
                }
            )
        else:
            metrics = {
                "status": "skipped_safe_identity",
                "reason": skip_reason,
                "valid_depth_observations": valid_depth_count,
                "short_drift_spike_count_before": None,
                "short_drift_spike_count_after": None,
                "max_abs_applied_camera_depth_delta_m": 0.0,
                "vertical_motion_preservation": {
                    "max_abs_root_z_change_m": 0.0,
                    "exactly_preserved": True,
                },
            }
        metrics.update(
            {
                "ground_consistency": _ground_consistency(
                    baseline_vertices,
                    corrected_vertices,
                    foot_stationary,
                ),
                "court_inside_ratio_before": _inside_court_ratio(baseline_root, court_width, court_length),
                "court_inside_ratio_after": _inside_court_ratio(corrected_root, court_width, court_length),
            }
        )
        player_metrics[str(player_id)] = metrics

        full_nan = np.full(bundle_frames, np.nan, dtype=np.float32)
        full_bool = np.zeros(bundle_frames, dtype=np.uint8)
        if result is None:
            cleaned_depth = np.full(len(frame_ids), np.nan, dtype=np.float64)
            target_depth = np.full(len(frame_ids), np.nan, dtype=np.float64)
            result_quality = quality
            outlier = np.zeros(len(frame_ids), dtype=bool)
            jump_score = np.zeros(len(frame_ids), dtype=np.float64)
            travel_score = np.zeros(len(frame_ids), dtype=np.float64)
            drift_score = np.zeros(len(frame_ids), dtype=np.float64)
        else:
            cleaned_depth = result.cleaned_observation_m
            target_depth = result.aligned_target_camera_depth_m
            result_quality = result.observation_quality
            outlier = result.observation_outlier
            jump_score = result.jump_score
            travel_score = result.travel_score
            drift_score = result.drift_score
        diagnostic_arrays = {
            "depth_observation_m": observed,
            "depth_cleaned_m": cleaned_depth,
            "depth_target_camera_m": target_depth,
            "camera_depth_baseline_m": baseline_camera[:, 2],
            "camera_depth_corrected_m": corrected_camera[:, 2],
            "depth_quality": result_quality,
            "foot_stationary_confidence": foot_stationary,
            "jump_score": jump_score,
            "travel_score": travel_score,
            "drift_score": drift_score,
            "applied_court_dx_m": correction_xy[:, 0],
            "applied_court_dy_m": correction_xy[:, 1],
        }
        for name, values in diagnostic_arrays.items():
            array = full_nan.copy()
            array[bundle_indices] = np.asarray(values, dtype=np.float32)
            output_arrays[f"p{player_id}_{name}"] = array
        outlier_array = full_bool.copy()
        outlier_array[bundle_indices] = outlier.astype(np.uint8)
        output_arrays[f"p{player_id}_depth_outlier"] = outlier_array

        for local_index, frame_id in enumerate(frame_ids):
            diagnostics_rows.append(
                {
                    "frame_id": int(frame_id),
                    "bundle_index": int(bundle_indices[local_index]),
                    "player_id": player_id,
                    "observed_depth_m": observed[local_index],
                    "cleaned_depth_m": cleaned_depth[local_index],
                    "aligned_target_camera_depth_m": target_depth[local_index],
                    "baseline_camera_depth_m": baseline_camera[local_index, 2],
                    "corrected_camera_depth_m": corrected_camera[local_index, 2],
                    "depth_quality": result_quality[local_index],
                    "depth_outlier": bool(outlier[local_index]),
                    "foot_stationary_confidence": foot_stationary[local_index],
                    "jump_score": jump_score[local_index],
                    "travel_score": travel_score[local_index],
                    "drift_score": drift_score[local_index],
                    "applied_court_dx_m": correction_xy[local_index, 0],
                    "applied_court_dy_m": correction_xy[local_index, 1],
                    "root_z_original_m": baseline_root[local_index, 2],
                    "root_z_corrected_m": corrected_root[local_index, 2],
                }
            )

    output_arrays["depth_constraint_frame_start_1based"] = np.int32(args.start_frame)
    output_arrays["depth_constraint_frame_end_1based"] = np.int32(end_frame)
    output_arrays["depth_constraint_bundle_start_frame_1based"] = np.int32(args.bundle_start_frame)
    output_arrays["depth_constraint_version"] = np.asarray("1.0")
    output_bundle = args.output_dir / "depth_constrained_smplx_court_animation.npz"
    np.savez_compressed(output_bundle, **output_arrays)
    source_bundle.close()

    diagnostics = pd.DataFrame(diagnostics_rows).sort_values(["frame_id", "player_id"])
    diagnostics.to_csv(args.output_dir / "depth_constraint_diagnostics.csv", index=False, float_format="%.8f")
    # Compatibility name retained from the experimental Phase 3 output.
    diagnostics.to_csv(args.output_dir / "depth_alignment.csv", index=False, float_format="%.8f")
    _write_diagnostic_plot(
        diagnostics,
        player_ids,
        args.output_dir / "diagnostics" / "depth_constraint_overview.png",
    )
    metrics_payload = {
        "video": str(args.video.resolve()),
        "pose_csv": str(args.pose_csv.resolve()) if args.pose_csv is not None else None,
        "source_gvhmr_bundle": str(args.gvhmr_bundle.resolve()),
        "source_scene_metadata": str(args.scene_metadata.resolve()),
        "output_bundle": str(output_bundle.resolve()),
        "ab_3d_results": {
            "A_gvhmr_only": str(args.gvhmr_bundle.resolve()),
            "B_depth_constrained": str(output_bundle.resolve()),
        },
        "frame_start": args.start_frame,
        "frame_end": end_frame,
        "frame_count": int(len(frame_ids)),
        "bundle_start_frame": args.bundle_start_frame,
        "fps": fps,
        "players": player_ids,
        "configuration": asdict(config),
        "depth_runtime": runtime,
        "player_metrics": player_metrics,
        "safety": {
            "vertical_root_motion_modified": False,
            "articulated_pose_modified": False,
            "correction_axis": "court XY projection of camera depth only",
            "source_bundle_overwritten": False,
        },
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ab_manifest.json").write_text(
        json.dumps(metrics_payload["ab_3d_results"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    constrained_metadata = dict(metadata)
    constrained_metadata["depth_constraint"] = metrics_payload
    constrained_metadata["bundle"] = output_bundle.name
    (args.output_dir / "scene_metadata.json").write_text(
        json.dumps(constrained_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
