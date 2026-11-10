"""Repair GVHMR court trajectories with stable 2-D, contact and depth cues.

This production entry point performs a full-sequence sparse optimization. It
never edits the source bundle in place and never changes root/mesh Z or local
SMPL-X articulation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
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
    clean_depth_observations,
    discover_player_ids,
    robust_affine_alignment,
)
from src.pose_biomechanics.trajectory_fusion import (  # noqa: E402
    TrajectoryFusionConfig,
    fuse_court_trajectory,
    project_court_points,
    trajectory_motion_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-sequence multi-cue root trajectory repair for a GVHMR court bundle."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--pose-csv", type=Path, required=True)
    parser.add_argument("--gvhmr-bundle", type=Path, required=True)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--person-depth-csv", type=Path)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--bundle-start-frame", type=int, default=1)
    parser.add_argument("--plausible-speed-mps", type=float)
    parser.add_argument("--absolute-speed-limit-mps", type=float)
    parser.add_argument("--max-correction-m", type=float)
    parser.add_argument("--depth-weight", type=float)
    return parser.parse_args()


def _config(args: argparse.Namespace) -> TrajectoryFusionConfig:
    config = TrajectoryFusionConfig()
    overrides = {
        "plausible_speed_mps": args.plausible_speed_mps,
        "absolute_speed_limit_mps": args.absolute_speed_limit_mps,
        "max_correction_m": args.max_correction_m,
        "depth_weight": args.depth_weight,
    }
    values = {key: value for key, value in overrides.items() if value is not None}
    if values:
        config = replace(config, **values)
    if config.plausible_speed_mps <= 0 or config.absolute_speed_limit_mps <= 0:
        raise ValueError("Speed limits must be positive")
    if config.max_correction_m <= 0 or config.depth_weight < 0:
        raise ValueError("Correction limit must be positive and depth weight non-negative")
    return config


def _prepare_output(output_dir: Path, input_paths: list[Path]) -> None:
    resolved = output_dir.resolve()
    if resolved in {path.resolve().parent for path in input_paths}:
        raise ValueError("--output-dir must be separate from every input directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics").mkdir(parents=True, exist_ok=True)


def _video_info(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    result = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    if result["frames"] <= 0 or result["fps"] <= 0:
        raise ValueError(f"Invalid video metadata: {result}")
    return result


def _pelvis_observations(
    pose: pd.DataFrame,
    frame_ids: np.ndarray,
    player_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    hips = pose[
        (pose.player_id == player_id)
        & pose.frame_id.isin(frame_ids)
        & pose.keypoint_id.isin([11, 12])
    ].copy()
    hips["weighted_x"] = hips.x * hips.confidence
    hips["weighted_y"] = hips.y * hips.confidence
    grouped = hips.groupby("frame_id").agg(
        weight=("confidence", "sum"),
        weighted_x=("weighted_x", "sum"),
        weighted_y=("weighted_y", "sum"),
        confidence=("confidence", "mean"),
        count=("keypoint_id", "count"),
    )
    grouped = grouped.reindex(frame_ids)
    weight = grouped.weight.to_numpy(dtype=float)
    xy = np.column_stack(
        [
            grouped.weighted_x.to_numpy(dtype=float) / np.maximum(weight, 1e-9),
            grouped.weighted_y.to_numpy(dtype=float) / np.maximum(weight, 1e-9),
        ]
    )
    confidence = grouped.confidence.to_numpy(dtype=float)
    confidence[grouped["count"].to_numpy(dtype=float) < 2] *= 0.5
    confidence[~np.isfinite(xy).all(axis=1)] = 0.0
    return xy, np.nan_to_num(confidence, nan=0.0)


def _aligned_depth_target(
    person_depth: pd.DataFrame | None,
    frame_ids: np.ndarray,
    player_id: int,
    baseline_camera_depth_m: np.ndarray,
    fps: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if person_depth is None:
        return None, None, {"status": "not_supplied"}
    group = person_depth[person_depth.player_id == player_id].set_index("frame_id").reindex(frame_ids)
    observed = group.depth_median.to_numpy(dtype=float)
    valid_ratio = np.clip(group.valid_ratio.to_numpy(dtype=float), 0.0, 1.0)
    roi_confidence = np.clip(group.roi_confidence.to_numpy(dtype=float), 0.0, 1.0)
    iqr = np.maximum(group.depth_iqr.to_numpy(dtype=float), 0.0)
    quality = roi_confidence * valid_ratio * np.clip(
        1.0 - iqr / np.maximum(observed, 1.0), 0.0, 1.0
    )
    quality[~np.isfinite(observed)] = 0.0
    depth_config = DepthConstraintConfig()
    valid_count = int(np.count_nonzero(quality > 0))
    if valid_count < depth_config.min_valid_observations:
        return None, None, {
            "status": "insufficient_observations",
            "valid_observations": valid_count,
        }
    cleaned, adjusted_quality, outlier = clean_depth_observations(
        observed, quality, fps, depth_config
    )
    scale, offset, residual_scale = robust_affine_alignment(
        cleaned, baseline_camera_depth_m, adjusted_quality
    )
    return scale * cleaned + offset, adjusted_quality, {
        "status": "aligned",
        "valid_observations": valid_count,
        "outliers": int(np.count_nonzero(outlier)),
        "scale": scale,
        "offset_m": offset,
        "residual_scale_m": residual_scale,
    }


def _projection_metrics(
    root_xyz: np.ndarray,
    pelvis_uv: np.ndarray,
    pelvis_confidence: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, float | int]:
    projected = project_court_points(root_xyz, camera_matrix, rotation, translation)
    valid = (
        np.isfinite(projected).all(axis=1)
        & np.isfinite(pelvis_uv).all(axis=1)
        & (pelvis_confidence > 0)
    )
    if not np.any(valid):
        return {"valid_frames": 0, "bias_corrected_rmse_px": float("nan")}
    bias = np.median(pelvis_uv[valid] - projected[valid], axis=0)
    residual = np.linalg.norm(projected[valid] + bias - pelvis_uv[valid], axis=1)
    return {
        "valid_frames": int(np.count_nonzero(valid)),
        "projection_bias_x_px": float(bias[0]),
        "projection_bias_y_px": float(bias[1]),
        "bias_corrected_rmse_px": float(np.sqrt(np.mean(np.square(residual)))),
        "bias_corrected_p95_px": float(np.percentile(residual, 95)),
    }


def _plot_diagnostics(table: pd.DataFrame, player_ids: list[int], path: Path) -> None:
    figure, axes = plt.subplots(len(player_ids), 2, figsize=(14, 4 * len(player_ids)), squeeze=False)
    for row, player_id in enumerate(player_ids):
        group = table[table.player_id == player_id].sort_values("frame_id")
        axes[row, 0].plot(group.raw_root_x_m, group.raw_root_y_m, "--", label="GVHMR A")
        axes[row, 0].plot(group.corrected_root_x_m, group.corrected_root_y_m, label="fused B")
        axes[row, 0].scatter(
            group.ground_x_m,
            group.ground_y_m,
            c=group.ground_quality,
            s=8,
            alpha=0.45,
            label="contact measurement",
        )
        axes[row, 0].set_title(f"Player {player_id}: court trajectory")
        axes[row, 0].set_xlabel("court X (m)")
        axes[row, 0].set_ylabel("court Y (m)")
        axes[row, 0].set_aspect("equal", adjustable="box")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].plot(group.frame_id, group.raw_speed_mps, "--", label="GVHMR speed")
        axes[row, 1].plot(group.frame_id, group.corrected_speed_mps, label="fused speed")
        axes[row, 1].plot(group.frame_id, group.motion_score * 5.0, ":", label="2D motion score × 5")
        axes[row, 1].set_title(f"Player {player_id}: root speed")
        axes[row, 1].set_xlabel("video frame")
        axes[row, 1].set_ylabel("m/s")
        axes[row, 1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    config = _config(args)
    input_paths = [args.video, args.pose_csv, args.gvhmr_bundle, args.scene_metadata]
    if args.person_depth_csv is not None:
        input_paths.append(args.person_depth_csv)
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    _prepare_output(args.output_dir, input_paths)

    video_info = _video_info(args.video)
    pose = pd.read_csv(args.pose_csv)
    required_pose = {"frame_id", "player_id", "keypoint_id", "x", "y", "confidence"}
    missing_pose = required_pose - set(pose.columns)
    if missing_pose:
        raise ValueError(f"Pose CSV missing columns: {sorted(missing_pose)}")
    person_depth = pd.read_csv(args.person_depth_csv) if args.person_depth_csv is not None else None
    if person_depth is not None:
        required_depth = {"frame_id", "player_id", "depth_median", "depth_iqr", "valid_ratio", "roi_confidence"}
        missing_depth = required_depth - set(person_depth.columns)
        if missing_depth:
            raise ValueError(f"Depth CSV missing columns: {sorted(missing_depth)}")

    source = np.load(args.gvhmr_bundle, allow_pickle=False)
    player_ids = discover_player_ids(source.files)
    if not player_ids:
        raise ValueError("GVHMR bundle has no pN_root_court/pN_vertices_court pairs")
    frame_count = len(source[f"p{player_ids[0]}_root_court"])
    bundle_end = args.bundle_start_frame + frame_count - 1
    overlap_end = min(int(video_info["frames"]), int(pose.frame_id.max()), bundle_end)
    end_frame = overlap_end if args.end_frame is None else args.end_frame
    if args.start_frame < 1 or end_frame < args.start_frame or end_frame > overlap_end:
        raise ValueError(f"Invalid frame range {args.start_frame}..{end_frame}; overlap ends at {overlap_end}")
    frame_ids = np.arange(args.start_frame, end_frame + 1, dtype=int)
    bundle_indices = frame_ids - args.bundle_start_frame

    metadata = json.loads(args.scene_metadata.read_text(encoding="utf-8"))
    calibration = metadata.get("calibration", {})
    camera_matrix = np.asarray(calibration.get("K"), dtype=np.float64)
    rotation = np.asarray(calibration.get("court_to_camera_R"), dtype=np.float64)
    translation = np.asarray(calibration.get("court_to_camera_t"), dtype=np.float64)
    if camera_matrix.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Scene metadata lacks valid K/R/t calibration")
    fps = float(np.asarray(source["fps"]).reshape(()))
    if abs(fps - float(video_info["fps"])) > 0.1:
        raise ValueError("Video and GVHMR bundle FPS differ")
    court_width = float(np.asarray(source["court_width_m"]).reshape(()))
    court_length = float(np.asarray(source["court_length_m"]).reshape(()))

    output_arrays = {key: np.asarray(source[key]) for key in source.files}
    metrics: dict[str, Any] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    for player_id in player_ids:
        root_full = np.asarray(source[f"p{player_id}_root_court"], dtype=np.float32)
        vertices_full = np.asarray(source[f"p{player_id}_vertices_court"], dtype=np.float32)
        root = np.asarray(root_full[bundle_indices], dtype=np.float64)
        pelvis_uv, pelvis_confidence = _pelvis_observations(pose, frame_ids, player_id)
        ground_key = f"p{player_id}_ground_measurements_court"
        contact_key = f"p{player_id}_foot_static_confidence"
        ground = (
            np.asarray(source[ground_key][bundle_indices], dtype=np.float64)
            if ground_key in source.files
            else np.full((len(frame_ids), 2), np.nan)
        )
        contact = (
            np.asarray(source[contact_key][bundle_indices], dtype=np.float64)
            if contact_key in source.files
            else np.zeros(len(frame_ids))
        )
        if contact.ndim > 1:
            contact = np.max(contact, axis=tuple(range(1, contact.ndim)))
        baseline_camera = root @ rotation.T + translation
        depth_target, depth_quality, depth_info = _aligned_depth_target(
            person_depth, frame_ids, player_id, baseline_camera[:, 2], fps
        )
        result = fuse_court_trajectory(
            root[:, :2],
            root[:, 2],
            pelvis_uv,
            pelvis_confidence,
            ground,
            contact,
            camera_matrix,
            rotation,
            translation,
            fps,
            court_width,
            court_length,
            depth_target_camera_m=depth_target,
            depth_quality=depth_quality,
            config=config,
        )

        corrected_root_full = root_full.copy()
        corrected_vertices_full = vertices_full.copy()
        correction = result.applied_xy_m.astype(np.float32)
        corrected_root_full[bundle_indices, :2] += correction
        corrected_vertices_full[bundle_indices, :, :2] += correction[:, None, :]
        output_arrays[f"p{player_id}_root_court"] = corrected_root_full
        output_arrays[f"p{player_id}_vertices_court"] = corrected_vertices_full
        corrected_root = np.asarray(corrected_root_full[bundle_indices], dtype=np.float64)

        raw_motion = trajectory_motion_metrics(root, fps)
        corrected_motion = trajectory_motion_metrics(corrected_root, fps)
        raw_speed = np.r_[0.0, np.linalg.norm(np.diff(root, axis=0), axis=1) * fps]
        corrected_speed = np.r_[0.0, np.linalg.norm(np.diff(corrected_root, axis=0), axis=1) * fps]
        inside_before = (np.abs(root[:, 0]) <= court_width / 2.0) & (np.abs(root[:, 1]) <= court_length / 2.0)
        inside_after = (np.abs(corrected_root[:, 0]) <= court_width / 2.0) & (
            np.abs(corrected_root[:, 1]) <= court_length / 2.0
        )
        metrics[str(player_id)] = {
            "baseline_motion": raw_motion,
            "corrected_motion": corrected_motion,
            "speed_spikes_above_plausible_before": int(np.count_nonzero(raw_speed > config.plausible_speed_mps)),
            "speed_spikes_above_plausible_after": int(np.count_nonzero(corrected_speed > config.plausible_speed_mps)),
            "pelvis_reprojection_before": _projection_metrics(
                root, pelvis_uv, pelvis_confidence, camera_matrix, rotation, translation
            ),
            "pelvis_reprojection_after": _projection_metrics(
                corrected_root, pelvis_uv, pelvis_confidence, camera_matrix, rotation, translation
            ),
            "court_inside_ratio_before": float(np.mean(inside_before)),
            "court_inside_ratio_after": float(np.mean(inside_after)),
            "max_xy_correction_m": float(np.max(np.linalg.norm(result.applied_xy_m, axis=1))),
            "root_z_exactly_preserved": bool(np.array_equal(root[:, 2], corrected_root[:, 2])),
            "depth": depth_info,
            "pelvis_alignment_offset_xy_m": result.pelvis_alignment_offset_xy_m.tolist(),
        }

        full_nan = np.full(frame_count, np.nan, dtype=np.float32)
        for name, values in {
            "fusion_applied_dx_m": result.applied_xy_m[:, 0],
            "fusion_applied_dy_m": result.applied_xy_m[:, 1],
            "fusion_pelvis_target_x_m": result.aligned_pelvis_target_xy_m[:, 0],
            "fusion_pelvis_target_y_m": result.aligned_pelvis_target_xy_m[:, 1],
            "fusion_raw_velocity_quality": result.raw_velocity_quality,
            "fusion_pelvis_quality": result.pelvis_quality,
            "fusion_ground_quality": result.ground_quality,
            "fusion_motion_score": result.motion_score,
        }.items():
            array = full_nan.copy()
            array[bundle_indices] = np.asarray(values, dtype=np.float32)
            output_arrays[f"p{player_id}_{name}"] = array

        for local, frame_id in enumerate(frame_ids):
            diagnostic_rows.append(
                {
                    "frame_id": int(frame_id),
                    "player_id": player_id,
                    "raw_root_x_m": root[local, 0],
                    "raw_root_y_m": root[local, 1],
                    "root_z_m": root[local, 2],
                    "corrected_root_x_m": corrected_root[local, 0],
                    "corrected_root_y_m": corrected_root[local, 1],
                    "pelvis_u_px": pelvis_uv[local, 0],
                    "pelvis_v_px": pelvis_uv[local, 1],
                    "pelvis_confidence": pelvis_confidence[local],
                    "pelvis_target_x_m": result.aligned_pelvis_target_xy_m[local, 0],
                    "pelvis_target_y_m": result.aligned_pelvis_target_xy_m[local, 1],
                    "ground_x_m": ground[local, 0],
                    "ground_y_m": ground[local, 1],
                    "foot_stationary_confidence": contact[local],
                    "raw_velocity_quality": result.raw_velocity_quality[local],
                    "pelvis_quality": result.pelvis_quality[local],
                    "ground_quality": result.ground_quality[local],
                    "motion_score": result.motion_score[local],
                    "raw_speed_mps": raw_speed[local],
                    "corrected_speed_mps": corrected_speed[local],
                    "applied_dx_m": result.applied_xy_m[local, 0],
                    "applied_dy_m": result.applied_xy_m[local, 1],
                }
            )

    output_arrays["trajectory_fusion_frame_start_1based"] = np.int32(args.start_frame)
    output_arrays["trajectory_fusion_frame_end_1based"] = np.int32(end_frame)
    output_arrays["trajectory_fusion_version"] = np.asarray("2.0")
    output_bundle = args.output_dir / "multicue_trajectory_fused_smplx_court_animation.npz"
    np.savez_compressed(output_bundle, **output_arrays)
    source.close()

    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(["frame_id", "player_id"])
    diagnostics.to_csv(args.output_dir / "trajectory_fusion_diagnostics.csv", index=False, float_format="%.8f")
    _plot_diagnostics(
        diagnostics,
        player_ids,
        args.output_dir / "diagnostics" / "trajectory_fusion_overview.png",
    )
    payload = {
        "video": str(args.video.resolve()),
        "pose_csv": str(args.pose_csv.resolve()),
        "source_bundle": str(args.gvhmr_bundle.resolve()),
        "output_bundle": str(output_bundle.resolve()),
        "frame_start": args.start_frame,
        "frame_end": end_frame,
        "frame_count": int(len(frame_ids)),
        "players": player_ids,
        "fps": fps,
        "configuration": asdict(config),
        "player_metrics": metrics,
        "safety": {
            "source_overwritten": False,
            "root_z_modified": False,
            "mesh_z_modified": False,
            "local_smplx_pose_modified": False,
            "output_directory_must_be_new": True,
        },
        "ab_3d_results": {
            "A_gvhmr_only": str(args.gvhmr_bundle.resolve()),
            "B_multicue_fused": str(output_bundle.resolve()),
        },
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ab_manifest.json").write_text(
        json.dumps(payload["ab_3d_results"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_metadata = dict(metadata)
    output_metadata["trajectory_fusion"] = payload
    output_metadata["bundle"] = output_bundle.name
    (args.output_dir / "scene_metadata.json").write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
