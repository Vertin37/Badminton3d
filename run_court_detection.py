"""Run automatic badminton-court line detection on a video or image.

Examples (from the repository root):

    python run_court_detection.py --video examples/data/test.mp4 --keyframes 5
    python run_court_detection.py --video examples/data/test.mp4 --frame 150

The default mode is image-space line inspection.  ``--pole-mapping`` detects
the target poles and net on each video frame, then uses court-line-supported
ground-plane Homography to project the standard court; a net-only PnP solve is
kept only as a guarded fallback.  It still does not build an Open3D/VGGT scene.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.court_detection import (  # noqa: E402
    CourtDetectorConfig,
    CourtLineDetector,
    annotate_pole_guided_court,
    recompute_pnp_ground_mapping,
    select_pole_guided_court,
)
from render_stable_pose_video import _draw_player, _load_points  # noqa: E402


def _frame_indices(cap: cv2.VideoCapture, frame: int | None, keyframes: int) -> list[int]:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return [0]
    if frame is not None:
        return [max(0, min(total - 1, int(frame)))]
    count = max(1, int(keyframes))
    if count == 1:
        return [0]
    indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
    return list(dict.fromkeys(int(index) for index in indices))


def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> tuple[bool, object]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    return cap.read()


def _load_player_targets(pose_csv: Path) -> tuple[dict[int, np.ndarray], np.ndarray | None, float | None]:
    """Load ankle-based foot centers and a robust player-connection direction."""

    if not pose_csv.exists():
        return {}, None, None
    points: dict[tuple[int, int, int], tuple[float, float]] = {}
    with pose_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame_id = int(row["frame_id"])
                player_id = int(row["player_id"])
                keypoint_id = int(row["keypoint_id"])
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if player_id not in (0, 1) or keypoint_id not in (15, 16):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            points[(frame_id, player_id, keypoint_id)] = (x, y)

    frame_pairs: dict[int, np.ndarray] = {}
    for frame_id in sorted({key[0] for key in points}):
        player_feet: list[np.ndarray] = []
        for player_id in (0, 1):
            feet = [
                np.asarray(points[(frame_id, player_id, keypoint_id)], dtype=float)
                for keypoint_id in (15, 16)
                if (frame_id, player_id, keypoint_id) in points
            ]
            if not feet:
                break
            player_feet.append(np.mean(feet, axis=0))
        if len(player_feet) == 2:
            frame_pairs[frame_id] = np.stack(player_feet, axis=0)

    if not frame_pairs:
        return {}, None, None
    all_points = np.concatenate(list(frame_pairs.values()), axis=0)
    doubled_angles: list[complex] = []
    for pair in frame_pairs.values():
        delta = pair[1] - pair[0]
        if np.linalg.norm(delta) < 1e-6:
            continue
        angle = math.atan2(float(delta[1]), float(delta[0]))
        doubled_angles.append(complex(math.cos(2.0 * angle), math.sin(2.0 * angle)))
    if not doubled_angles:
        axis_angle = None
    else:
        mean_angle = sum(doubled_angles) / len(doubled_angles)
        axis_angle = math.degrees(0.5 * math.atan2(mean_angle.imag, mean_angle.real)) % 180.0
    return frame_pairs, all_points, axis_angle


def _nearest_player_pair(frame_pairs: dict[int, np.ndarray], frame_index: int) -> tuple[int, np.ndarray] | tuple[None, None]:
    if not frame_pairs:
        return None, None
    target_frame = min(frame_pairs, key=lambda value: abs(value - frame_index))
    return target_frame, frame_pairs[target_frame]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect likely badminton-court lines from video frames.")
    parser.add_argument("--video", type=Path, required=True, help="input video path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pose_test/court_detection"),
        help="directory for overlays, edge previews, and JSON line parameters",
    )
    parser.add_argument("--frame", type=int, help="detect one explicit frame index")
    parser.add_argument("--keyframes", type=int, default=5, help="number of evenly spaced frames (default: 5)")
    parser.add_argument(
        "--pose-csv",
        type=Path,
        default=Path("outputs/pose_test/pose_data_stable.csv"),
        help="fixed-ID filtered pose CSV used to lock the two target players",
    )
    parser.add_argument(
        "--no-target-filter",
        action="store_true",
        help="show all Hough candidates instead of selecting the four-sided court around the players",
    )
    parser.add_argument("--no-edge-preview", action="store_true", help="do not save Canny/Hough preview images")
    parser.add_argument("--no-labels", action="store_true", help="draw lines without text labels")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="show each annotated frame for visual confirmation; press Q to stop (frames are saved automatically)",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="render a detection video; --pole-mapping recomputes court geometry per frame",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        help="output MP4 path; implies --all-frames",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        help="frame used to establish the static net/post/court geometry (default: video midpoint)",
    )
    parser.add_argument(
        "--pole-mapping",
        action="store_true",
        help="select adjacent pole bases by intersection with the player-foot segment and map the standard court with PnP",
    )
    return parser.parse_args()


def _draw_pose_tracks(frame: np.ndarray, pose_points: dict, pose_frame: int | None) -> None:
    """Overlay the already-filtered two-player skeleton for one video frame."""

    if pose_frame is None:
        return
    colors = {0: (0, 165, 255), 1: (255, 220, 0)}
    for player_id, points in pose_points.get(pose_frame, {}).items():
        _draw_player(
            frame,
            player_id,
            points,
            colors.get(player_id, (0, 255, 0)),
            show_raw=False,
        )


def _pole_model_motion(first: Any, second: Any) -> float:
    """Return the largest post-foot displacement between two frame models."""

    first_feet = np.asarray(first.pole_feet, dtype=float)
    second_feet = np.asarray(second.pole_feet, dtype=float)
    if first_feet.shape != second_feet.shape or first_feet.shape != (2, 2):
        return float("inf")
    return float(np.max(np.linalg.norm(second_feet - first_feet, axis=1)))


def _polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return 0.0
    return abs(float(cv2.contourArea(points.astype(np.float32))))


def _normalized_model_shape(model: Any) -> tuple[np.ndarray, float, float] | None:
    """Describe court shape relative to the detected net/pole span."""

    feet = np.asarray(model.pole_feet, dtype=float)
    corners = np.asarray(model.image_corners, dtype=float)
    if feet.shape != (2, 2) or corners.shape != (4, 2):
        return None
    if not np.isfinite(feet).all() or not np.isfinite(corners).all():
        return None
    span = float(np.linalg.norm(feet[1] - feet[0]))
    if span < 20.0:
        return None
    center = np.mean(feet, axis=0)
    width_axis = (feet[1] - feet[0]) / span
    depth_axis = np.asarray((-width_axis[1], width_axis[0]), dtype=float)
    offsets = corners - center
    # Compare shapes in a basis attached to the current net.  This makes the
    # majority prior insensitive to small camera roll or net rotation.
    signature = np.stack(
        [offsets @ width_axis, offsets @ depth_axis],
        axis=1,
    ) / span
    normalized_area = _polygon_area(corners) / max(1.0, span * span)
    return signature, normalized_area, span


def _rebuild_model_from_corners(model: Any, corners: np.ndarray) -> Any:
    """Rebuild a line-ground Homography after a majority-shape correction."""

    corrected = copy.deepcopy(model)
    corrected.image_corners = np.asarray(corners, dtype=float)
    mapping_name = str(corrected.diagnostics.get("mapping", ""))
    if mapping_name.startswith("ground_plane_homography"):
        corrected.homography_world_to_image = cv2.getPerspectiveTransform(
            np.asarray(corrected.world_corners, dtype=np.float32),
            corrected.image_corners.astype(np.float32),
        )
    corrected.diagnostics = dict(corrected.diagnostics)
    corrected.diagnostics.update(
        {
            "majority_geometry_correction": True,
            "majority_geometry_correction_method": "local_median_normalized_court_shape",
        }
    )
    return corrected


def _model_basis_state(model: Any) -> tuple[np.ndarray, float, float, np.ndarray] | None:
    """Return center, net angle, net span, and normalized court shape."""

    shape_state = _normalized_model_shape(model)
    if shape_state is None:
        return None
    shape, _, span = shape_state
    feet = np.asarray(model.pole_feet, dtype=float)
    delta = feet[1] - feet[0]
    angle = math.atan2(float(delta[1]), float(delta[0]))
    return np.mean(feet, axis=0), angle, span, shape


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _reconstruct_pole_basis(
    center: np.ndarray,
    angle: float,
    span: float,
    shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct pole feet and court corners from filtered relative state."""

    width_axis = np.asarray((math.cos(angle), math.sin(angle)), dtype=float)
    depth_axis = np.asarray((-width_axis[1], width_axis[0]), dtype=float)
    feet = np.stack(
        [center - 0.5 * span * width_axis, center + 0.5 * span * width_axis]
    )
    corners = center + (
        shape[:, 0:1] * width_axis[None, :]
        + shape[:, 1:2] * depth_axis[None, :]
    ) * span
    return feet, corners


def _adaptive_filter_alpha(motion_ratio: float, base: float = 0.22) -> float:
    """Use low gain for jitter and high gain for genuine camera movement."""

    motion_ratio = max(0.0, float(motion_ratio))
    return float(np.clip(base + 3.8 * min(0.16, motion_ratio), base, 0.82))


def _smooth_pole_model_sequence(
    models: list[Any | None],
    indices: list[int],
) -> dict[int, Any]:
    """Adaptive temporal smoothing in a pole-attached coordinate system."""

    if not indices or models[indices[0]] is None:
        return {}
    first_model = models[indices[0]]
    first_state = _model_basis_state(first_model)
    if first_state is None:
        return {indices[0]: copy.deepcopy(first_model)}
    state_center, state_angle, state_span, state_shape = first_state
    previous_model = copy.deepcopy(first_model)
    smoothed: dict[int, Any] = {indices[0]: previous_model}

    for index in indices[1:]:
        current_model = models[index]
        if current_model is None:
            smoothed[index] = copy.deepcopy(previous_model)
            continue
        measurement = _model_basis_state(current_model)
        if measurement is None:
            smoothed[index] = copy.deepcopy(previous_model)
            continue
        measured_center, measured_angle, measured_span, measured_shape = measurement
        center_motion_ratio = float(np.linalg.norm(measured_center - state_center) / max(state_span, 20.0))
        angle_delta = abs(_wrap_angle(measured_angle - state_angle))
        span_motion_ratio = abs(math.log(max(1e-6, measured_span / max(state_span, 1e-6))))
        alpha_center = _adaptive_filter_alpha(center_motion_ratio)
        alpha_angle = _adaptive_filter_alpha(angle_delta, base=0.18)
        alpha_span = _adaptive_filter_alpha(span_motion_ratio, base=0.18)
        shape_error = float(np.max(np.linalg.norm(measured_shape - state_shape, axis=1)))
        alpha_shape = float(np.clip(0.12 + 1.4 * min(0.18, center_motion_ratio), 0.12, 0.34))
        if shape_error > 0.50:
            # A sudden shape change is much more likely a line assignment
            # error than a real camera event; let the majority prior dominate.
            alpha_shape *= 0.25

        state_center = state_center + alpha_center * (measured_center - state_center)
        state_angle = state_angle + alpha_angle * _wrap_angle(measured_angle - state_angle)
        state_span = state_span * math.exp(alpha_span * math.log(max(1e-6, measured_span / max(state_span, 1e-6))))
        state_shape = state_shape + alpha_shape * (measured_shape - state_shape)
        filtered_feet, filtered_corners = _reconstruct_pole_basis(
            state_center,
            state_angle,
            state_span,
            state_shape,
        )

        filtered_model = copy.deepcopy(current_model)
        filtered_model.pole_feet = filtered_feet
        filtered_model.image_corners = filtered_corners
        point_alpha = _adaptive_filter_alpha(center_motion_ratio, base=0.20)
        filtered_model.net_points = (
            np.asarray(previous_model.net_points, dtype=float)
            + point_alpha
            * (
                np.asarray(current_model.net_points, dtype=float)
                - np.asarray(previous_model.net_points, dtype=float)
            )
        )
        filtered_model.player_intersection = (
            np.asarray(previous_model.player_intersection, dtype=float)
            + point_alpha
            * (
                np.asarray(current_model.player_intersection, dtype=float)
                - np.asarray(previous_model.player_intersection, dtype=float)
            )
        )
        mapping_name = str(filtered_model.diagnostics.get("mapping", ""))
        if mapping_name.startswith("ground_plane_homography"):
            filtered_model.homography_world_to_image = cv2.getPerspectiveTransform(
                np.asarray(filtered_model.world_corners, dtype=np.float32),
                filtered_model.image_corners.astype(np.float32),
            )
        filtered_model.diagnostics = dict(filtered_model.diagnostics)
        filtered_model.diagnostics.update(
            {
                "temporal_filter": "adaptive_pole_basis_filter",
                "temporal_center_alpha": round(float(alpha_center), 5),
                "temporal_angle_alpha": round(float(alpha_angle), 5),
                "temporal_span_alpha": round(float(alpha_span), 5),
                "temporal_shape_alpha": round(float(alpha_shape), 5),
                "temporal_shape_error": round(float(shape_error), 5),
            }
        )
        previous_model = filtered_model
        smoothed[index] = filtered_model
    return smoothed


def _smooth_tracked_pole_models(
    models: list[Any | None],
    anchor_frame: int | None,
) -> list[Any | None]:
    """Smooth both sides of the trusted anchor without changing court identity."""

    if anchor_frame is None or not models:
        return models
    smoothed: list[Any | None] = [None] * len(models)
    forward = _smooth_pole_model_sequence(models, list(range(anchor_frame, len(models))))
    backward = _smooth_pole_model_sequence(models, list(range(anchor_frame, -1, -1)))
    for index, model in {**backward, **forward}.items():
        smoothed[index] = model
    return smoothed


def _correct_majority_geometry_outliers(
    raw_models: list[Any | None],
) -> tuple[list[Any | None], int]:
    """Replace isolated court-frame shrink/expansion with local-majority shape.

    The court is normalized by the selected pole span, so a camera translation
    or mild zoom can remain in the current frame while a wrong service-line
    hypothesis cannot suddenly make the court much smaller or larger.
    """

    corrected_models = list(raw_models)
    valid = [
        _normalized_model_shape(model) if model is not None else None
        for model in raw_models
    ]
    global_shapes = [item[0] for item in valid if item is not None]
    global_areas = [item[1] for item in valid if item is not None]
    global_median_shape = (
        np.median(np.stack(global_shapes), axis=0)
        if global_shapes
        else None
    )
    global_median_area = (
        float(np.median(np.asarray(global_areas, dtype=float)))
        if global_areas
        else None
    )
    correction_count = 0
    radius = 3
    for index, model in enumerate(raw_models):
        current = valid[index]
        if model is None or current is None:
            continue
        neighbor_shapes: list[np.ndarray] = []
        neighbor_areas: list[float] = []
        for neighbor_index in range(max(0, index - radius), min(len(raw_models), index + radius + 1)):
            if neighbor_index == index or valid[neighbor_index] is None:
                continue
            neighbor_shapes.append(valid[neighbor_index][0])
            neighbor_areas.append(valid[neighbor_index][1])
        if len(neighbor_shapes) < 3:
            continue
        median_shape = np.median(np.stack(neighbor_shapes), axis=0)
        median_area = float(np.median(np.asarray(neighbor_areas, dtype=float)))
        current_shape, current_area, current_span = current
        local_area_ratio = current_area / max(1e-8, median_area)
        local_shape_error = float(np.max(np.linalg.norm(current_shape - median_shape, axis=1)))
        global_area_ratio = (
            current_area / max(1e-8, global_median_area)
            if global_median_area is not None
            else 1.0
        )
        global_shape_error = (
            float(np.max(np.linalg.norm(current_shape - global_median_shape, axis=1)))
            if global_median_shape is not None
            else 0.0
        )
        local_bad = local_area_ratio < 0.72 or local_area_ratio > 1.40 or local_shape_error > 0.62
        global_bad = global_area_ratio < 0.72 or global_area_ratio > 1.40 or global_shape_error > 0.62
        if not local_bad and not global_bad:
            continue

        current_feet = np.asarray(model.pole_feet, dtype=float)
        current_center = np.mean(current_feet, axis=0)
        current_width_axis = (current_feet[1] - current_feet[0]) / max(current_span, 1e-8)
        current_depth_axis = np.asarray((-current_width_axis[1], current_width_axis[0]), dtype=float)
        # Global majority wins when a wrong line hypothesis persists for
        # several consecutive frames; local median is only used for isolated
        # deviations that are still compatible with the video-wide shape.
        if global_bad:
            reference_shape = global_median_shape
            correction_method = "global_median_normalized_court_shape"
            reference_area_ratio = global_area_ratio
            reference_shape_error = global_shape_error
        else:
            reference_shape = median_shape
            correction_method = "local_median_normalized_court_shape"
            reference_area_ratio = local_area_ratio
            reference_shape_error = local_shape_error
        corrected_offsets = (
            reference_shape[:, 0:1] * current_width_axis[None, :]
            + reference_shape[:, 1:2] * current_depth_axis[None, :]
        ) * current_span
        corrected_corners = current_center + corrected_offsets
        if not np.isfinite(corrected_corners).all():
            continue
        corrected_models[index] = _rebuild_model_from_corners(model, corrected_corners)
        corrected_models[index].diagnostics.update(
            {
                "majority_area_ratio_before": round(float(reference_area_ratio), 5),
                "majority_shape_error_before": round(float(reference_shape_error), 5),
                "majority_geometry_reference": correction_method,
            }
        )
        correction_count += 1
    return corrected_models, correction_count


def _blend_pole_models(
    previous: Any,
    current: Any,
    width: int,
    height: int,
    motion_px: float,
) -> Any:
    """Smooth a valid current detection while recomputing its PnP mapping."""

    corner_motion_px = float(
        np.max(
            np.linalg.norm(
                np.asarray(current.image_corners, dtype=float)
                - np.asarray(previous.image_corners, dtype=float),
                axis=1,
            )
        )
    )
    # Keep most of the current detection so camera movement is followed, while
    # damping one-frame Hough jitter.  A larger motion gets less smoothing.
    alpha = 0.80 if corner_motion_px < 70.0 else 0.92
    model = copy.deepcopy(current)
    for field_name in ("pole_feet", "net_points", "player_intersection", "image_corners"):
        previous_value = np.asarray(getattr(previous, field_name), dtype=float)
        current_value = np.asarray(getattr(current, field_name), dtype=float)
        setattr(model, field_name, alpha * current_value + (1.0 - alpha) * previous_value)

    mapping_name = str(model.diagnostics.get("mapping", ""))
    try:
        if mapping_name.startswith("ground_plane_homography"):
            source_corners = np.asarray(model.world_corners, dtype=np.float32)
            target_corners = np.asarray(model.image_corners, dtype=np.float32)
            model.homography_world_to_image = cv2.getPerspectiveTransform(
                source_corners,
                target_corners,
            )
        else:
            homography, image_corners = recompute_pnp_ground_mapping(
                model.pole_feet,
                model.net_points,
                width,
                height,
            )
            model.homography_world_to_image = homography
            model.image_corners = image_corners
    except (cv2.error, RuntimeError, ValueError, np.linalg.LinAlgError):
        # Keep the current frame's already-valid mapping if the blended four
        # points happen to be numerically degenerate.
        pass
    model.diagnostics = dict(model.diagnostics)
    model.diagnostics.update(
        {
            "temporal_tracking": "accepted_current_detection_and_recomputed_mapping",
            "temporal_blend_alpha": round(float(alpha), 4),
            "temporal_motion_px": round(float(motion_px), 3),
            "temporal_corner_motion_px": round(float(corner_motion_px), 3),
        }
    )
    return model


def _hold_pole_model(model: Any, reason: str, source_frame: int) -> Any:
    """Copy the last trusted model and record why this frame was held."""

    held = copy.deepcopy(model)
    held.diagnostics = dict(held.diagnostics)
    held.diagnostics.update(
        {
            "temporal_tracking": reason,
            "temporal_source_frame": int(source_frame),
        }
    )
    return held


def _track_pole_models(
    raw_models: list[Any | None],
    anchor_frame: int | None,
    width: int,
    height: int,
) -> tuple[list[Any | None], int | None]:
    """Track the target pole pair from a trusted anchor in both directions.

    Pole IDs are local to each frame, so position—not the component ID—is used
    for identity.  This rejects a jump to an adjacent court while still
    allowing the entire mapped court to move with camera shake.
    """

    tracked: list[Any | None] = [None] * len(raw_models)
    valid_indices = [index for index, model in enumerate(raw_models) if model is not None]
    if not valid_indices:
        return tracked, None

    if anchor_frame is None or anchor_frame not in valid_indices:
        anchor_frame = min(valid_indices, key=lambda index: abs(index - (anchor_frame or 0)))
    tracked[anchor_frame] = copy.deepcopy(raw_models[anchor_frame])
    gate_px = max(70.0, 0.10 * float(width))

    def extend(indices: list[int]) -> None:
        previous = tracked[anchor_frame]
        previous_index = anchor_frame
        if previous is None:
            return
        for index in indices:
            current = raw_models[index]
            if current is None:
                tracked[index] = _hold_pole_model(previous, "held_previous_detection_missing", previous_index)
            else:
                motion_px = _pole_model_motion(previous, current)
                corner_motion_px = float(
                    np.max(
                        np.linalg.norm(
                            np.asarray(current.image_corners, dtype=float)
                            - np.asarray(previous.image_corners, dtype=float),
                            axis=1,
                        )
                    )
                )
                previous_area = _polygon_area(previous.image_corners)
                current_area = _polygon_area(current.image_corners)
                area_ratio = current_area / max(1.0, previous_area)
                geometry_jump = (
                    corner_motion_px > max(260.0, 0.27 * float(width))
                    or area_ratio < 0.45
                    or area_ratio > 2.20
                )
                if motion_px <= gate_px and not geometry_jump:
                    tracked[index] = _blend_pole_models(previous, current, width, height, motion_px)
                else:
                    reason = (
                        "held_previous_detection_geometry_jump_rejected"
                        if geometry_jump
                        else "held_previous_detection_jump_rejected"
                    )
                    tracked[index] = _hold_pole_model(previous, reason, previous_index)
            previous = tracked[index]
            previous_index = index

    extend(list(range(anchor_frame - 1, -1, -1)))
    extend(list(range(anchor_frame + 1, len(raw_models))))
    return tracked, anchor_frame


def run(
    video_path: Path,
    output_dir: Path,
    frame: int | None,
    keyframes: int,
    save_edges: bool,
    show_labels: bool,
    interactive: bool,
    pose_csv: Path | None = None,
    use_target_filter: bool = True,
    all_frames: bool = False,
    video_output: Path | None = None,
    reference_frame: int | None = None,
    pole_mapping: bool = False,
) -> dict[str, Any]:
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")

    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = _frame_indices(cap, frame, keyframes)
    detector = CourtLineDetector(CourtDetectorConfig())
    frame_pairs, all_player_points, player_axis_angle = _load_player_targets(
        pose_csv or Path("outputs/pose_test/pose_data_stable.csv")
    )
    pose_points = {}
    effective_pose_csv = pose_csv or Path("outputs/pose_test/pose_data_stable.csv")
    if effective_pose_csv.exists():
        pose_points = _load_points(effective_pose_csv)
    if use_target_filter and not frame_pairs:
        print("[WARN] no two-player ankle targets found; falling back to all Hough candidates")
    records: list[dict] = []
    video_path_out: Path | None = None
    video_mapping_stats: dict[str, Any] = {}
    try:
        for frame_index in indices:
            ok, image = _read_frame(cap, frame_index)
            if not ok or image is None:
                print(f"[WARN] could not read frame {frame_index}")
                continue
            pose_frame, player_points = _nearest_player_pair(frame_pairs, frame_index)
            result = detector.detect(
                image,
                frame_index=frame_index,
                target_player_points=(
                    None if pole_mapping else player_points if use_target_filter else None
                ),
                target_axis_angle_deg=None if pole_mapping else player_axis_angle if use_target_filter else None,
            )
            mapped_court = (
                select_pole_guided_court(image, result.lines, player_points)
                if pole_mapping and use_target_filter and player_points is not None
                else None
            )
            target_selected = mapped_court is not None if pole_mapping else result.target_court is not None
            if pole_mapping and mapped_court is not None:
                annotated = annotate_pole_guided_court(
                    image,
                    mapped_court,
                    player_points=player_points,
                    show_labels=show_labels,
                )
            else:
                annotated = detector.annotate(
                    image,
                    result,
                    show_ids=show_labels,
                    target_only=use_target_filter and target_selected,
                )
            output_path = output_dir / f"court_detection_frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError(f"failed to write {output_path}")
            if save_edges:
                edge_path = output_dir / f"court_detection_edges_{frame_index:06d}.jpg"
                cv2.imwrite(str(edge_path), detector.edge_preview(result))

            record = result.as_dict()
            record["pole_mapped_court"] = None if mapped_court is None else mapped_court.as_dict()
            record["annotated_image"] = str(output_path)
            record["target_pose_frame"] = pose_frame
            if save_edges:
                record["edge_preview"] = str(output_dir / f"court_detection_edges_{frame_index:06d}.jpg")
            records.append(record)
            role_counts: dict[str, int] = {}
            for line in result.lines:
                role_counts[line.role] = role_counts.get(line.role, 0) + 1
            if pole_mapping and mapped_court is not None:
                print(
                    f"frame={frame_index} pole-pair={mapped_court.pole_ids} "
                    f"net={mapped_court.net_line_id} cross_ratio={mapped_court.player_intersection_ratio:.2f} "
                    f"score={mapped_court.score:.2f}"
                )
            elif target_selected:
                court = result.target_court
                print(
                    f"frame={frame_index} target=4-sides boundaries={court.boundary_line_ids} "
                    f"net={court.net_line_id} score={court.score:.2f}"
                )
            else:
                print(f"frame={frame_index} target=not-selected all_candidates={len(result.lines)} roles={role_counts}")

            if interactive:
                cv2.imshow("Court detection - press S to save / Q to stop", annotated)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("q"), 27):
                    break
        if interactive:
            cv2.destroyAllWindows()

        if all_frames or video_output is not None:
            video_path_out = video_output or (output_dir / "court_detection_video.mp4")
            model_frame = (
                max(0, min(video_frame_count - 1, int(reference_frame)))
                if reference_frame is not None
                else max(0, (video_frame_count - 1) // 2)
            )
            video_path_out.parent.mkdir(parents=True, exist_ok=True)

            if pole_mapping:
                # The camera is not fixed.  Detect the target poles and net on
                # every frame, then track the selected pair from the trusted
                # reference frame so a neighbouring court cannot take over.
                raw_models: list[Any | None] = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for detection_frame_index in range(video_frame_count):
                    ok, detection_image = cap.read()
                    if not ok or detection_image is None:
                        raw_models.append(None)
                        continue
                    _, detection_points = _nearest_player_pair(frame_pairs, detection_frame_index)
                    detection_result = detector.detect(
                        detection_image,
                        frame_index=detection_frame_index,
                        target_player_points=None,
                    )
                    model = (
                        select_pole_guided_court(
                            detection_image,
                            detection_result.lines,
                            detection_points,
                        )
                        if use_target_filter and detection_points is not None
                        else None
                    )
                    raw_models.append(model)
                    if (detection_frame_index + 1) % 25 == 0 or detection_frame_index == video_frame_count - 1:
                        print(
                            f"per-frame court mapping: {detection_frame_index + 1}/{video_frame_count} "
                            f"valid={sum(item is not None for item in raw_models)}"
                        )

                raw_models, majority_correction_count = _correct_majority_geometry_outliers(raw_models)
                tracked_models, mapping_anchor_frame = _track_pole_models(
                    raw_models,
                    model_frame,
                    video_width,
                    video_height,
                )
                # Smooth the tracked court in a coordinate system attached to
                # the selected net.  This suppresses line-assignment jitter
                # without freezing the court at a fixed image location.
                tracked_models = _smooth_tracked_pole_models(
                    tracked_models,
                    mapping_anchor_frame,
                )
                video_mapping_stats = {
                    "raw_valid_frames": int(sum(item is not None for item in raw_models)),
                    "total_frames": int(len(raw_models)),
                    "tracked_model_frames": int(sum(item is not None for item in tracked_models)),
                    "anchor_frame": mapping_anchor_frame,
                    "recomputed_per_frame": True,
                    "temporal_filter": "adaptive_pole_basis_filter",
                    "majority_geometry_corrections": int(majority_correction_count),
                    "line_ground_homography_frames": int(
                        sum(
                            item is not None
                            and str(item.diagnostics.get("mapping", "")).startswith("ground_plane_homography")
                            for item in raw_models
                        )
                    ),
                    "pnp_fallback_frames": int(
                        sum(
                            item is not None
                            and bool(item.diagnostics.get("mapping_fallback", False))
                            for item in raw_models
                        )
                    ),
                }
                if mapping_anchor_frame is None:
                    print("[WARN] no usable per-frame pole/net model; video will show pose only")
                else:
                    print(
                        f"per-frame mapping anchor={mapping_anchor_frame} "
                        f"raw-valid={sum(item is not None for item in raw_models)}/{len(raw_models)}"
                    )

                writer = cv2.VideoWriter(
                    str(video_path_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video_fps if video_fps > 0 else 25.0,
                    (video_width, video_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to create video writer: {video_path_out}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_frame_index = 0
                try:
                    while True:
                        ok, image = cap.read()
                        if not ok or image is None:
                            break
                        pose_frame, current_points = _nearest_player_pair(frame_pairs, video_frame_index)
                        mapped_model = (
                            tracked_models[video_frame_index]
                            if video_frame_index < len(tracked_models)
                            else None
                        )
                        if mapped_model is not None:
                            annotated = annotate_pole_guided_court(
                                image,
                                mapped_model,
                                player_points=current_points,
                                show_labels=show_labels,
                            )
                        else:
                            annotated = image.copy()
                        _draw_pose_tracks(annotated, pose_points, pose_frame)
                        anchor_text = (
                            f"{mapping_anchor_frame}"
                            if mapping_anchor_frame is not None
                            else "none"
                        )
                        cv2.putText(
                            annotated,
                            f"Per-frame pole mapping | anchor {anchor_text} | video frame {video_frame_index}",
                            (18, video_height - 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        writer.write(annotated)
                        video_frame_index += 1
                finally:
                    writer.release()
            else:
                ok, reference_image = _read_frame(cap, model_frame)
                if not ok or reference_image is None:
                    raise RuntimeError(f"could not read reference frame {model_frame}")
                reference_pose_frame, reference_points = _nearest_player_pair(frame_pairs, model_frame)
                reference_result = detector.detect(
                    reference_image,
                    frame_index=model_frame,
                    target_player_points=reference_points if use_target_filter else None,
                    target_axis_angle_deg=player_axis_angle if use_target_filter else None,
                )
                if reference_result.target_court is None:
                    print(f"[WARN] reference frame {model_frame} has no usable court model; video will show candidates")

                writer = cv2.VideoWriter(
                    str(video_path_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video_fps if video_fps > 0 else 25.0,
                    (video_width, video_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to create video writer: {video_path_out}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_frame_index = 0
                try:
                    while True:
                        ok, image = cap.read()
                        if not ok or image is None:
                            break
                        pose_frame, current_points = _nearest_player_pair(frame_pairs, video_frame_index)
                        frame_result = copy.deepcopy(reference_result)
                        frame_result.frame_index = video_frame_index
                        if frame_result.target_court is not None:
                            frame_result.target_court.player_points = (
                                current_points.copy()
                                if current_points is not None
                                else reference_result.target_court.player_points.copy()
                            )
                        annotated = detector.annotate(
                            image,
                            frame_result,
                            show_ids=show_labels,
                            target_only=use_target_filter and frame_result.target_court is not None,
                        )
                        _draw_pose_tracks(annotated, pose_points, pose_frame)
                        cv2.putText(
                            annotated,
                            f"Court model frame {model_frame} | video frame {video_frame_index}",
                            (18, video_height - 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        writer.write(annotated)
                        video_frame_index += 1
                finally:
                    writer.release()
            print(f"detection video: {video_path_out}")
    finally:
        cap.release()

    metadata = {
        "video": str(video_path),
        "frame_count": video_frame_count,
        "fps": video_fps,
        "width": video_width,
        "height": video_height,
        "requested_frame_indices": indices,
        "pose_csv": None if pose_csv is None else str(pose_csv),
        "player_guided": bool(use_target_filter and bool(frame_pairs)),
        "player_axis_angle_deg": player_axis_angle,
        "target_description": (
            "adjacent pole bases whose connecting segment intersects the two-player foot segment; "
            "standard 13.4m x 6.1m court projected with pole-guided court-line Homography; guarded net-only PnP fallback"
            if pole_mapping
            else "player corridor -> net edge -> two posts -> visible or synthesized court long edges; image-space only"
        ),
        "mapping_mode": (
            "pole_pair_line_ground_homography_per_frame_with_temporal_pair_tracking"
            if pole_mapping
            else "legacy_player_guided_image_space"
        ),
        "per_frame_mapping": bool(pole_mapping and (all_frames or video_output is not None)),
        "video_output": None if video_path_out is None else str(video_path_out),
        "video_mapping_stats": video_mapping_stats,
        "frames": records,
        "scope": "image-space court structure only; no homography or 3-D reconstruction",
    }
    json_path = output_dir / "court_detection_lines.json"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(records)} frame results to {output_dir}")
    print(f"line parameters: {json_path}")
    return metadata


def main() -> None:
    args = _parse_args()
    run(
        video_path=args.video,
        output_dir=args.output_dir,
        frame=args.frame,
        keyframes=args.keyframes,
        save_edges=not args.no_edge_preview,
        show_labels=not args.no_labels,
        interactive=args.interactive,
        pose_csv=args.pose_csv,
        use_target_filter=not args.no_target_filter,
        all_frames=args.all_frames,
        video_output=args.video_output,
        reference_frame=args.reference_frame,
        pole_mapping=args.pole_mapping,
    )


if __name__ == "__main__":
    main()
