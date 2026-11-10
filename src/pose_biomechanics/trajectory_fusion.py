"""Robust multi-cue court trajectory optimization for reconstructed humans.

The optimizer repairs root translation only.  It combines stable 2-D pelvis
motion, learned foot-contact confidence, court measurements, optional video
depth and the original GVHMR trajectory in one full-sequence sparse solve.
Articulated SMPL-X motion and every vertical coordinate remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.sparse import block_diag, csr_matrix, diags, hstack, vstack
from scipy.sparse.linalg import lsqr


@dataclass(frozen=True)
class TrajectoryFusionConfig:
    min_pelvis_confidence: float = 0.20
    ground_contact_threshold: float = 0.55
    raw_position_weight: float = 0.12
    raw_velocity_weight: float = 1.25
    pelvis_position_weight: float = 0.45
    pelvis_velocity_weight: float = 5.0
    ground_position_weight: float = 8.0
    depth_weight: float = 2.0
    acceleration_weight: float = 30.0
    acceleration_weight_min: float = 3.0
    jerk_weight: float = 6.0
    jerk_weight_min: float = 0.6
    plausible_speed_mps: float = 7.5
    absolute_speed_limit_mps: float = 8.0
    max_correction_m: float = 2.50
    court_margin_m: float = 0.10
    robust_floor_m: float = 0.04


@dataclass(frozen=True)
class TrajectoryFusionResult:
    corrected_xy_m: np.ndarray
    pelvis_target_xy_m: np.ndarray
    aligned_pelvis_target_xy_m: np.ndarray
    raw_velocity_quality: np.ndarray
    pelvis_quality: np.ndarray
    ground_quality: np.ndarray
    motion_score: np.ndarray
    applied_xy_m: np.ndarray
    pelvis_alignment_offset_xy_m: np.ndarray


def _mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    center = np.median(finite)
    return float(1.4826 * np.median(np.abs(finite - center)))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else 0.0
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _difference_matrix(length: int, order: int) -> csr_matrix:
    if order == 1:
        return diags([-np.ones(length - 1), np.ones(length - 1)], [0, 1], shape=(length - 1, length)).tocsr()
    if order == 2:
        return diags(
            [np.ones(length - 2), -2.0 * np.ones(length - 2), np.ones(length - 2)],
            [0, 1, 2],
            shape=(length - 2, length),
        ).tocsr()
    if order == 3:
        return diags(
            [-np.ones(length - 3), 3.0 * np.ones(length - 3), -3.0 * np.ones(length - 3), np.ones(length - 3)],
            [0, 1, 2, 3],
            shape=(length - 3, length),
        ).tocsr()
    raise ValueError(f"Unsupported difference order: {order}")


def image_points_to_court_xy_at_height(
    image_xy_px: np.ndarray,
    court_height_m: np.ndarray,
    camera_matrix: np.ndarray,
    court_to_camera_rotation: np.ndarray,
    court_to_camera_translation: np.ndarray,
) -> np.ndarray:
    """Back-project image points to horizontal court planes at known heights."""
    image_xy = np.asarray(image_xy_px, dtype=np.float64)
    height = np.asarray(court_height_m, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    rotation = np.asarray(court_to_camera_rotation, dtype=np.float64)
    translation = np.asarray(court_to_camera_translation, dtype=np.float64)
    if image_xy.ndim != 2 or image_xy.shape[1] != 2 or height.shape != (len(image_xy),):
        raise ValueError("image_xy_px must be Nx2 and court_height_m must be N")
    if camera_matrix.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Camera calibration must be K=3x3, R=3x3 and t=3")

    output = np.full((len(image_xy), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(image_xy).all(axis=1) & np.isfinite(height)
    homogeneous = np.column_stack([image_xy[valid], np.ones(int(np.count_nonzero(valid)))])
    rays = np.linalg.solve(camera_matrix, homogeneous.T).T
    for output_index, ray, z_value in zip(np.flatnonzero(valid), rays, height[valid]):
        system = np.column_stack([rotation[:, 0], rotation[:, 1], -ray])
        if abs(np.linalg.det(system)) < 1e-10:
            continue
        rhs = -(rotation[:, 2] * z_value + translation)
        output[output_index] = np.linalg.solve(system, rhs)[:2]
    return output


def project_court_points(
    court_xyz_m: np.ndarray,
    camera_matrix: np.ndarray,
    court_to_camera_rotation: np.ndarray,
    court_to_camera_translation: np.ndarray,
) -> np.ndarray:
    points = np.asarray(court_xyz_m, dtype=np.float64)
    camera = points @ np.asarray(court_to_camera_rotation, dtype=np.float64).T + np.asarray(
        court_to_camera_translation, dtype=np.float64
    )
    pixels_h = camera @ np.asarray(camera_matrix, dtype=np.float64).T
    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(pixels_h).all(axis=1) & (pixels_h[:, 2] > 1e-6)
    output[valid] = pixels_h[valid, :2] / pixels_h[valid, 2:3]
    return output


def _clean_xy(values: np.ndarray, quality: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    quality = np.clip(np.asarray(quality, dtype=np.float64), 0.0, 1.0)
    cleaned = values.copy()
    valid = np.isfinite(values).all(axis=1) & (quality > 0)
    if int(np.count_nonzero(valid)) < 2:
        return cleaned, np.zeros_like(quality)
    indices = np.arange(len(values))
    for axis in range(2):
        filled = np.interp(indices, indices[valid], values[valid, axis])
        local = median_filter(filled, size=window, mode="nearest")
        residual = np.abs(filled - local)
        scale = max(_mad(residual[valid]), 0.02)
        outlier = valid & (residual > 4.0 * scale)
        cleaned[:, axis] = filled
        cleaned[outlier, axis] = local[outlier]
        quality[outlier] *= 0.10
    return cleaned, quality


def _step_quality(step: np.ndarray, fps: float, plausible_speed_mps: float, floor_m: float) -> np.ndarray:
    speed = np.linalg.norm(step, axis=1) * fps
    center = float(np.median(speed[np.isfinite(speed)])) if np.isfinite(speed).any() else 0.0
    scale = max(_mad(speed), plausible_speed_mps * 0.12, floor_m * fps)
    excess = np.maximum(speed - max(center + 3.0 * scale, plausible_speed_mps), 0.0)
    return 1.0 / (1.0 + np.square(excess / max(scale, 1e-6)))


def _clip_vector_steps(values: np.ndarray, maximum_step: float) -> np.ndarray:
    original = np.asarray(values, dtype=np.float64)
    if len(original) < 2 or maximum_step <= 0:
        return original.copy()
    forward = original.copy()
    for index in range(1, len(forward)):
        delta = forward[index] - forward[index - 1]
        norm = float(np.linalg.norm(delta))
        if norm > maximum_step:
            forward[index] = forward[index - 1] + delta * (maximum_step / norm)
    backward = original.copy()
    for index in range(len(backward) - 2, -1, -1):
        delta = backward[index] - backward[index + 1]
        norm = float(np.linalg.norm(delta))
        if norm > maximum_step:
            backward[index] = backward[index + 1] + delta * (maximum_step / norm)
    # The average of two step-bounded trajectories is also step-bounded and
    # avoids privileging either endpoint.
    return 0.5 * (forward + backward)


def fuse_court_trajectory(
    raw_root_xy_m: np.ndarray,
    root_height_m: np.ndarray,
    pelvis_image_xy_px: np.ndarray,
    pelvis_confidence: np.ndarray,
    ground_xy_m: np.ndarray,
    foot_stationary_confidence: np.ndarray,
    camera_matrix: np.ndarray,
    court_to_camera_rotation: np.ndarray,
    court_to_camera_translation: np.ndarray,
    fps: float,
    court_width_m: float,
    court_length_m: float,
    depth_target_camera_m: np.ndarray | None = None,
    depth_quality: np.ndarray | None = None,
    config: TrajectoryFusionConfig | None = None,
) -> TrajectoryFusionResult:
    """Fuse all trajectory cues in one sparse, full-sequence optimization."""
    config = config or TrajectoryFusionConfig()
    raw = np.asarray(raw_root_xy_m, dtype=np.float64)
    height = np.asarray(root_height_m, dtype=np.float64)
    pelvis_uv = np.asarray(pelvis_image_xy_px, dtype=np.float64)
    pelvis_conf = np.clip(np.asarray(pelvis_confidence, dtype=np.float64), 0.0, 1.0)
    ground = np.asarray(ground_xy_m, dtype=np.float64)
    contact = np.clip(np.nan_to_num(foot_stationary_confidence, nan=0.0), 0.0, 1.0)
    length = len(raw)
    if length < 4 or raw.shape != (length, 2) or pelvis_uv.shape != (length, 2) or ground.shape != (length, 2):
        raise ValueError("Trajectory inputs must share an Nx2 shape with at least four frames")
    if height.shape != (length,) or pelvis_conf.shape != (length,) or contact.shape != (length,):
        raise ValueError("Height and confidence inputs must share the trajectory length")
    if fps <= 0:
        raise ValueError("FPS must be positive")

    pelvis_target = image_points_to_court_xy_at_height(
        pelvis_uv,
        height,
        camera_matrix,
        court_to_camera_rotation,
        court_to_camera_translation,
    )
    pelvis_quality = pelvis_conf.copy()
    pelvis_quality[pelvis_conf < config.min_pelvis_confidence] = 0.0
    pelvis_quality[~np.isfinite(pelvis_target).all(axis=1)] = 0.0
    pelvis_clean, pelvis_quality = _clean_xy(pelvis_target, pelvis_quality)

    ground_quality = contact.copy()
    ground_quality[contact < config.ground_contact_threshold] = 0.0
    ground_quality[~np.isfinite(ground).all(axis=1)] = 0.0
    ground_clean, ground_quality = _clean_xy(ground, ground_quality)
    ground_step_quality = np.ones(length)
    ground_step_quality[1:] = _step_quality(
        np.diff(ground_clean, axis=0), fps, config.plausible_speed_mps, config.robust_floor_m
    )
    ground_quality *= ground_step_quality

    alignment_quality = ground_quality * pelvis_quality
    if np.count_nonzero(alignment_quality > 0) >= 3:
        offset = np.array(
            [
                _weighted_median(ground_clean[:, axis] - pelvis_clean[:, axis], alignment_quality)
                for axis in range(2)
            ],
            dtype=np.float64,
        )
    else:
        offset = np.array(
            [
                _weighted_median(raw[:, axis] - pelvis_clean[:, axis], pelvis_quality)
                for axis in range(2)
            ],
            dtype=np.float64,
        )
    aligned_pelvis = pelvis_clean + offset

    # A high contact score does not make a bad 2-D ground projection valid.
    # Reject ground clusters that disagree with the independently observed
    # pelvis ray, while retaining ordinary root/foot offsets and body lean.
    ground_pelvis_residual = np.linalg.norm(ground_clean - aligned_pelvis, axis=1)
    consistency_valid = (ground_quality > 0) & np.isfinite(ground_pelvis_residual)
    if np.count_nonzero(consistency_valid) >= 3:
        consistency_center = _weighted_median(
            ground_pelvis_residual[consistency_valid],
            ground_quality[consistency_valid],
        )
        consistency_scale = max(
            _mad(ground_pelvis_residual[consistency_valid]),
            0.15,
        )
        consistency_limit = consistency_center + 3.0 * consistency_scale
        excess = np.maximum(ground_pelvis_residual - consistency_limit, 0.0)
        ground_quality *= 1.0 / (1.0 + np.square(excess / consistency_scale))

    raw_step = np.diff(raw, axis=0)
    pelvis_step = np.diff(aligned_pelvis, axis=0)
    pelvis_step_quality = _step_quality(
        pelvis_step, fps, config.plausible_speed_mps, config.robust_floor_m
    )
    mismatch = np.linalg.norm(raw_step - pelvis_step, axis=1)
    mismatch_scale = max(_mad(mismatch), config.robust_floor_m)
    raw_velocity_quality = 1.0 / (1.0 + np.square(mismatch / (3.0 * mismatch_scale)))
    raw_speed = np.linalg.norm(raw_step, axis=1) * fps
    excessive = raw_speed > config.plausible_speed_mps
    raw_velocity_quality[excessive] *= np.square(
        config.plausible_speed_mps / np.maximum(raw_speed[excessive], config.plausible_speed_mps)
    )
    pair_pelvis_quality = np.minimum(pelvis_quality[:-1], pelvis_quality[1:]) * pelvis_step_quality

    motion_speed = np.linalg.norm(pelvis_step, axis=1) * fps
    motion_step = np.clip(motion_speed / max(config.plausible_speed_mps * 0.55, 1e-6), 0.0, 1.0)
    motion_score = np.zeros(length, dtype=np.float64)
    motion_score[1:] = motion_step
    motion_score[:-1] = np.maximum(motion_score[:-1], motion_step)

    identity = diags(np.ones(length), 0, shape=(length, length), format="csr")
    zero = csr_matrix((length, length))
    point_matrix = vstack([hstack([identity, zero]), hstack([zero, identity])], format="csr")
    d1 = _difference_matrix(length, 1)
    velocity_matrix = block_diag((d1, d1), format="csr")
    d2 = _difference_matrix(length, 2)
    acceleration_matrix = block_diag((d2, d2), format="csr")
    d3 = _difference_matrix(length, 3)
    jerk_matrix = block_diag((d3, d3), format="csr")

    matrices: list[csr_matrix] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    def add(matrix: csr_matrix, target: np.ndarray, weight: np.ndarray) -> None:
        matrices.append(matrix)
        targets.append(np.asarray(target, dtype=np.float64))
        weights.append(np.maximum(np.asarray(weight, dtype=np.float64), 0.0))

    raw_point_quality = np.ones(length, dtype=np.float64)
    raw_point_quality[1:] = np.minimum(raw_point_quality[1:], raw_velocity_quality)
    raw_point_quality[:-1] = np.minimum(raw_point_quality[:-1], raw_velocity_quality)
    add(point_matrix, np.r_[raw[:, 0], raw[:, 1]], np.tile(config.raw_position_weight * raw_point_quality, 2))
    add(
        point_matrix,
        np.r_[aligned_pelvis[:, 0], aligned_pelvis[:, 1]],
        np.tile(config.pelvis_position_weight * pelvis_quality, 2),
    )
    add(
        point_matrix,
        np.r_[ground_clean[:, 0], ground_clean[:, 1]],
        np.tile(config.ground_position_weight * ground_quality, 2),
    )
    add(
        velocity_matrix,
        np.r_[raw_step[:, 0], raw_step[:, 1]],
        np.tile(config.raw_velocity_weight * raw_velocity_quality, 2),
    )
    add(
        velocity_matrix,
        np.r_[pelvis_step[:, 0], pelvis_step[:, 1]],
        np.tile(config.pelvis_velocity_weight * pair_pelvis_quality, 2),
    )

    acceleration_motion = motion_score[1:-1]
    acceleration_weight = config.acceleration_weight_min + (
        config.acceleration_weight - config.acceleration_weight_min
    ) * (1.0 - 0.80 * acceleration_motion)
    add(acceleration_matrix, np.zeros(2 * (length - 2)), np.tile(acceleration_weight, 2))
    jerk_motion = np.maximum.reduce(
        [motion_score[:-3], motion_score[1:-2], motion_score[2:-1], motion_score[3:]]
    )
    jerk_weight = config.jerk_weight_min + (config.jerk_weight - config.jerk_weight_min) * (
        1.0 - 0.80 * jerk_motion
    )
    add(jerk_matrix, np.zeros(2 * (length - 3)), np.tile(jerk_weight, 2))

    if depth_target_camera_m is not None:
        depth_target = np.asarray(depth_target_camera_m, dtype=np.float64)
        depth_conf = np.ones(length, dtype=np.float64) if depth_quality is None else np.asarray(depth_quality, dtype=np.float64)
        if depth_target.shape != (length,) or depth_conf.shape != (length,):
            raise ValueError("Depth target and quality must share the trajectory length")
        rotation = np.asarray(court_to_camera_rotation, dtype=np.float64)
        translation = np.asarray(court_to_camera_translation, dtype=np.float64)
        depth_matrix = hstack([rotation[2, 0] * identity, rotation[2, 1] * identity], format="csr")
        depth_rhs = depth_target - rotation[2, 2] * height - translation[2]
        depth_conf = np.clip(np.nan_to_num(depth_conf, nan=0.0), 0.0, 1.0)
        depth_conf[~np.isfinite(depth_rhs)] = 0.0
        add(depth_matrix, np.nan_to_num(depth_rhs), config.depth_weight * depth_conf)

    matrix = vstack(matrices, format="csr")
    target = np.concatenate(targets)
    weight = np.concatenate(weights)
    valid_rows = np.isfinite(target) & np.isfinite(weight) & (weight > 0)
    weighted_matrix = matrix[valid_rows].multiply(np.sqrt(weight[valid_rows])[:, None])
    weighted_target = target[valid_rows] * np.sqrt(weight[valid_rows])
    solution = lsqr(weighted_matrix, weighted_target, atol=1e-8, btol=1e-8, iter_lim=5000)[0]
    corrected = np.column_stack([solution[:length], solution[length:]])

    correction = corrected - raw
    correction_norm = np.linalg.norm(correction, axis=1)
    over = correction_norm > config.max_correction_m
    correction[over] *= config.max_correction_m / correction_norm[over, None]
    corrected = raw + correction
    corrected = _clip_vector_steps(corrected, config.absolute_speed_limit_mps / fps)
    limits = np.array(
        [court_width_m / 2.0 + config.court_margin_m, court_length_m / 2.0 + config.court_margin_m],
        dtype=np.float64,
    )
    corrected = np.clip(corrected, -limits, limits)

    return TrajectoryFusionResult(
        corrected_xy_m=corrected,
        pelvis_target_xy_m=pelvis_target,
        aligned_pelvis_target_xy_m=aligned_pelvis,
        raw_velocity_quality=np.r_[raw_velocity_quality[0], raw_velocity_quality],
        pelvis_quality=pelvis_quality,
        ground_quality=ground_quality,
        motion_score=motion_score,
        applied_xy_m=corrected - raw,
        pelvis_alignment_offset_xy_m=offset,
    )


def trajectory_motion_metrics(xyz_m: np.ndarray, fps: float) -> dict[str, float]:
    points = np.asarray(xyz_m, dtype=np.float64)
    speed = np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
    acceleration = np.linalg.norm(np.diff(points, n=2, axis=0), axis=1) * fps**2
    jerk = np.linalg.norm(np.diff(points, n=3, axis=0), axis=1) * fps**3
    return {
        "speed_median_mps": float(np.median(speed)),
        "speed_p95_mps": float(np.percentile(speed, 95)),
        "speed_max_mps": float(np.max(speed)),
        "acceleration_p95_mps2": float(np.percentile(acceleration, 95)),
        "acceleration_max_mps2": float(np.max(acceleration)),
        "jerk_median_mps3": float(np.median(jerk)),
        "jerk_p95_mps3": float(np.percentile(jerk, 95)),
        "jerk_max_mps3": float(np.max(jerk)),
    }
