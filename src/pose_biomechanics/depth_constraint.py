"""Depth-guided trajectory regularization for GVHMR court bundles.

The module corrects only the horizontal court-plane component that affects
camera depth.  Vertical root motion and every articulated body pose are kept
unchanged, so a jump cannot be flattened by the depth constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class DepthConstraintConfig:
    """Tunable, video-independent constraint settings."""

    min_valid_observations: int = 8
    hampel_window_seconds: float = 0.33
    hampel_sigma: float = 3.5
    smooth_window_seconds: float = 0.50
    activity_window_seconds: float = 1.00
    travel_depth_speed_mps: float = 0.45
    jump_vertical_speed_mps: float = 0.65
    jump_height_m: float = 0.16
    drift_velocity_floor_mps: float = 0.20
    drift_residual_floor_m: float = 0.08
    depth_weight: float = 4.0
    gvhmr_weight: float = 1.0
    drift_depth_gain: float = 5.0
    acceleration_weight: float = 12.0
    acceleration_weight_min: float = 1.5
    jerk_weight: float = 2.0
    jerk_weight_min: float = 0.25
    max_correction_m: float = 0.75
    max_correction_speed_mps: float = 1.80
    max_unexplained_depth_speed_mps: float = 0.90
    court_margin_m: float = 0.10


@dataclass(frozen=True)
class DepthConstraintResult:
    cleaned_observation_m: np.ndarray
    aligned_target_camera_depth_m: np.ndarray
    corrected_camera_depth_m: np.ndarray
    applied_camera_depth_delta_m: np.ndarray
    observation_quality: np.ndarray
    observation_outlier: np.ndarray
    jump_score: np.ndarray
    travel_score: np.ndarray
    drift_score: np.ndarray
    affine_scale: float
    affine_offset_m: float
    affine_residual_scale_m: float


def discover_player_ids(bundle_keys: Iterable[str]) -> list[int]:
    """Return player ids that have both root and mesh arrays in a bundle."""
    keys = set(bundle_keys)
    candidates = []
    for key in keys:
        match = re.fullmatch(r"p(\d+)_root_court", key)
        if match and f"p{match.group(1)}_vertices_court" in keys:
            candidates.append(int(match.group(1)))
    return sorted(candidates)


def _odd_window(length: int, fps: float, seconds: float, minimum: int = 5) -> int:
    if length < 3:
        return length
    desired = max(minimum, int(round(fps * seconds)))
    if desired % 2 == 0:
        desired += 1
    largest = length if length % 2 == 1 else length - 1
    return max(3, min(desired, largest))


def _robust_mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = np.median(finite)
    return float(1.4826 * np.median(np.abs(finite - median)))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] <= 0:
        return float(np.median(values))
    return float(sorted_values[np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def clean_depth_observations(
    observations_m: np.ndarray,
    quality: np.ndarray,
    fps: float,
    config: DepthConstraintConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reject local spikes, interpolate for filtering, and return a smooth signal.

    Interpolated values do not gain confidence: the returned quality stays zero
    where the source observation was invalid.
    """
    observations = np.asarray(observations_m, dtype=np.float64)
    adjusted_quality = np.clip(np.asarray(quality, dtype=np.float64), 0.0, 1.0)
    valid = np.isfinite(observations) & (observations > 0) & (adjusted_quality > 0)
    if int(valid.sum()) < config.min_valid_observations:
        raise ValueError(
            f"Only {int(valid.sum())} valid depth observations; "
            f"need at least {config.min_valid_observations}"
        )

    indices = np.arange(len(observations), dtype=np.float64)
    filled = np.interp(indices, indices[valid], observations[valid])
    half_window = max(1, _odd_window(len(filled), fps, config.hampel_window_seconds) // 2)
    outlier = np.zeros(len(filled), dtype=bool)
    cleaned = filled.copy()
    for index in np.flatnonzero(valid):
        start = max(0, index - half_window)
        end = min(len(filled), index + half_window + 1)
        local = filled[start:end]
        median = float(np.median(local))
        scale = max(_robust_mad(local), 0.01 * max(abs(median), 1.0), 1e-4)
        if abs(filled[index] - median) > config.hampel_sigma * scale:
            outlier[index] = True
            cleaned[index] = median
            adjusted_quality[index] *= 0.05

    smooth_window = _odd_window(len(cleaned), fps, config.smooth_window_seconds)
    if smooth_window >= 5:
        cleaned = savgol_filter(cleaned, smooth_window, polyorder=2, mode="interp")
    elif smooth_window >= 3:
        cleaned = savgol_filter(cleaned, smooth_window, polyorder=1, mode="interp")
    return cleaned, adjusted_quality, outlier


def robust_affine_alignment(
    observed_depth_m: np.ndarray,
    reference_camera_depth_m: np.ndarray,
    quality: np.ndarray,
) -> tuple[float, float, float]:
    """Fit positive scale and offset with Huber iteratively reweighted LS."""
    x = np.asarray(observed_depth_m, dtype=np.float64)
    y = np.asarray(reference_camera_depth_m, dtype=np.float64)
    base_weights = np.clip(np.asarray(quality, dtype=np.float64), 0.0, 1.0)
    valid = np.isfinite(x) & np.isfinite(y) & (base_weights > 0)
    x = x[valid]
    y = y[valid]
    base_weights = base_weights[valid]
    if x.size < 3:
        raise ValueError("At least three observations are required for scale/offset alignment")

    design = np.column_stack([x, np.ones_like(x)])
    weights = np.maximum(base_weights, 1e-6)
    coefficients = np.array([1.0, _weighted_median(y - x, weights)], dtype=np.float64)
    for _ in range(20):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = y * np.sqrt(weights)
        candidate = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
        residual = y - design @ candidate
        scale = max(_robust_mad(residual), 1e-4)
        huber_limit = 1.345 * scale
        huber = np.ones_like(residual)
        large = np.abs(residual) > huber_limit
        huber[large] = huber_limit / np.maximum(np.abs(residual[large]), 1e-9)
        new_weights = np.maximum(base_weights * huber, 1e-6)
        if np.max(np.abs(candidate - coefficients)) < 1e-8:
            coefficients = candidate
            weights = new_weights
            break
        coefficients = candidate
        weights = new_weights

    scale_value = float(coefficients[0])
    if not np.isfinite(scale_value) or scale_value <= 0.01 or scale_value > 20.0:
        x_iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
        y_iqr = float(np.percentile(y, 75) - np.percentile(y, 25))
        scale_value = y_iqr / x_iqr if x_iqr > 1e-6 and y_iqr > 0 else 1.0
        scale_value = float(np.clip(scale_value, 0.01, 20.0))
    offset_value = _weighted_median(y - scale_value * x, base_weights)
    residual_scale = max(_robust_mad(y - (scale_value * x + offset_value)), 1e-4)
    return scale_value, offset_value, residual_scale


def compute_activity_scores(
    baseline_camera_depth_m: np.ndarray,
    target_camera_depth_m: np.ndarray,
    root_height_m: np.ndarray,
    fps: float,
    config: DepthConstraintConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return continuous jump, travel, and short-drift scores.

    Depth change defines forward/back travel.  Jump score is high only when
    vertical body motion occurs while depth remains stable.
    """
    baseline = np.asarray(baseline_camera_depth_m, dtype=np.float64)
    target = np.asarray(target_camera_depth_m, dtype=np.float64)
    height = np.asarray(root_height_m, dtype=np.float64)
    depth_velocity = np.gradient(target) * fps
    baseline_velocity = np.gradient(baseline) * fps
    vertical_velocity = np.gradient(height) * fps

    depth_stability = np.exp(-np.square(np.abs(depth_velocity) / max(config.travel_depth_speed_mps, 1e-6)))
    travel_score = np.clip(1.0 - depth_stability, 0.0, 1.0)
    activity_window = _odd_window(len(height), fps, config.activity_window_seconds)
    if activity_window >= 3:
        height_baseline = median_filter(height, size=activity_window, mode="nearest")
    else:
        height_baseline = np.full_like(height, np.median(height))
    elevation = np.maximum(height - height_baseline, 0.0)
    vertical_activity = np.maximum(
        np.abs(vertical_velocity) / max(config.jump_vertical_speed_mps, 1e-6),
        elevation / max(config.jump_height_m, 1e-6),
    )
    jump_score = np.clip(vertical_activity, 0.0, 1.0) * depth_stability

    velocity_mismatch = np.abs(baseline_velocity - depth_velocity)
    velocity_center = float(np.median(velocity_mismatch))
    velocity_scale = max(_robust_mad(velocity_mismatch), config.drift_velocity_floor_mps)
    velocity_score = np.clip((velocity_mismatch - velocity_center) / (3.0 * velocity_scale), 0.0, 1.0)

    residual = target - baseline
    residual_window = _odd_window(len(residual), fps, config.activity_window_seconds)
    if residual_window >= 5:
        residual_low = savgol_filter(residual, residual_window, polyorder=2, mode="interp")
    elif residual_window >= 3:
        residual_low = savgol_filter(residual, residual_window, polyorder=1, mode="interp")
    else:
        residual_low = np.full_like(residual, np.median(residual))
    transient = np.abs(residual - residual_low)
    transient_center = float(np.median(transient))
    transient_scale = max(_robust_mad(transient), config.drift_residual_floor_m)
    transient_score = np.clip((transient - transient_center) / (3.0 * transient_scale), 0.0, 1.0)
    drift_score = np.maximum(velocity_score, transient_score)
    return jump_score, travel_score, drift_score


def _difference_matrix(length: int, order: int):
    if order == 2:
        return diags(
            [np.ones(length - 2), -2.0 * np.ones(length - 2), np.ones(length - 2)],
            [0, 1, 2],
            shape=(length - 2, length),
            format="csr",
        )
    if order == 3:
        return diags(
            [
                -np.ones(length - 3),
                3.0 * np.ones(length - 3),
                -3.0 * np.ones(length - 3),
                np.ones(length - 3),
            ],
            [0, 1, 2, 3],
            shape=(length - 3, length),
            format="csr",
        )
    raise ValueError(f"Unsupported difference order: {order}")


def _slew_limit(values: np.ndarray, maximum_step: float) -> np.ndarray:
    limited = np.asarray(values, dtype=np.float64).copy()
    if maximum_step <= 0 or len(limited) < 2:
        return limited
    for index in range(1, len(limited)):
        limited[index] = np.clip(limited[index], limited[index - 1] - maximum_step, limited[index - 1] + maximum_step)
    for index in range(len(limited) - 2, -1, -1):
        limited[index] = np.clip(limited[index], limited[index + 1] - maximum_step, limited[index + 1] + maximum_step)
    return limited


def _target_rate_limit(values: np.ndarray, target: np.ndarray, maximum_excess_step: float) -> np.ndarray:
    """Limit output motion to the measured depth trend plus a small tolerance."""
    limited = np.asarray(values, dtype=np.float64).copy()
    reference = np.asarray(target, dtype=np.float64)
    if maximum_excess_step <= 0 or len(limited) < 2:
        return limited
    reference_steps = np.diff(reference)
    for index in range(1, len(limited)):
        expected = limited[index - 1] + reference_steps[index - 1]
        limited[index] = np.clip(limited[index], expected - maximum_excess_step, expected + maximum_excess_step)
    for index in range(len(limited) - 2, -1, -1):
        expected = limited[index + 1] - reference_steps[index]
        limited[index] = np.clip(limited[index], expected - maximum_excess_step, expected + maximum_excess_step)
    return limited


def constrain_camera_depth(
    baseline_camera_depth_m: np.ndarray,
    observed_depth_m: np.ndarray,
    observation_quality: np.ndarray,
    root_height_m: np.ndarray,
    fps: float,
    config: DepthConstraintConfig | None = None,
    foot_stationary_confidence: np.ndarray | None = None,
) -> DepthConstraintResult:
    """Solve the full-sequence depth trajectory with robust temporal priors."""
    config = config or DepthConstraintConfig()
    baseline = np.asarray(baseline_camera_depth_m, dtype=np.float64)
    observations = np.asarray(observed_depth_m, dtype=np.float64)
    quality = np.asarray(observation_quality, dtype=np.float64)
    height = np.asarray(root_height_m, dtype=np.float64)
    if not (baseline.shape == observations.shape == quality.shape == height.shape):
        raise ValueError("Baseline, observations, quality and root height must share one shape")
    if baseline.ndim != 1 or len(baseline) < 4:
        raise ValueError("Depth constraint requires a one-dimensional sequence of at least four frames")
    if fps <= 0:
        raise ValueError("FPS must be positive")
    if foot_stationary_confidence is None:
        contact_confidence = np.zeros_like(baseline)
    else:
        contact_confidence = np.asarray(foot_stationary_confidence, dtype=np.float64)
        if contact_confidence.shape != baseline.shape:
            raise ValueError("foot_stationary_confidence must share the sequence shape")
        contact_confidence = np.nan_to_num(contact_confidence, nan=0.0, posinf=1.0, neginf=0.0)
        contact_confidence = np.clip(contact_confidence, 0.0, 1.0)

    cleaned, adjusted_quality, outlier = clean_depth_observations(observations, quality, fps, config)
    affine_scale, affine_offset, affine_residual = robust_affine_alignment(cleaned, baseline, adjusted_quality)
    target = affine_scale * cleaned + affine_offset
    jump_score, travel_score, drift_score = compute_activity_scores(baseline, target, height, fps, config)
    # Contact is a continuous prior, not a discrete AIRBORNE state. Depth still
    # decides forward/back travel; low contact merely increases depth support
    # and high contact strengthens the existing court/GVHMR trajectory prior.
    airborne_support = 1.0 - contact_confidence
    jump_score = jump_score * (0.35 + 0.65 * airborne_support)

    depth_weights = (
        config.depth_weight
        * adjusted_quality
        * (1.0 + config.drift_depth_gain * drift_score)
        * (1.0 + 0.35 * airborne_support)
    )
    baseline_weights = (
        config.gvhmr_weight
        * (1.0 - 0.85 * drift_score)
        * (1.0 + 1.25 * contact_confidence)
        + 0.05
    )
    system = diags(depth_weights + baseline_weights + 1e-8, format="csr")
    rhs = depth_weights * target + baseline_weights * baseline

    if len(baseline) >= 3:
        second = _difference_matrix(len(baseline), 2)
        travel_mid = travel_score[1:-1]
        acceleration_weights = (
            config.acceleration_weight_min
            + (config.acceleration_weight - config.acceleration_weight_min) * (1.0 - 0.75 * travel_mid)
        )
        system = system + second.T @ diags(acceleration_weights, format="csr") @ second
    if len(baseline) >= 4:
        third = _difference_matrix(len(baseline), 3)
        travel_mid = np.maximum.reduce([travel_score[:-3], travel_score[1:-2], travel_score[2:-1], travel_score[3:]])
        jerk_weights = config.jerk_weight_min + (config.jerk_weight - config.jerk_weight_min) * (1.0 - 0.75 * travel_mid)
        system = system + third.T @ diags(jerk_weights, format="csr") @ third

    solved = np.asarray(spsolve(system.tocsr(), rhs), dtype=np.float64)
    correction = np.clip(solved - baseline, -config.max_correction_m, config.max_correction_m)
    correction = _slew_limit(correction, config.max_correction_speed_mps / fps)
    corrected = _target_rate_limit(
        baseline + correction,
        target,
        config.max_unexplained_depth_speed_mps / fps,
    )
    correction = np.clip(corrected - baseline, -config.max_correction_m, config.max_correction_m)
    corrected = baseline + correction
    return DepthConstraintResult(
        cleaned_observation_m=cleaned,
        aligned_target_camera_depth_m=target,
        corrected_camera_depth_m=corrected,
        applied_camera_depth_delta_m=correction,
        observation_quality=adjusted_quality,
        observation_outlier=outlier,
        jump_score=jump_score,
        travel_score=travel_score,
        drift_score=drift_score,
        affine_scale=affine_scale,
        affine_offset_m=affine_offset,
        affine_residual_scale_m=affine_residual,
    )


def camera_depth_delta_to_court_xy(delta_camera_depth_m: np.ndarray, court_to_camera_rotation: np.ndarray) -> np.ndarray:
    """Convert camera-Z correction to court XY while preserving court Z."""
    delta = np.asarray(delta_camera_depth_m, dtype=np.float64)
    rotation = np.asarray(court_to_camera_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("court_to_camera_rotation must be 3x3")
    camera_z_gradient_xy = rotation[2, :2]
    denominator = float(camera_z_gradient_xy @ camera_z_gradient_xy)
    if denominator < 1e-8:
        raise ValueError("Camera depth has no usable projection on the court plane")
    return delta[:, None] * camera_z_gradient_xy[None, :] / denominator


def clip_court_xy_correction(
    baseline_root_xy: np.ndarray,
    correction_xy: np.ndarray,
    court_width_m: float,
    court_length_m: float,
    margin_m: float = 0.10,
) -> np.ndarray:
    """Keep corrected roots inside the court plus a small configurable margin."""
    baseline_xy = np.asarray(baseline_root_xy, dtype=np.float64)
    correction = np.asarray(correction_xy, dtype=np.float64)
    if baseline_xy.shape != correction.shape or baseline_xy.ndim != 2 or baseline_xy.shape[1] != 2:
        raise ValueError("baseline_root_xy and correction_xy must both be Nx2")
    limits = np.array([court_width_m / 2.0 + margin_m, court_length_m / 2.0 + margin_m])
    corrected = np.clip(baseline_xy + correction, -limits, limits)
    return corrected - baseline_xy


def trajectory_metrics(
    baseline_root_court: np.ndarray,
    corrected_root_court: np.ndarray,
    baseline_camera_depth_m: np.ndarray,
    corrected_camera_depth_m: np.ndarray,
    target_camera_depth_m: np.ndarray,
    fps: float,
) -> dict[str, object]:
    """Compute compact before/after drift and motion-preservation metrics."""
    baseline_root = np.asarray(baseline_root_court, dtype=np.float64)
    corrected_root = np.asarray(corrected_root_court, dtype=np.float64)

    def jerk_stats(points: np.ndarray) -> dict[str, float]:
        values = np.linalg.norm(np.diff(points, n=3, axis=0) * fps**3, axis=1)
        return {
            "mean_mps3": float(np.mean(values)),
            "median_mps3": float(np.median(values)),
            "p95_mps3": float(np.percentile(values, 95)),
            "max_mps3": float(np.max(values)),
        }

    def depth_stats(depth: np.ndarray) -> dict[str, float]:
        residual = depth - target_camera_depth_m
        frame_delta = np.abs(np.diff(depth))
        return {
            "rmse_to_depth_target_m": float(np.sqrt(np.mean(residual**2))),
            "median_abs_error_to_depth_target_m": float(np.median(np.abs(residual))),
            "p95_one_frame_change_m": float(np.percentile(frame_delta, 95)),
            "max_one_frame_change_m": float(np.max(frame_delta)),
        }

    return {
        "baseline": {
            "temporal_jerk": jerk_stats(baseline_root),
            "depth_consistency": depth_stats(np.asarray(baseline_camera_depth_m, dtype=np.float64)),
        },
        "corrected": {
            "temporal_jerk": jerk_stats(corrected_root),
            "depth_consistency": depth_stats(np.asarray(corrected_camera_depth_m, dtype=np.float64)),
        },
        "vertical_motion_preservation": {
            "max_abs_root_z_change_m": float(np.max(np.abs(corrected_root[:, 2] - baseline_root[:, 2]))),
            "exactly_preserved": bool(np.array_equal(corrected_root[:, 2], baseline_root[:, 2])),
        },
    }
