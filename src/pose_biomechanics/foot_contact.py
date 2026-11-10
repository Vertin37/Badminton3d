"""Continuous foot-contact refinement for court-space SMPL-X sequences.

The refinement applies only rigid per-frame root translations. It uses the
actual SMPL-X skinning weights to identify each sole, keeps a stationary
support foot from sliding in court XY, and aligns its lower sole surface with
the Z=0 court plane. Local articulated mesh shape is never deformed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from scipy.sparse.linalg import lsqr


@dataclass(frozen=True)
class FootContactConfig:
    skin_weight_threshold: float = 0.50
    template_sole_quantile: float = 0.18
    sole_height_quantile: float = 0.10
    min_contact_confidence: float = 0.55
    stationary_local_speed_mps: float = 1.20
    max_stationary_height_gap_m: float = 0.08
    position_prior_weight: float = 0.35
    velocity_prior_weight: float = 1.5
    acceleration_weight: float = 8.0
    foot_lock_weight: float = 80.0
    floor_prior_weight: float = 0.12
    floor_velocity_weight: float = 8.0
    floor_acceleration_weight: float = 3.0
    floor_contact_weight: float = 120.0
    max_xy_correction_m: float = 0.75
    max_z_correction_m: float = 0.60
    # This limits the *correction signal*, not the athlete.  It must be high
    # enough to cancel a short root-trajectory jump while a learned support
    # foot is planted; the final athlete speed is still bounded upstream.
    max_correction_speed_mps: float = 8.00
    max_root_xy_speed_mps: float = 8.00
    max_root_z_speed_mps: float = 6.00
    court_margin_m: float = 0.10


@dataclass(frozen=True)
class FootContactResult:
    corrected_vertices: np.ndarray
    corrected_root: np.ndarray
    correction_xyz_m: np.ndarray
    sole_points_before: np.ndarray
    sole_points_after: np.ndarray
    stationary_scores: np.ndarray
    support_foot: np.ndarray
    metrics: dict


def _difference_matrix(length: int, order: int) -> csr_matrix:
    if order == 1:
        return diags(
            [-np.ones(length - 1), np.ones(length - 1)],
            [0, 1],
            shape=(length - 1, length),
        ).tocsr()
    if order == 2:
        return diags(
            [np.ones(length - 2), -2.0 * np.ones(length - 2), np.ones(length - 2)],
            [0, 1, 2],
            shape=(length - 2, length),
        ).tocsr()
    raise ValueError(f"Unsupported difference order: {order}")


def smplx_sole_vertex_indices(
    weights: np.ndarray,
    template_vertices: np.ndarray,
    config: FootContactConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable left/right sole vertex IDs from official SMPL-X data."""
    cfg = config or FootContactConfig()
    weights = np.asarray(weights)
    template_vertices = np.asarray(template_vertices)
    if weights.shape[0] != len(template_vertices) or weights.shape[1] <= 11:
        raise ValueError("Unexpected SMPL-X weights/template shape")
    foot_joint_pairs = ((7, 10), (8, 11))
    soles: list[np.ndarray] = []
    for ankle, foot in foot_joint_pairs:
        region = (weights[:, ankle] + weights[:, foot]) >= cfg.skin_weight_threshold
        region_ids = np.flatnonzero(region)
        if len(region_ids) < 20:
            raise ValueError("SMPL-X foot skinning region is unexpectedly small")
        floor_limit = np.quantile(template_vertices[region_ids, 1], cfg.template_sole_quantile)
        sole_ids = region_ids[template_vertices[region_ids, 1] <= floor_limit]
        if len(sole_ids) < 8:
            raise ValueError("SMPL-X sole vertex set is unexpectedly small")
        soles.append(sole_ids.astype(np.int32))
    return soles[0], soles[1]


def sole_points(vertices: np.ndarray, sole_indices: tuple[np.ndarray, np.ndarray], height_quantile: float) -> np.ndarray:
    """Compute left/right sole centers with a robust lower-surface height."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError("vertices must have shape [frames, vertices, 3]")
    result = np.empty((len(vertices), 2, 3), dtype=np.float64)
    for foot, indices in enumerate(sole_indices):
        points = vertices[:, indices]
        result[:, foot, :2] = points[:, :, :2].mean(axis=1)
        result[:, foot, 2] = np.quantile(points[:, :, 2], height_quantile, axis=1)
    return result


def _stationary_scores(
    sole: np.ndarray,
    root: np.ndarray,
    confidence_lr: np.ndarray,
    fps: float,
    config: FootContactConfig,
) -> np.ndarray:
    confidence_lr = np.clip(np.asarray(confidence_lr, dtype=np.float64), 0.0, 1.0)
    if confidence_lr.shape != sole.shape[:2]:
        raise ValueError("Left/right contact confidence shape does not match the sequence")
    local = sole - root[:, None, :]
    local_step_speed = np.linalg.norm(np.diff(local, axis=0), axis=2) * fps
    local_speed = np.empty_like(confidence_lr)
    local_speed[0] = local_step_speed[0]
    local_speed[-1] = local_step_speed[-1]
    if len(sole) > 2:
        local_speed[1:-1] = 0.5 * (local_step_speed[:-1] + local_step_speed[1:])
    contact = np.clip(
        (confidence_lr - config.min_contact_confidence) / (1.0 - config.min_contact_confidence),
        0.0,
        1.0,
    )
    motion_gate = 1.0 / (1.0 + (local_speed / config.stationary_local_speed_mps) ** 4)
    # A raised swing foot may be nearly motionless for a few frames and can
    # receive a high learned static confidence.  It is not a support contact.
    # Compare only feet that passed the confidence threshold so a low-score,
    # downward reconstruction outlier cannot suppress the true support foot.
    eligible_height = np.where(contact > 0.0, sole[:, :, 2], np.inf)
    reference_height = np.min(eligible_height, axis=1)
    no_candidate = ~np.isfinite(reference_height)
    reference_height[no_candidate] = np.min(sole[no_candidate, :, 2], axis=1)
    height_gap = np.maximum(sole[:, :, 2] - reference_height[:, None], 0.0)
    height_gate = 1.0 / (1.0 + (height_gap / config.max_stationary_height_gap_m) ** 4)
    return contact * motion_gate * height_gate


def _limit_vector_correction(values: np.ndarray, max_norm: float, max_step: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(result, axis=1)
    over = norms > max_norm
    result[over] *= (max_norm / norms[over])[:, None]
    for _ in range(2):
        for index in range(1, len(result)):
            delta = result[index] - result[index - 1]
            norm = float(np.linalg.norm(delta))
            if norm > max_step:
                result[index] = result[index - 1] + delta * (max_step / norm)
        for index in range(len(result) - 2, -1, -1):
            delta = result[index] - result[index + 1]
            norm = float(np.linalg.norm(delta))
            if norm > max_step:
                result[index] = result[index + 1] + delta * (max_step / norm)
    return result


def _project_correction_and_result_steps(
    root: np.ndarray,
    correction: np.ndarray,
    max_correction_norm: float,
    max_correction_step: float,
    max_result_step: float,
) -> np.ndarray:
    """Project a rigid correction onto correction and final-speed bounds.

    Limiting only the correction derivative is insufficient: an upstream
    trajectory already at its speed limit and an opposite foot-lock update
    can add to twice that limit.  Alternating projections are safe here
    because zero correction is always feasible when the upstream trajectory
    has already passed its own speed gate.
    """

    root = np.asarray(root, dtype=np.float64)
    projected = np.asarray(correction, dtype=np.float64).copy()
    if root.shape != projected.shape or root.ndim != 2:
        raise ValueError("Root and correction must have the same NxD shape")
    for _ in range(24):
        projected = _limit_vector_correction(
            projected,
            max_correction_norm,
            max_correction_step,
        )
        result = root + projected
        for index in range(1, len(result)):
            delta = result[index] - result[index - 1]
            norm = float(np.linalg.norm(delta))
            if norm > max_result_step:
                result[index] = result[index - 1] + delta * (max_result_step / norm)
        for index in range(len(result) - 2, -1, -1):
            delta = result[index] - result[index + 1]
            norm = float(np.linalg.norm(delta))
            if norm > max_result_step:
                result[index] = result[index + 1] + delta * (max_result_step / norm)
        projected = result - root
    return projected


def _solve_xy_correction(
    sole: np.ndarray,
    scores: np.ndarray,
    fps: float,
    config: FootContactConfig,
) -> np.ndarray:
    length = len(sole)
    identity = diags(np.ones(length), 0, shape=(length, length)).tocsr()
    d1 = _difference_matrix(length, 1)
    d2 = _difference_matrix(length, 2)
    matrices = [
        identity * np.sqrt(config.position_prior_weight),
        d1 * np.sqrt(config.velocity_prior_weight),
        d2 * np.sqrt(config.acceleration_weight),
    ]
    targets = [
        np.zeros((length, 2)),
        np.zeros((length - 1, 2)),
        np.zeros((length - 2, 2)),
    ]
    for foot in range(2):
        step_score = np.minimum(scores[:-1, foot], scores[1:, foot])
        row_weight = np.sqrt(config.foot_lock_weight * step_score)
        matrices.append(diags(row_weight) @ d1)
        targets.append(-np.diff(sole[:, foot, :2], axis=0) * row_weight[:, None])
    matrix = vstack(matrices).tocsr()
    target = np.vstack(targets)
    correction = np.column_stack(
        [lsqr(matrix, target[:, axis], atol=1e-8, btol=1e-8, iter_lim=400)[0] for axis in range(2)]
    )
    return _limit_vector_correction(
        correction,
        config.max_xy_correction_m,
        config.max_correction_speed_mps / fps,
    )


def _solve_z_correction(
    sole: np.ndarray,
    scores: np.ndarray,
    fps: float,
    config: FootContactConfig,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(sole)
    identity = diags(np.ones(length), 0, shape=(length, length)).tocsr()
    d1 = _difference_matrix(length, 1)
    d2 = _difference_matrix(length, 2)
    support = np.argmax(scores, axis=1).astype(np.int8)
    both_reliable = np.min(scores, axis=1) >= 0.35
    lower_foot = np.argmin(sole[:, :, 2], axis=1).astype(np.int8)
    support[both_reliable] = lower_foot[both_reliable]
    support_score = scores[np.arange(length), support]
    support_height = sole[np.arange(length), support, 2]
    row_weight = np.sqrt(config.floor_contact_weight * support_score)
    matrix = vstack(
        [
            identity * np.sqrt(config.floor_prior_weight),
            d1 * np.sqrt(config.floor_velocity_weight),
            d2 * np.sqrt(config.floor_acceleration_weight),
            diags(row_weight) @ identity,
        ]
    ).tocsr()
    target = np.concatenate(
        [
            np.zeros(length),
            np.zeros(length - 1),
            np.zeros(length - 2),
            -support_height * row_weight,
        ]
    )
    correction = lsqr(matrix, target, atol=1e-8, btol=1e-8, iter_lim=400)[0]
    correction = np.clip(correction, -config.max_z_correction_m, config.max_z_correction_m)
    limited = _limit_vector_correction(
        np.column_stack([correction, np.zeros(length)]),
        config.max_z_correction_m,
        config.max_correction_speed_mps / fps,
    )[:, 0]
    return limited, support


def _metrics(
    sole_before: np.ndarray,
    sole_after: np.ndarray,
    scores: np.ndarray,
    support: np.ndarray,
    root_before: np.ndarray,
    root_after: np.ndarray,
    fps: float,
) -> dict:
    step_score = np.minimum(scores[:-1], scores[1:])
    stationary = step_score >= 0.35
    before_step = np.linalg.norm(np.diff(sole_before[:, :, :2], axis=0), axis=2)
    after_step = np.linalg.norm(np.diff(sole_after[:, :, :2], axis=0), axis=2)
    frame_score = scores[np.arange(len(scores)), support]
    contact = frame_score >= 0.35
    support_before = sole_before[np.arange(len(sole_before)), support, 2]
    support_after = sole_after[np.arange(len(sole_after)), support, 2]

    def stats(values: np.ndarray) -> dict:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"count": 0, "median": None, "p95": None, "max": None}
        return {
            "count": int(finite.size),
            "median": float(np.median(finite)),
            "p95": float(np.percentile(finite, 95)),
            "max": float(np.max(finite)),
        }

    return {
        "stationary_foot_xy_step_before_m": stats(before_step[stationary]),
        "stationary_foot_xy_step_after_m": stats(after_step[stationary]),
        "support_sole_abs_height_before_m": stats(np.abs(support_before[contact])),
        "support_sole_abs_height_after_m": stats(np.abs(support_after[contact])),
        "support_contact_frames": int(contact.sum()),
        "root_xy_speed_before_mps": stats(
            np.linalg.norm(np.diff(root_before[:, :2], axis=0), axis=1) * fps
        ),
        "root_xy_speed_after_mps": stats(
            np.linalg.norm(np.diff(root_after[:, :2], axis=0), axis=1) * fps
        ),
        "root_abs_z_speed_before_mps": stats(np.abs(np.diff(root_before[:, 2])) * fps),
        "root_abs_z_speed_after_mps": stats(np.abs(np.diff(root_after[:, 2])) * fps),
    }


def refine_foot_contacts(
    vertices: np.ndarray,
    root: np.ndarray,
    confidence_lr: np.ndarray,
    sole_indices: tuple[np.ndarray, np.ndarray],
    fps: float,
    config: FootContactConfig | None = None,
    court_half_extents_m: np.ndarray | None = None,
) -> FootContactResult:
    cfg = config or FootContactConfig()
    vertices = np.asarray(vertices, dtype=np.float64)
    root = np.asarray(root, dtype=np.float64)
    if len(vertices) < 3 or root.shape != (len(vertices), 3):
        raise ValueError("Foot refinement requires matching multi-frame vertices and roots")
    if fps <= 0:
        raise ValueError("FPS must be positive")
    sole_before = sole_points(vertices, sole_indices, cfg.sole_height_quantile)
    scores = _stationary_scores(sole_before, root, confidence_lr, fps, cfg)
    correction_xy = _solve_xy_correction(sole_before, scores, fps, cfg)
    correction_z, support = _solve_z_correction(sole_before, scores, fps, cfg)
    correction_xy = _project_correction_and_result_steps(
        root[:, :2],
        correction_xy,
        cfg.max_xy_correction_m,
        cfg.max_correction_speed_mps / fps,
        cfg.max_root_xy_speed_mps / fps,
    )
    correction_z = _project_correction_and_result_steps(
        root[:, 2:3],
        correction_z[:, None],
        cfg.max_z_correction_m,
        cfg.max_correction_speed_mps / fps,
        cfg.max_root_z_speed_mps / fps,
    )[:, 0]
    if court_half_extents_m is not None:
        limits = np.asarray(court_half_extents_m, dtype=np.float64).reshape(2)
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("Court half extents must contain two positive finite values")
        bounded_xy = np.clip(root[:, :2] + correction_xy, -limits, limits)
        correction_xy = bounded_xy - root[:, :2]
    correction = np.column_stack([correction_xy, correction_z])
    corrected_vertices = vertices + correction[:, None, :]
    corrected_root = root + correction
    sole_after = sole_before + correction[:, None, :]
    return FootContactResult(
        corrected_vertices=corrected_vertices.astype(np.float32),
        corrected_root=corrected_root.astype(np.float32),
        correction_xyz_m=correction.astype(np.float32),
        sole_points_before=sole_before.astype(np.float32),
        sole_points_after=sole_after.astype(np.float32),
        stationary_scores=scores.astype(np.float32),
        support_foot=support,
        metrics=_metrics(sole_before, sole_after, scores, support, root, corrected_root, fps),
    )
