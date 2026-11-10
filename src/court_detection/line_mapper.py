"""Ground-line-first badminton court calibration.

Unlike :mod:`pole_mapper`, this module does not require the net posts, the net,
or any outer court corner to be visible.  It matches two projective families
of detected floor lines to the metric badminton-court template, re-estimates
the best homography from snapped line intersections with RANSAC, and scores
the result against every visible template line.

The returned object intentionally uses ``PoleMappedCourt`` for compatibility
with the existing 3-D pipeline.  When physical posts are absent,
``pole_feet`` are the projected metric net endpoints on the floor and the
diagnostics explicitly mark them as synthetic geometry rather than detections.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from .pole_mapper import PoleMappedCourt


COURT_X_LINES_M = (-3.05, -2.59, 0.0, 2.59, 3.05)
COURT_Y_LINES_M = (-6.70, -5.94, -1.98, 1.98, 5.94, 6.70)
COURT_WORLD_CORNERS = np.asarray(
    [[-3.05, -6.70], [3.05, -6.70], [3.05, 6.70], [-3.05, 6.70]],
    dtype=np.float64,
)

# The most useful visible rectangles.  Interior pairs let the solver recover
# a full court even when every outer corner is outside the image.
COURT_X_PAIR_PRIORS = (
    (-3.05, 3.05),
    (-2.59, 2.59),
    (-3.05, 0.0),
    (0.0, 3.05),
    (-3.05, -2.59),
    (2.59, 3.05),
)
COURT_Y_PAIR_PRIORS = (
    (-6.70, 6.70),
    (-5.94, 5.94),
    (-1.98, 1.98),
    (-6.70, -1.98),
    (1.98, 6.70),
    (-5.94, -1.98),
    (1.98, 5.94),
)


@dataclass(frozen=True)
class LineMappingConfig:
    max_lines_per_family: int = 8
    max_pairs_per_family: int = 10
    max_full_score_candidates: int = 192
    candidates_per_metric_prior: int = 2
    minimum_pair_separation_px: float = 24.0
    maximum_family_angle_difference_deg: float = 34.0
    minimum_visible_template_lines: int = 4
    minimum_observed_inlier_lines: int = 5
    minimum_line_coverage: float = 0.30
    minimum_confidence: float = 0.40
    line_distance_scale_px: float = 9.0
    player_margin_ratio: float = 0.075
    ransac_reprojection_px: float = 7.0
    enable_distance_refinement: bool = True


def _line_equation(line: Any) -> np.ndarray:
    equation = np.asarray(
        [line.y1 - line.y2, line.x2 - line.x1, line.x1 * line.y2 - line.x2 * line.y1],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(equation[:2]))
    return equation / max(norm, 1e-9)


def _intersection(first: Any, second: Any) -> np.ndarray | None:
    point = np.cross(_line_equation(first), _line_equation(second))
    if abs(float(point[2])) < 1e-8:
        return None
    point = point[:2] / point[2]
    return point if np.isfinite(point).all() else None


def _angle_distance(first: float, second: float) -> float:
    distance = abs(float(first) - float(second)) % 180.0
    return min(distance, 180.0 - distance)


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float64)]
    projected = (np.asarray(homography, dtype=np.float64) @ homogeneous.T).T
    denominator = projected[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[:, :2] / denominator


def _unproject(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    try:
        inverse = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    except np.linalg.LinAlgError:
        return np.full((len(points), 2), np.nan, dtype=np.float64)
    return _project(inverse, points)


def _pair_candidates(lines: Sequence[Any], config: LineMappingConfig) -> list[tuple[float, Any, Any]]:
    ranked = sorted(
        lines,
        key=lambda line: float(line.score) * math.sqrt(max(float(line.length), 1.0)),
        reverse=True,
    )[: config.max_lines_per_family]
    candidates: list[tuple[float, Any, Any]] = []
    for first, second in itertools.combinations(ranked, 2):
        if _angle_distance(first.angle_deg, second.angle_deg) > config.maximum_family_angle_difference_deg:
            continue
        first_midpoint = np.asarray(first.midpoint, dtype=np.float64)
        second_midpoint = np.asarray(second.midpoint, dtype=np.float64)
        separation = float(np.linalg.norm(first_midpoint - second_midpoint))
        if separation < config.minimum_pair_separation_px:
            continue
        angle_score = math.exp(
            -_angle_distance(first.angle_deg, second.angle_deg)
            / max(1.0, config.maximum_family_angle_difference_deg)
        )
        score = (
            float(first.score + second.score)
            + 0.35 * min(1.0, separation / 180.0)
            + 0.30 * angle_score
        )
        candidates.append((score, first, second))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[: config.max_pairs_per_family]


def _line_mask(lines: Sequence[Any], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        thickness = 5 if float(line.score) >= 0.45 else 3
        cv2.line(
            mask,
            (int(line.x1), int(line.y1)),
            (int(line.x2), int(line.y2)),
            255,
            thickness,
            cv2.LINE_AA,
        )
    return mask


def _bilinear_values(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return np.empty(0, dtype=np.float32)
    x_map = points[:, 0].reshape(1, -1)
    y_map = points[:, 1].reshape(1, -1)
    return cv2.remap(
        image.astype(np.float32),
        x_map,
        y_map,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(max(image.shape)),
    ).reshape(-1)


def _template_segments() -> list[tuple[str, np.ndarray]]:
    segments: list[tuple[str, np.ndarray]] = []
    for x in COURT_X_LINES_M:
        if abs(x) < 1e-9:
            segments.append(("center_far", np.asarray([[x, -5.94], [x, -1.98]], dtype=np.float64)))
            segments.append(("center_near", np.asarray([[x, 1.98], [x, 5.94]], dtype=np.float64)))
        else:
            segments.append((f"x_{x:.2f}", np.asarray([[x, -6.70], [x, 6.70]], dtype=np.float64)))
    for y in COURT_Y_LINES_M:
        segments.append((f"y_{y:.2f}", np.asarray([[-3.05, y], [3.05, y]], dtype=np.float64)))
    return segments


def _sample_template(homography: np.ndarray, width: int, height: int) -> tuple[np.ndarray, int]:
    visible_samples: list[np.ndarray] = []
    visible_lines = 0
    margin = 4.0
    for _, endpoints in _template_segments():
        ratios = np.linspace(0.0, 1.0, 33, dtype=np.float64)[:, None]
        world = endpoints[0] + ratios * (endpoints[1] - endpoints[0])
        projected = _project(homography, world)
        visible = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= -margin)
            & (projected[:, 0] <= width - 1 + margin)
            & (projected[:, 1] >= -margin)
            & (projected[:, 1] <= height - 1 + margin)
        )
        if int(np.count_nonzero(visible)) >= 5:
            visible_lines += 1
            visible_samples.append(projected[visible])
    if not visible_samples:
        return np.empty((0, 2), dtype=np.float64), 0
    return np.concatenate(visible_samples, axis=0), visible_lines


def _observed_world_support(homography: np.ndarray, lines: Sequence[Any]) -> tuple[float, int]:
    distances: list[float] = []
    inliers = 0
    for line in lines:
        ratios = np.linspace(0.0, 1.0, 7, dtype=np.float64)
        image_points = np.column_stack(
            [
                float(line.x1) + ratios * float(line.x2 - line.x1),
                float(line.y1) + ratios * float(line.y2 - line.y1),
            ]
        )
        world = _unproject(homography, image_points)
        finite = np.isfinite(world).all(axis=1)
        if not finite.any():
            continue
        world = world[finite]
        x_distance = np.min(
            np.abs(world[:, 0:1] - np.asarray(COURT_X_LINES_M, dtype=np.float64)[None, :]),
            axis=1,
        )
        y_distance = np.min(
            np.abs(world[:, 1:2] - np.asarray(COURT_Y_LINES_M, dtype=np.float64)[None, :]),
            axis=1,
        )
        distance = np.minimum(x_distance, y_distance)
        median_distance = float(np.median(distance))
        distances.append(median_distance)
        if median_distance <= 0.14:
            inliers += 1
    if not distances:
        return 0.0, 0
    robust_distance = float(np.median(np.asarray(distances, dtype=np.float64)))
    support = math.exp(-robust_distance / 0.18)
    return float(support), int(inliers)


def _player_support(
    image_corners: np.ndarray,
    player_points: np.ndarray,
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[float, list[float]]:
    polygon = np.asarray(image_corners, dtype=np.float32)
    margin = margin_ratio * float(math.hypot(width, height))
    signed = [
        float(cv2.pointPolygonTest(polygon, tuple(map(float, point)), True))
        for point in np.asarray(player_points, dtype=np.float64)
    ]
    support = float(np.mean([1.0 if distance >= 0.0 else math.exp(distance / max(margin, 1.0)) for distance in signed]))
    return support, signed


def score_court_homography(
    homography: np.ndarray,
    lines: Sequence[Any],
    player_points: Sequence[Sequence[float]],
    width: int,
    height: int,
    config: LineMappingConfig | None = None,
    distance_field: np.ndarray | None = None,
) -> dict[str, Any]:
    """Score a metric court homography against all observed floor lines."""

    config = config or LineMappingConfig()
    homography = np.asarray(homography, dtype=np.float64)
    image_corners = _project(homography, COURT_WORLD_CORNERS)
    if image_corners.shape != (4, 2) or not np.isfinite(image_corners).all():
        return {"valid": False, "confidence": 0.0, "reason": "non_finite_corners"}
    hull = cv2.convexHull(image_corners.astype(np.float32)).reshape(-1, 2)
    if len(hull) != 4:
        return {"valid": False, "confidence": 0.0, "reason": "non_convex_court"}
    area = abs(float(cv2.contourArea(image_corners.astype(np.float32))))
    area_ratio = area / max(1.0, float(width * height))
    if area_ratio < 0.012 or area_ratio > 2.4 or float(np.max(np.abs(image_corners))) > 6.0 * max(width, height):
        return {"valid": False, "confidence": 0.0, "reason": "implausible_projected_area"}

    if distance_field is None:
        observed_mask = _line_mask(lines, width, height)
        distance_field = cv2.distanceTransform(255 - observed_mask, cv2.DIST_L2, 3)
    template_points, visible_lines = _sample_template(homography, width, height)
    if visible_lines < config.minimum_visible_template_lines or len(template_points) < 30:
        return {"valid": False, "confidence": 0.0, "reason": "too_few_visible_template_lines"}
    distances_px = _bilinear_values(distance_field, template_points)
    clipped = np.minimum(distances_px, 4.0 * config.line_distance_scale_px)
    line_coverage = float(np.mean(np.exp(-clipped / config.line_distance_scale_px)))
    median_distance_px = float(np.median(distances_px))
    p95_distance_px = float(np.percentile(distances_px, 95))
    observed_support, observed_inlier_lines = _observed_world_support(homography, lines)

    players = np.asarray(player_points, dtype=np.float64).reshape(-1, 2)
    if players.shape == (2, 2) and np.isfinite(players).all():
        player_support, signed_player_distances = _player_support(
            image_corners,
            players,
            width,
            height,
            config.player_margin_ratio,
        )
    else:
        player_support = 0.5
        signed_player_distances = []

    # Coverage is the primary cue.  Reverse observed-line support prevents a
    # small subset of unrelated white lines from explaining the template, and
    # players only resolve adjacent-court ambiguity.
    confidence = float(
        np.clip(
            0.58 * line_coverage
            + 0.27 * observed_support
            + 0.15 * player_support,
            0.0,
            1.0,
        )
    )
    valid = bool(
        line_coverage >= config.minimum_line_coverage
        and confidence >= config.minimum_confidence
        and observed_inlier_lines >= config.minimum_observed_inlier_lines
        and player_support >= 0.48
    )
    return {
        "valid": valid,
        "confidence": confidence,
        "line_coverage": line_coverage,
        "template_median_distance_px": median_distance_px,
        "template_p95_distance_px": p95_distance_px,
        "visible_template_lines": int(visible_lines),
        "observed_line_support": float(observed_support),
        "observed_inlier_lines": int(observed_inlier_lines),
        "player_support": float(player_support),
        "player_signed_distances_px": signed_player_distances,
        "projected_area_ratio": float(area_ratio),
        "image_corners": image_corners,
        "reason": "accepted" if valid else "below_confidence_gate",
    }


def _quick_candidate_score(
    homography: np.ndarray,
    player_points: np.ndarray,
    width: int,
    height: int,
    anchor_score: float,
    config: LineMappingConfig,
) -> tuple[float, np.ndarray] | None:
    corners = _project(homography, COURT_WORLD_CORNERS)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)
    if len(hull) != 4:
        return None
    area_ratio = abs(float(cv2.contourArea(corners.astype(np.float32)))) / max(1.0, width * height)
    if not 0.012 <= area_ratio <= 2.4 or float(np.max(np.abs(corners))) > 6.0 * max(width, height):
        return None
    player_score, _ = _player_support(
        corners,
        player_points,
        width,
        height,
        config.player_margin_ratio,
    )
    if player_score < 0.35:
        return None
    center = _project(homography, np.asarray([[0.0, 0.0]], dtype=np.float64))[0]
    center_score = math.exp(
        -float(np.linalg.norm(center - np.mean(player_points, axis=0)))
        / max(1.0, 0.8 * width)
    )
    # This stage only removes geometrically impossible candidates.  The full
    # all-line distance-field score is deliberately deferred until we have
    # preserved representatives for every metric-line prior.  Globally
    # ranking here can otherwise discard the true outer court in favour of a
    # visually similar service-box rectangle.
    score = float(anchor_score + 2.0 * player_score + 0.25 * center_score)
    return score, corners


def _candidate_homographies(
    first_family: Sequence[Any],
    second_family: Sequence[Any],
    player_points: np.ndarray,
    width: int,
    height: int,
    config: LineMappingConfig,
) -> tuple[list[tuple[float, np.ndarray]], int]:
    first_pairs = _pair_candidates(first_family, config)
    second_pairs = _pair_candidates(second_family, config)
    hypotheses_by_prior: dict[
        tuple[bool, tuple[float, float], tuple[float, float]],
        list[tuple[float, np.ndarray]],
    ] = {}
    tested = 0
    for families_swapped in (False, True):
        x_pairs = second_pairs if families_swapped else first_pairs
        y_pairs = first_pairs if families_swapped else second_pairs
        for x_pair, y_pair in itertools.product(x_pairs, y_pairs):
            anchor_score = float(x_pair[0] + y_pair[0])
            observed_x = (x_pair[1], x_pair[2])
            observed_y = (y_pair[1], y_pair[2])
            for x_world in COURT_X_PAIR_PRIORS:
                for y_world in COURT_Y_PAIR_PRIORS:
                    world = np.asarray(
                        [
                            [x_world[0], y_world[0]],
                            [x_world[1], y_world[0]],
                            [x_world[1], y_world[1]],
                            [x_world[0], y_world[1]],
                        ],
                        dtype=np.float32,
                    )
                    for reverse_x, reverse_y in itertools.product((False, True), repeat=2):
                        x_lines = observed_x[::-1] if reverse_x else observed_x
                        y_lines = observed_y[::-1] if reverse_y else observed_y
                        image_points = [
                            _intersection(x_lines[0], y_lines[0]),
                            _intersection(x_lines[1], y_lines[0]),
                            _intersection(x_lines[1], y_lines[1]),
                            _intersection(x_lines[0], y_lines[1]),
                        ]
                        tested += 1
                        if any(point is None for point in image_points):
                            continue
                        image = np.asarray(image_points, dtype=np.float32)
                        if not np.isfinite(image).all():
                            continue
                        homography = cv2.getPerspectiveTransform(world, image)
                        quick = _quick_candidate_score(
                            homography,
                            player_points,
                            width,
                            height,
                            anchor_score,
                            config,
                        )
                        if quick is not None:
                            prior_key = (families_swapped, x_world, y_world)
                            bucket = hypotheses_by_prior.setdefault(prior_key, [])
                            bucket.append((quick[0], homography))
                            bucket.sort(key=lambda item: item[0], reverse=True)
                            del bucket[config.candidates_per_metric_prior :]
    hypotheses = [
        hypothesis
        for bucket in hypotheses_by_prior.values()
        for hypothesis in bucket
    ]
    hypotheses.sort(key=lambda item: item[0], reverse=True)
    return hypotheses[: config.max_full_score_candidates], tested


def _canonicalize_homography(homography: np.ndarray) -> np.ndarray:
    """Choose a stable world-axis orientation for a symmetric court."""

    result = np.asarray(homography, dtype=np.float64)
    x_axis = _project(
        result,
        np.asarray([[-3.05, 0.0], [3.05, 0.0]], dtype=np.float64),
    )
    if np.isfinite(x_axis).all() and float(x_axis[1, 0]) < float(x_axis[0, 0]):
        result = result @ np.asarray(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    y_axis = _project(
        result,
        np.asarray([[0.0, -6.70], [0.0, 6.70]], dtype=np.float64),
    )
    if np.isfinite(y_axis).all() and float(y_axis[1, 1]) < float(y_axis[0, 1]):
        result = result @ np.asarray(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return result / max(abs(float(result[2, 2])), 1e-9)


def _ransac_snap_intersections(
    homography: np.ndarray,
    first_family: Sequence[Any],
    second_family: Sequence[Any],
    config: LineMappingConfig,
) -> tuple[np.ndarray, int, int]:
    world_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    inverse_points: list[np.ndarray] = []
    raw_intersections: list[np.ndarray] = []
    for first, second in itertools.product(first_family, second_family):
        point = _intersection(first, second)
        if point is None:
            continue
        raw_intersections.append(point)
    if raw_intersections:
        inverse_points = list(_unproject(homography, np.asarray(raw_intersections, dtype=np.float64)))
    for image_point, world_estimate in zip(raw_intersections, inverse_points):
        if not np.isfinite(world_estimate).all():
            continue
        x = min(COURT_X_LINES_M, key=lambda value: abs(value - float(world_estimate[0])))
        y = min(COURT_Y_LINES_M, key=lambda value: abs(value - float(world_estimate[1])))
        if abs(float(world_estimate[0]) - x) > 0.38 or abs(float(world_estimate[1]) - y) > 0.52:
            continue
        candidate_world = np.asarray([x, y], dtype=np.float64)
        if any(np.linalg.norm(candidate_world - previous) < 1e-5 for previous in world_points):
            continue
        world_points.append(candidate_world)
        image_points.append(np.asarray(image_point, dtype=np.float64))
    if len(world_points) < 4:
        return homography, len(world_points), 0
    refined, mask = cv2.findHomography(
        np.asarray(world_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reprojection_px,
        maxIters=2500,
        confidence=0.995,
    )
    if refined is None or not np.isfinite(refined).all():
        return homography, len(world_points), 0
    inliers = int(np.count_nonzero(mask)) if mask is not None else 0
    return refined / max(abs(float(refined[2, 2])), 1e-9), len(world_points), inliers


def _distance_refine(
    homography: np.ndarray,
    lines: Sequence[Any],
    player_points: np.ndarray,
    width: int,
    height: int,
    config: LineMappingConfig,
    distance_field: np.ndarray,
) -> tuple[np.ndarray, bool]:
    if not config.enable_distance_refinement:
        return homography, False
    try:
        from scipy.optimize import minimize
    except ImportError:
        return homography, False

    initial_corners = _project(homography, COURT_WORLD_CORNERS)
    if not np.isfinite(initial_corners).all():
        return homography, False
    initial = (initial_corners / np.asarray([width, height], dtype=np.float64)).reshape(-1)
    bounds = [(-2.0, 3.0), (-2.0, 3.0)] * 4

    def objective(parameters: np.ndarray) -> float:
        corners = parameters.reshape(4, 2) * np.asarray([width, height], dtype=np.float64)
        hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)
        if len(hull) != 4:
            return 20.0
        candidate = cv2.getPerspectiveTransform(
            COURT_WORLD_CORNERS.astype(np.float32),
            corners.astype(np.float32),
        )
        result = score_court_homography(
            candidate,
            lines,
            player_points,
            width,
            height,
            config,
            distance_field,
        )
        if "line_coverage" not in result:
            return 20.0
        displacement = float(np.mean((parameters - initial) ** 2))
        return float(
            1.0
            - result["line_coverage"]
            + 0.28 * (1.0 - result["observed_line_support"])
            + 0.15 * (1.0 - result["player_support"])
            + 0.08 * displacement
        )

    optimized = minimize(
        objective,
        initial,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 70, "xtol": 5e-4, "ftol": 2e-4},
    )
    if not optimized.success and not np.isfinite(float(optimized.fun)):
        return homography, False
    optimized_corners = optimized.x.reshape(4, 2) * np.asarray([width, height], dtype=np.float64)
    refined = cv2.getPerspectiveTransform(
        COURT_WORLD_CORNERS.astype(np.float32),
        optimized_corners.astype(np.float32),
    )
    before = score_court_homography(
        homography, lines, player_points, width, height, config, distance_field
    )
    after = score_court_homography(
        refined, lines, player_points, width, height, config, distance_field
    )
    if float(after.get("confidence", 0.0)) + 0.005 < float(before.get("confidence", 0.0)):
        return homography, False
    return refined, True


def select_line_guided_court(
    frame_bgr: np.ndarray,
    lines: Sequence[Any],
    player_points: Sequence[Sequence[float]],
    *,
    refine: bool = False,
    config: LineMappingConfig | None = None,
) -> PoleMappedCourt | None:
    """Recover a standard-court homography without requiring net posts."""

    config = config or LineMappingConfig()
    points = np.asarray(player_points, dtype=np.float64).reshape(-1, 2)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        return None
    height, width = frame_bgr.shape[:2]
    first_family = [line for line in lines if line.family == "cross_court_family"]
    second_family = [line for line in lines if line.family == "length_court_family"]
    if len(first_family) < 2 or len(second_family) < 2:
        return None
    hypotheses, tested_hypotheses = _candidate_homographies(
        first_family,
        second_family,
        points,
        width,
        height,
        config,
    )
    if not hypotheses:
        return None
    observed_mask = _line_mask(lines, width, height)
    distance_field = cv2.distanceTransform(255 - observed_mask, cv2.DIST_L2, 3)
    scored: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for _, hypothesis in hypotheses:
        result = score_court_homography(
            hypothesis,
            lines,
            points,
            width,
            height,
            config,
            distance_field,
        )
        if "line_coverage" in result:
            scored.append((float(result["confidence"]), hypothesis, result))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    _, homography, diagnostics = scored[0]
    homography, ransac_correspondences, ransac_inliers = _ransac_snap_intersections(
        homography,
        first_family,
        second_family,
        config,
    )
    homography, distance_refined = _distance_refine(
        homography,
        lines,
        points,
        width,
        height,
        config if refine else LineMappingConfig(**{**config.__dict__, "enable_distance_refinement": False}),
        distance_field,
    )
    homography = _canonicalize_homography(homography)
    diagnostics = score_court_homography(
        homography,
        lines,
        points,
        width,
        height,
        config,
        distance_field,
    )
    if not bool(diagnostics.get("valid", False)):
        return None

    image_corners = np.asarray(diagnostics.pop("image_corners"), dtype=np.float64)
    net_floor_points = _project(
        homography,
        np.asarray([[-3.05, 0.0], [3.05, 0.0]], dtype=np.float64),
    )
    if not np.isfinite(net_floor_points).all():
        return None
    player_direction = points[1] - points[0]
    net_direction = net_floor_points[1] - net_floor_points[0]
    denominator = float(
        net_direction[0] * player_direction[1]
        - net_direction[1] * player_direction[0]
    )
    if abs(denominator) < 1e-8:
        player_ratio = 0.5
        player_intersection = np.mean(net_floor_points, axis=0)
    else:
        delta = points[0] - net_floor_points[0]
        net_ratio = float(
            (delta[0] * player_direction[1] - delta[1] * player_direction[0])
            / denominator
        )
        player_ratio = float(
            (delta[0] * net_direction[1] - delta[1] * net_direction[0])
            / denominator
        )
        player_intersection = net_floor_points[0] + net_ratio * net_direction

    vanishing_width = np.asarray(homography[:, 0], dtype=np.float64)
    vanishing_length = np.asarray(homography[:, 1], dtype=np.float64)
    mapping_diagnostics = {
        "mapping": "ground_plane_homography_from_lines_ransac_distance_refined",
        "mapping_fallback": False,
        "mapping_source": "ground_lines_without_required_posts",
        "pole_evidence": False,
        "synthetic_net_floor_endpoints": True,
        "court_length_m": 13.4,
        "court_width_m": 6.1,
        "approximate_camera_focal_px": round(0.78 * float(width), 3),
        "camera_calibration": "planar_homography_with_assumed_intrinsics",
        "camera_pnp_image_corners": np.round(image_corners, 6).tolist(),
        "candidate_hypotheses_tested": int(tested_hypotheses),
        "candidate_hypotheses_fully_scored": int(len(scored)),
        "ransac_correspondences": int(ransac_correspondences),
        "ransac_inliers": int(ransac_inliers),
        "distance_field_refined": bool(distance_refined),
        **{
            key: value
            for key, value in diagnostics.items()
            if key not in {"valid", "reason"}
        },
    }
    return PoleMappedCourt(
        pole_ids=(-1, -2),
        pole_feet=net_floor_points,
        player_intersection=np.asarray(player_intersection, dtype=np.float64),
        player_intersection_ratio=float(player_ratio),
        net_line_id=None,
        net_points=net_floor_points.copy(),
        net_score=float(diagnostics["line_coverage"]),
        vanishing_width=vanishing_width,
        vanishing_length=vanishing_length,
        homography_world_to_image=homography,
        image_corners=image_corners,
        world_corners=COURT_WORLD_CORNERS.copy(),
        score=10.0 * float(diagnostics["confidence"]),
        diagnostics=mapping_diagnostics,
    )
