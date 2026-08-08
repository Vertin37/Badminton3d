"""Pole-first badminton-court selection and image-space mapping.

This module implements the geometry used by the original calibration scripts,
but replaces manual corner clicks with a player-guided pole pair:

1. find visible blue/dark post bases;
2. pair adjacent bases whose connecting segment intersects the two-player foot
   segment;
3. select the net edge between that pair;
4. use the net direction and the court-line direction as vanishing directions;
5. build a projective mapping from the standard 13.4 m x 6.1 m court to the
   image.

The mapping is image-space only.  It does not claim metric camera calibration
when the video does not provide enough reliable vanishing-point evidence.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np


def _line_coeff(line: Any) -> np.ndarray:
    equation = np.array(
        [line.y1 - line.y2, line.x2 - line.x1, line.x1 * line.y2 - line.x2 * line.y1],
        dtype=float,
    )
    norm = float(np.linalg.norm(equation[:2]))
    return equation / max(norm, 1e-8)


def _line_y_at_x(line: Any, x: float) -> float | None:
    if abs(line.x2 - line.x1) < 1e-7:
        return None
    return float(line.y1 + (x - line.x1) * (line.y2 - line.y1) / (line.x2 - line.x1))


def _segment_intersection(
    first_start: Sequence[float],
    first_end: Sequence[float],
    second_start: Sequence[float],
    second_end: Sequence[float],
) -> tuple[np.ndarray, float, float] | None:
    first_start = np.asarray(first_start, dtype=float)
    first_end = np.asarray(first_end, dtype=float)
    second_start = np.asarray(second_start, dtype=float)
    second_end = np.asarray(second_end, dtype=float)
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    cross = float(first_direction[0] * second_direction[1] - first_direction[1] * second_direction[0])
    if abs(cross) < 1e-8:
        return None
    delta = second_start - first_start
    first_ratio = float((delta[0] * second_direction[1] - delta[1] * second_direction[0]) / cross)
    second_ratio = float((delta[0] * first_direction[1] - delta[1] * first_direction[0]) / cross)
    point = first_start + first_ratio * first_direction
    return point, first_ratio, second_ratio


def _point_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    direction = np.asarray(end, dtype=float) - start
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-8:
        return float(np.linalg.norm(point - start))
    ratio = np.clip(float(np.dot(point - start, direction) / denominator), 0.0, 1.0)
    return float(np.linalg.norm(point - (start + ratio * direction)))


@dataclass
class PoleCandidate:
    pole_id: int
    foot: np.ndarray
    area: float
    support: float
    bbox: tuple[int, int, int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pole_id": self.pole_id,
            "foot": np.round(self.foot, 3).tolist(),
            "component_area": round(float(self.area), 3),
            "support": round(float(self.support), 5),
            "bbox": list(self.bbox),
        }


@dataclass
class PoleMappedCourt:
    pole_ids: tuple[int, int]
    pole_feet: np.ndarray
    player_intersection: np.ndarray
    player_intersection_ratio: float
    net_line_id: int | None
    net_points: np.ndarray
    net_score: float
    vanishing_width: np.ndarray
    vanishing_length: np.ndarray
    homography_world_to_image: np.ndarray
    image_corners: np.ndarray
    world_corners: np.ndarray
    score: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pole_ids": list(self.pole_ids),
            "pole_feet": np.round(self.pole_feet, 3).tolist(),
            "player_intersection": np.round(self.player_intersection, 3).tolist(),
            "player_intersection_ratio": round(float(self.player_intersection_ratio), 5),
            "net_line_id": self.net_line_id,
            "net_points": np.round(self.net_points, 3).tolist(),
            "net_score": round(float(self.net_score), 5),
            "vanishing_width": np.round(self.vanishing_width, 5).tolist(),
            "vanishing_length": np.round(self.vanishing_length, 5).tolist(),
            "homography_world_to_image": np.round(self.homography_world_to_image, 8).tolist(),
            "image_corners": np.round(self.image_corners, 3).tolist(),
            "world_corners": np.round(self.world_corners, 3).tolist(),
            "score": round(float(self.score), 5),
            "diagnostics": self.diagnostics,
        }


def detect_pole_candidates(frame_bgr: np.ndarray) -> list[PoleCandidate]:
    """Detect compact blue/dark post bases, not arbitrary vertical Hough lines."""

    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(
        hsv,
        np.array([90, 35, 20], dtype=np.uint8),
        np.array([135, 255, 215], dtype=np.uint8),
    )
    # A small close operation joins fragmented blue pixels in a base without
    # merging the large cyan wall region into every candidate.
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(blue, 8)
    candidates: list[PoleCandidate] = []
    for component_id in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[component_id])
        center = centers[component_id]
        if not 20 <= area <= 2000:
            continue
        if box_width > 80 or box_height > 80:
            continue
        if center[1] < 0.38 * height:
            continue
        # Very wide blue blobs are wall/floor reflections.  A post base is
        # compact relative to its bounding box.
        fill = float(area) / max(1.0, box_width * box_height)
        if fill < 0.08:
            continue
        support = min(1.0, 0.45 * fill + 0.55 * min(1.0, area / 220.0))
        candidates.append(
            PoleCandidate(
                pole_id=-1,
                foot=np.asarray(center, dtype=float),
                area=float(area),
                support=support,
                bbox=(x, y, box_width, box_height),
            )
        )

    # Merge duplicate components from a single base and keep deterministic IDs.
    candidates.sort(key=lambda item: (item.foot[0], item.foot[1]))
    merged: list[PoleCandidate] = []
    for candidate in candidates:
        if any(np.linalg.norm(candidate.foot - previous.foot) < 22.0 for previous in merged):
            continue
        candidate.pole_id = len(merged) + 1
        merged.append(candidate)
    return merged


def _select_pole_pair(
    poles: Sequence[PoleCandidate],
    player_points: np.ndarray,
    width: int,
    height: int,
) -> tuple[PoleCandidate, PoleCandidate, np.ndarray, float, float] | None:
    if len(poles) < 2:
        return None
    first_player, second_player = player_points
    best: tuple[float, PoleCandidate, PoleCandidate, np.ndarray, float, float] | None = None
    ordered = sorted(poles, key=lambda pole: pole.foot[0])
    for left, right in itertools.combinations(ordered, 2):
        feet_distance = float(np.linalg.norm(right.foot - left.foot))
        if feet_distance < max(35.0, 0.035 * width) or feet_distance > 0.42 * width:
            continue
        crossing = _segment_intersection(left.foot, right.foot, first_player, second_player)
        if crossing is None:
            continue
        point, pole_ratio, player_ratio = crossing
        if not 0.02 <= pole_ratio <= 0.98 or not 0.12 <= player_ratio <= 0.88:
            continue

        # Do not bridge over a third compact pole base.  This is what prevents
        # a wide pair from combining two neighbouring courts.
        intermediate = False
        for middle in ordered:
            if middle.pole_id in (left.pole_id, right.pole_id):
                continue
            projection = float(
                np.dot(middle.foot - left.foot, right.foot - left.foot)
                / max(1e-8, np.dot(right.foot - left.foot, right.foot - left.foot))
            )
            if 0.08 < projection < 0.92 and _point_segment_distance(middle.foot, left.foot, right.foot) < 48.0:
                intermediate = True
                break
        if intermediate:
            continue
        central_score = math.exp(-abs(player_ratio - 0.5) / 0.27)
        width_score = math.exp(-abs(feet_distance - 0.12 * width) / (0.16 * width))
        score = 3.0 * central_score + 1.4 * width_score + left.support + right.support
        if best is None or score > best[0]:
            best = (score, left, right, point, player_ratio, pole_ratio)
    if best is None:
        return None
    score, left, right, point, player_ratio, pole_ratio = best
    return left, right, point, float(player_ratio), float(score)


def _select_net_edge(
    lines: Sequence[Any],
    left_pole: PoleCandidate,
    right_pole: PoleCandidate,
    width: int,
    height: int,
) -> tuple[Any | None, np.ndarray | None, float]:
    """Select the net edge that lies between the chosen pole bases."""

    candidates: list[tuple[float, Any, np.ndarray]] = []
    for line in lines:
        if line.length < max(55.0, 0.045 * width):
            continue
        if line.white_support < 0.25:
            continue
        left_y = _line_y_at_x(line, left_pole.foot[0])
        right_y = _line_y_at_x(line, right_pole.foot[0])
        if left_y is None or right_y is None:
            continue
        left_gap = left_pole.foot[1] - left_y
        right_gap = right_pole.foot[1] - right_y
        if not 18.0 <= left_gap <= 240.0 or not 18.0 <= right_gap <= 240.0:
            continue
        midpoint_x = 0.5 * (left_pole.foot[0] + right_pole.foot[0])
        midpoint_y = 0.5 * (left_y + right_y)
        if not 0.18 * height <= midpoint_y <= 0.72 * height:
            continue
        role_bonus = 1.8 if line.role == "net_candidate" else 0.0
        support = 0.75 * line.white_support + 0.30 * line.floor_support
        length_score = min(1.0, line.length / max(1.0, 0.20 * width))
        # The net edge should be visibly above both bases and approximately
        # span the selected pole interval, even when Hough only saw a fragment.
        interval_score = math.exp(-abs((left_gap - right_gap)) / 90.0)
        score = 1.45 * line.score + support + 0.40 * length_score + role_bonus + 0.45 * interval_score
        candidates.append((score, line, np.asarray(((left_pole.foot[0], left_y), (right_pole.foot[0], right_y)), dtype=float)))
    if not candidates:
        return None, None, 0.0
    score, line, points = max(candidates, key=lambda item: item[0])
    return line, points, float(score)


def _angle_mean(lines: Sequence[Any], default: float) -> float:
    if not lines:
        return float(default)
    values = []
    weights = []
    for line in lines:
        angle = math.radians(float(line.angle_deg))
        values.append(complex(math.cos(2.0 * angle), math.sin(2.0 * angle)))
        weights.append(max(0.1, float(line.score) * float(line.length)))
    value = sum(item * weight for item, weight in zip(values, weights))
    return math.degrees(0.5 * math.atan2(value.imag, value.real)) % 180.0


def _line_intersection(first: Any, second: Any) -> np.ndarray | None:
    point = np.cross(_line_coeff(first), _line_coeff(second))
    if abs(float(point[2])) < 1e-7:
        return None
    result = point[:2] / point[2]
    return result if np.isfinite(result).all() else None


def _estimate_vanishing_point(lines: Sequence[Any], angle: float, width: int, height: int) -> np.ndarray:
    family = [
        line
        for line in lines
        if abs(((line.angle_deg - angle + 90.0) % 180.0) - 90.0) <= 18.0
        and line.length >= max(45.0, 0.045 * width)
    ]
    intersections: list[np.ndarray] = []
    for first, second in itertools.combinations(family[:24], 2):
        if abs(((first.angle_deg - second.angle_deg + 90.0) % 180.0) - 90.0) < 1.5:
            continue
        point = _line_intersection(first, second)
        if point is None:
            continue
        if -2.0 * width <= point[0] <= 3.0 * width and -2.0 * height <= point[1] <= 2.0 * height:
            intersections.append(point)
    if intersections:
        # The median suppresses intersections from short fragments belonging
        # to a neighbouring court.
        return np.r_[np.median(np.asarray(intersections), axis=0), 1.0]
    direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.0], dtype=float)
    return direction


def _homography_from_homogeneous(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for source, target in zip(src, dst):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        projected_basis = np.zeros((3, 9), dtype=float)
        source_x = np.array(
            [
                [source[0], source[1], source[2], 0, 0, 0, 0, 0, 0],
                [0, 0, 0, source[0], source[1], source[2], 0, 0, 0],
                [0, 0, 0, 0, 0, 0, source[0], source[1], source[2],
                ],
            ]
        )
        # target cross product with H*source; retain all three equations.
        tx, ty, tz = target
        rows.extend(
            [
                ty * source_x[2] - tz * source_x[1],
                tz * source_x[0] - tx * source_x[2],
                tx * source_x[1] - ty * source_x[0],
            ]
        )
    matrix = np.asarray(rows, dtype=float)
    _, _, vh = np.linalg.svd(matrix)
    homography = vh[-1].reshape(3, 3)
    if abs(homography[2, 2]) > 1e-8:
        homography /= homography[2, 2]
    return homography


def _project_world(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))]
    projected = (homography @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


COURT_HALF_WIDTH_M = 3.05
COURT_HALF_LENGTH_M = 6.70
COURT_LINE_DEPTHS_M = (-6.70, -5.94, -1.98, 1.98, 5.94, 6.70)


def _angle_distance(first: float, second: float) -> float:
    distance = abs(float(first) - float(second)) % 180.0
    return min(distance, 180.0 - distance)


def _line_through_point(point: Sequence[float], direction: Sequence[float]) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    second = point + direction
    return _line_coeff(type("Line", (), {"x1": point[0], "y1": point[1], "x2": second[0], "y2": second[1]})())


def _intersect_point_direction(
    point: Sequence[float],
    direction: Sequence[float],
    line: Any,
) -> np.ndarray | None:
    first = _line_through_point(point, direction)
    intersection = np.cross(first, _line_coeff(line))
    if abs(float(intersection[2])) < 1e-7:
        return None
    result = intersection[:2] / intersection[2]
    return result if np.isfinite(result).all() else None


def _line_parameter(line: Any, point: Sequence[float]) -> float:
    start = np.asarray((line.x1, line.y1), dtype=float)
    direction = np.asarray((line.x2 - line.x1, line.y2 - line.y1), dtype=float)
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-8:
        return 0.0
    return float(np.dot(np.asarray(point, dtype=float) - start, direction) / denominator)


def _project_world_line(
    homography: np.ndarray,
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[np.ndarray, np.ndarray] | None:
    projected = _project_world(homography, np.asarray([first, second], dtype=float))
    if not np.isfinite(projected).all():
        return None
    return projected[0], projected[1]


def _line_alignment_score(
    observed: Any,
    projected_first: Sequence[float],
    projected_second: Sequence[float],
) -> float:
    observed_equation = _line_coeff(observed)
    projected_line = type(
        "ProjectedLine",
        (),
        {
            "x1": float(projected_first[0]),
            "y1": float(projected_first[1]),
            "x2": float(projected_second[0]),
            "y2": float(projected_second[1]),
        },
    )()
    projected_equation = _line_coeff(projected_line)
    observed_points = np.asarray(
        [[observed.x1, observed.y1], [observed.x2, observed.y2]],
        dtype=float,
    )
    projected_points = np.asarray([projected_first, projected_second], dtype=float)
    observed_distance = np.abs(observed_points @ projected_equation[:2] + projected_equation[2]).mean()
    projected_distance = np.abs(projected_points @ observed_equation[:2] + observed_equation[2]).mean()
    angle = _angle_distance(float(observed.angle_deg), math.degrees(math.atan2(
        float(projected_second[1] - projected_first[1]),
        float(projected_second[0] - projected_first[0]),
    )))
    distance_score = math.exp(-0.5 * float(observed_distance + projected_distance) / 32.0)
    angle_score = math.exp(-angle / 16.0)
    return float(distance_score * angle_score * max(0.1, float(observed.score)))


def _template_line_alignment(
    homography: np.ndarray,
    lines: Sequence[Any],
    width: int,
) -> float:
    """Score how well detected line segments agree with the standard court."""

    template_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for x in (-COURT_HALF_WIDTH_M, 0.0, COURT_HALF_WIDTH_M):
        template_lines.append(((x, -COURT_HALF_LENGTH_M), (x, COURT_HALF_LENGTH_M)))
    for y in COURT_LINE_DEPTHS_M:
        template_lines.append(((-COURT_HALF_WIDTH_M, y), (COURT_HALF_WIDTH_M, y)))

    matches: list[float] = []
    for first, second in template_lines:
        projected = _project_world_line(homography, first, second)
        if projected is None:
            continue
        best = 0.0
        for line in lines:
            if line.length < 0.055 * float(width):
                continue
            best = max(best, _line_alignment_score(line, projected[0], projected[1]))
        matches.append(best)
    if not matches:
        return 0.0
    matches.sort(reverse=True)
    return float(sum(matches[: min(7, len(matches))]) / min(7, len(matches)))


def _estimate_line_ground_mapping(
    lines: Sequence[Any],
    net_line: Any,
    pole_feet: np.ndarray,
    player_points: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Estimate a court-plane H from pole feet and real court-line evidence.

    The two pole feet anchor the standard court's net intersections.  A pair
    of candidate side-line directions and one detected cross-court line supply
    a second, non-coplanar-in-the-net-depth pair of ground points.  We test
    standard badminton depths (back boundary and service lines), score the
    resulting H against all detected court lines, and keep the best hypothesis.
    This avoids camera intrinsics and avoids extrapolating court length from a
    four-point PnP solve on the vertical net plane.
    """

    left_foot, right_foot = np.asarray(pole_feet, dtype=float)
    net_span = float(np.linalg.norm(right_foot - left_foot))
    if net_span < 35.0:
        return None

    side_pool = [
        line
        for line in lines
        if line.line_id != net_line.line_id
        and line.length >= max(55.0, 0.06 * width)
        and line.white_support >= 0.18
        and line.floor_support >= 0.28
        and _angle_distance(line.angle_deg, net_line.angle_deg) >= 12.0
    ]
    side_pool.sort(key=lambda line: (line.score * math.sqrt(max(1.0, line.length))), reverse=True)
    side_pool = side_pool[:8]
    cross_pool = [
        line
        for line in lines
        if line.line_id != net_line.line_id
        and line.length >= max(70.0, 0.075 * width)
        and line.white_support >= 0.20
        and line.floor_support >= 0.28
    ]
    cross_pool.sort(key=lambda line: (line.score * math.sqrt(max(1.0, line.length))), reverse=True)
    cross_pool = cross_pool[:10]
    if len(side_pool) < 2 or not cross_pool:
        return None

    pre_candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, Any, float, Any, Any]] = []
    for left_line, right_line in itertools.permutations(side_pool, 2):
        if left_line.line_id == right_line.line_id:
            continue
        if _angle_distance(left_line.angle_deg, right_line.angle_deg) > 38.0:
            continue
        left_direction = np.asarray((left_line.x2 - left_line.x1, left_line.y2 - left_line.y1), dtype=float)
        right_direction = np.asarray((right_line.x2 - right_line.x1, right_line.y2 - right_line.y1), dtype=float)
        left_direction /= max(1e-8, float(np.linalg.norm(left_direction)))
        right_direction /= max(1e-8, float(np.linalg.norm(right_direction)))
        for left_sign in (-1.0, 1.0):
            for right_sign in (-1.0, 1.0):
                left_direction_signed = left_sign * left_direction
                right_direction_signed = right_sign * right_direction
                for cross_line in cross_pool:
                    left_cross = _intersect_point_direction(left_foot, left_direction_signed, cross_line)
                    right_cross = _intersect_point_direction(right_foot, right_direction_signed, cross_line)
                    if left_cross is None or right_cross is None:
                        continue
                    left_parameter = _line_parameter(cross_line, left_cross)
                    right_parameter = _line_parameter(cross_line, right_cross)
                    if not -0.30 <= left_parameter <= 1.30 or not -0.30 <= right_parameter <= 1.30:
                        continue
                    depth_distance = float(np.linalg.norm(left_cross - left_foot) + np.linalg.norm(right_cross - right_foot))
                    if depth_distance < 0.55 * net_span or depth_distance > 8.0 * net_span:
                        continue

                    for depth in COURT_LINE_DEPTHS_M:
                        source = np.asarray(
                            [
                                [-COURT_HALF_WIDTH_M, 0.0],
                                [COURT_HALF_WIDTH_M, 0.0],
                                [-COURT_HALF_WIDTH_M, depth],
                                [COURT_HALF_WIDTH_M, depth],
                            ],
                            dtype=np.float32,
                        )
                        target = np.asarray(
                            [left_foot, right_foot, left_cross, right_cross],
                            dtype=np.float32,
                        )
                        homography = cv2.getPerspectiveTransform(source, target)
                        corners = _project_world(
                            homography,
                            np.asarray(
                                [
                                    [-COURT_HALF_WIDTH_M, -COURT_HALF_LENGTH_M],
                                    [COURT_HALF_WIDTH_M, -COURT_HALF_LENGTH_M],
                                    [COURT_HALF_WIDTH_M, COURT_HALF_LENGTH_M],
                                    [-COURT_HALF_WIDTH_M, COURT_HALF_LENGTH_M],
                                ],
                                dtype=float,
                            ),
                        )
                        if not np.isfinite(corners).all():
                            continue
                        hull = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)
                        if len(hull) != 4:
                            continue
                        area = abs(float(cv2.contourArea(corners.astype(np.float32))))
                        if not 0.012 * width * height <= area <= 1.20 * width * height:
                            continue
                        if np.max(np.abs(corners)) > 5.0 * max(width, height):
                            continue
                        player_inliers = [
                            cv2.pointPolygonTest(corners.astype(np.float32), tuple(map(float, point)), True) >= -35.0
                            for point in player_points
                        ]
                        player_score = float(np.mean(player_inliers))
                        # The two locked players are the strongest available
                        # target-court constraint in a multi-court hall.  A
                        # hypothesis containing only one player is almost
                        # always a short service patch or a neighbouring
                        # court and must not become the full 13.4 m model.
                        if player_score < 0.99:
                            continue
                        anchor_score = (
                            0.55 * left_line.score
                            + 0.55 * right_line.score
                            + 0.70 * cross_line.score
                            + 0.60 * min(1.0, cross_line.length / max(1.0, 0.30 * width))
                        )
                        depth_prior = 0.25 if abs(depth) == COURT_HALF_LENGTH_M else 0.0
                        score = 4.2 * player_score + anchor_score + depth_prior
                        pre_candidates.append(
                            (
                                score,
                                homography,
                                corners,
                                np.asarray([left_cross, right_cross]),
                                cross_line,
                                depth,
                                left_line,
                                right_line,
                            )
                        )

    if not pre_candidates:
        return None
    pre_candidates.sort(key=lambda item: item[0], reverse=True)
    best: tuple[float, np.ndarray, np.ndarray, np.ndarray, Any, float, float] | None = None
    for candidate in pre_candidates[:24]:
        _, homography, corners, cross_points, cross_line, depth, left_line, right_line = candidate
        line_score = _template_line_alignment(homography, lines, width)
        score = (
            float(candidate[0])
            + 2.5 * line_score
            + 0.35 * float(left_line.score + right_line.score)
        )
        if best is None or score > best[0]:
            best = (score, homography, corners, cross_points, cross_line, depth, line_score)

    if best is None:
        return None
    score, homography, corners, cross_points, cross_line, depth, line_score = best
    return homography, corners, {
        "mapping": "ground_plane_homography_from_pole_feet_side_directions_and_cross_court_line",
        "ground_anchor_cross_line_id": int(cross_line.line_id),
        "ground_anchor_cross_depth_m": float(depth),
        "ground_anchor_cross_points": np.round(cross_points, 3).tolist(),
        "line_template_alignment_score": round(float(line_score), 5),
        "court_length_m": 13.4,
        "court_width_m": 6.1,
    }


def recompute_pnp_ground_mapping(
    pole_feet: np.ndarray,
    net_points: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the camera-ground mapping from current-frame net geometry."""

    focal = 0.78 * float(width)
    camera_matrix = np.array(
        [[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    object_points = np.array(
        [
            [-3.05, 0.0, 0.0],
            [3.05, 0.0, 0.0],
            [-3.05, 0.0, 1.524],
            [3.05, 0.0, 1.524],
        ],
        dtype=np.float32,
    )
    image_points = np.asarray(
        [pole_feet[0], pole_feet[1], net_points[0], net_points[1]],
        dtype=np.float32,
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed for current-frame net geometry")
    rotation, _ = cv2.Rodrigues(rvec)
    homography = camera_matrix @ np.column_stack((rotation[:, 0], rotation[:, 1], tvec.reshape(3)))
    homography /= max(1e-8, float(homography[2, 2]))
    world_corners = np.array(
        [
            [-3.05, -6.7, 0.0],
            [3.05, -6.7, 0.0],
            [3.05, 6.7, 0.0],
            [-3.05, 6.7, 0.0],
        ],
        dtype=np.float32,
    )
    projected, _ = cv2.projectPoints(world_corners, rvec, tvec, camera_matrix, None)
    return homography, projected.reshape(-1, 2)


def _estimate_mapping(
    lines: Sequence[Any],
    net_line: Any,
    pole_feet: np.ndarray,
    net_points: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # The repository's original PnP script uses known net width and height
    # for this step.  We do the same here: two post feet and the two net-top
    # points form four 3-D correspondences.  This removes the scale ambiguity
    # that remains if one uses only two vanishing directions.  The source
    # video has no camera calibration file, so use a documented approximate
    # pinhole matrix; it can be replaced by calibrated intrinsics later.
    homography, image_corners = recompute_pnp_ground_mapping(
        pole_feet,
        net_points,
        width,
        height,
    )
    net_angle = float(net_line.angle_deg)
    cross_lines = [line for line in lines if line.family == "cross_court_family" and abs(((line.angle_deg - net_angle + 90.0) % 180.0) - 90.0) > 25.0]
    length_angle = _angle_mean(cross_lines, 0.0)
    vanishing_width = _estimate_vanishing_point(lines, net_angle, width, height)
    vanishing_length = np.array(
        [math.cos(math.radians(length_angle)), math.sin(math.radians(length_angle)), 0.0],
        dtype=float,
    )
    world_corners = np.array(
        [
            [-3.05, -6.7],
            [3.05, -6.7],
            [3.05, 6.7],
            [-3.05, 6.7],
        ],
        dtype=float,
    )
    return homography, image_corners, vanishing_width, vanishing_length


def select_pole_guided_court(
    frame_bgr: np.ndarray,
    lines: Sequence[Any],
    player_points: Sequence[Sequence[float]],
) -> PoleMappedCourt | None:
    """Find the target court by the intersecting pole-foot/player segments."""

    points = np.asarray(player_points, dtype=float).reshape(-1, 2)
    if len(points) != 2 or not np.isfinite(points).all():
        return None
    height, width = frame_bgr.shape[:2]
    poles = detect_pole_candidates(frame_bgr)
    pair = _select_pole_pair(poles, points, width, height)
    if pair is None:
        return None
    left, right, player_intersection, player_ratio, pair_score = pair
    net_line, net_points, net_score = _select_net_edge(lines, left, right, width, height)
    if net_line is None or net_points is None:
        return None
    pole_feet = np.stack([left.foot, right.foot])
    line_mapping = None
    try:
        line_mapping = _estimate_line_ground_mapping(
            lines,
            net_line,
            pole_feet,
            points,
            width,
            height,
        )
    except (cv2.error, RuntimeError, ValueError, np.linalg.LinAlgError):
        line_mapping = None

    if line_mapping is not None:
        homography, image_corners, mapping_diagnostics = line_mapping
        vanishing_width = np.r_[right.foot - left.foot, 0.0]
        vanishing_length = np.r_[image_corners[2] - image_corners[1], 0.0]
        mapping_diagnostics = dict(mapping_diagnostics)
        mapping_diagnostics["mapping_fallback"] = False
    else:
        try:
            homography, image_corners, vanishing_width, vanishing_length = _estimate_mapping(
                lines,
                net_line,
                pole_feet,
                net_points,
                width,
                height,
            )
        except (cv2.error, RuntimeError, ValueError, np.linalg.LinAlgError):
            # A single noisy frame must not terminate a video-wide detector run.
            return None
        mapping_diagnostics = {
            "mapping": "fallback_pnp_from_net_width_and_height",
            "approximate_camera_focal_px": round(0.78 * float(width), 3),
            "mapping_fallback": True,
            "court_length_m": 13.4,
            "court_width_m": 6.1,
            "net_height_m": 1.524,
        }
    finite_corners = np.isfinite(image_corners).all()
    if not finite_corners:
        return None
    score = pair_score + net_score + 2.0 * math.exp(-abs(player_ratio - 0.5) / 0.25)
    return PoleMappedCourt(
        pole_ids=(left.pole_id, right.pole_id),
        pole_feet=pole_feet,
        player_intersection=player_intersection,
        player_intersection_ratio=player_ratio,
        net_line_id=net_line.line_id,
        net_points=net_points,
        net_score=net_score,
        vanishing_width=vanishing_width,
        vanishing_length=vanishing_length,
        homography_world_to_image=homography,
        image_corners=image_corners,
        world_corners=np.array([[-3.05, -6.7], [3.05, -6.7], [3.05, 6.7], [-3.05, 6.7]], dtype=float),
        score=score,
        diagnostics={
            "pole_candidates": [pole.as_dict() for pole in poles],
            "pole_pair_distance_px": round(float(np.linalg.norm(right.foot - left.foot)), 3),
            **mapping_diagnostics,
            "net_height_m": 1.524,
        },
    )


def annotate_pole_guided_court(
    frame_bgr: np.ndarray,
    court: PoleMappedCourt,
    player_points: np.ndarray | None = None,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw exactly four mapped court sides, the chosen net and the pole pair."""

    output = frame_bgr.copy()
    polygon = np.round(court.image_corners).astype(np.int32)
    cv2.polylines(output, [polygon.reshape(-1, 1, 2)], True, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.line(
        output,
        tuple(np.round(court.net_points[0]).astype(int)),
        tuple(np.round(court.net_points[1]).astype(int)),
        (255, 0, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.line(
        output,
        tuple(np.round(court.pole_feet[0]).astype(int)),
        tuple(np.round(court.pole_feet[1]).astype(int)),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for index, point in enumerate(court.pole_feet, start=1):
        cv2.circle(output, tuple(np.round(point).astype(int)), 8, (255, 120, 0), 3, cv2.LINE_AA)
        if show_labels:
            cv2.putText(output, f"POST {index}", tuple(np.round(point).astype(int) + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2, cv2.LINE_AA)
    cv2.circle(output, tuple(np.round(court.player_intersection).astype(int)), 7, (0, 255, 255), -1, cv2.LINE_AA)
    if player_points is not None:
        colors = ((0, 165, 255), (255, 180, 0))
        cv2.line(output, tuple(np.round(player_points[0]).astype(int)), tuple(np.round(player_points[1]).astype(int)), (255, 255, 255), 2, cv2.LINE_AA)
        for point, color in zip(player_points, colors):
            cv2.circle(output, tuple(np.round(point).astype(int)), 7, color, -1, cv2.LINE_AA)
    title = f"Pole-pair court mapping | net L{court.net_line_id} | poles {court.pole_ids}"
    cv2.rectangle(output, (8, 8), (min(output.shape[1] - 8, 760), 40), (0, 0, 0), -1)
    cv2.putText(output, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return output
