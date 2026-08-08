"""Player-guided selection of one badminton court from Hough candidates.

The selector is deliberately net-first.  In a multi-court hall the highest
scoring Hough quadrilateral is often made from lines belonging to different
courts.  The locked players provide the search corridor, but the net and its
two posts are the anchor: the net gives the target court's width, post feet
give the floor reference, and long-edge lines are only accepted when they are
geometrically compatible with those anchors.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


def _angular_distance(a: float, b: float) -> float:
    distance = abs(float(a) - float(b)) % 180.0
    return min(distance, 180.0 - distance)


def _coeff(line: Any) -> np.ndarray:
    a = float(line.y1 - line.y2)
    b = float(line.x2 - line.x1)
    c = float(line.x1 * line.y2 - line.x2 * line.y1)
    norm = math.hypot(a, b)
    if norm < 1e-8:
        raise ValueError("degenerate line segment")
    return np.array([a / norm, b / norm, c / norm], dtype=float)


def _line_distance(line: Any, point: Sequence[float]) -> float:
    equation = _coeff(line)
    return abs(float(np.dot(equation[:2], point) + equation[2]))


def _intersection(first: Any, second: Any) -> np.ndarray | None:
    point = np.cross(_coeff(first), _coeff(second))
    if abs(float(point[2])) < 1e-7:
        return None
    result = point[:2] / point[2]
    return result if np.isfinite(result).all() else None


def _quadrilateral(first: Any, second: Any, left: Any, right: Any) -> np.ndarray | None:
    points = [
        _intersection(first, left),
        _intersection(first, right),
        _intersection(second, right),
        _intersection(second, left),
    ]
    if any(point is None for point in points):
        return None
    polygon = np.asarray(points, dtype=float)
    hull = cv2.convexHull(polygon.astype(np.float32)).reshape(-1, 2)
    if len(hull) != 4:
        return None
    center = polygon.mean(axis=0)
    order = np.argsort(np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0]))
    return polygon[order]


def _point_inside(polygon: np.ndarray, point: Sequence[float]) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.float32), tuple(map(float, point)), False) >= 0


@dataclass
class SelectedCourt:
    """One player-guided four-sided court hypothesis in image coordinates."""

    boundary_line_ids: tuple[int, int, int, int]
    vertices: np.ndarray
    score: float
    player_inlier_ratio: float
    player_points: np.ndarray
    player_axis_angle_deg: float
    net_line_id: int | None = None
    net_intersection: np.ndarray | None = None
    net_score: float = 0.0
    net_endpoint_points: np.ndarray | None = None
    pole_line_ids: tuple[int, int] | None = None
    pole_foot_points: np.ndarray | None = None
    long_edge_line_ids: tuple[int | None, int | None] = (None, None)
    construction_method: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "boundary_line_ids": list(self.boundary_line_ids),
            "vertices": np.round(self.vertices, 3).tolist(),
            "score": round(float(self.score), 5),
            "player_inlier_ratio": round(float(self.player_inlier_ratio), 5),
            "player_points": np.round(self.player_points, 3).tolist(),
            "player_axis_angle_deg": round(float(self.player_axis_angle_deg), 3),
            "net_line_id": self.net_line_id,
            "net_intersection": None
            if self.net_intersection is None
            else np.round(self.net_intersection, 3).tolist(),
            "net_score": round(float(self.net_score), 5),
            "net_endpoint_points": None
            if self.net_endpoint_points is None
            else np.round(self.net_endpoint_points, 3).tolist(),
            "pole_line_ids": None if self.pole_line_ids is None else list(self.pole_line_ids),
            "pole_foot_points": None
            if self.pole_foot_points is None
            else np.round(self.pole_foot_points, 3).tolist(),
            "long_edge_line_ids": list(self.long_edge_line_ids),
            "construction_method": self.construction_method,
        }


def _select_net(
    lines: Sequence[Any],
    boundary_ids: set[int],
    player_points: np.ndarray,
    axis_angle: float,
    width: int,
    height: int,
) -> tuple[int | None, np.ndarray | None, float]:
    """Find the target net from the player corridor.

    A real net can meet the player connection close to one player, especially
    in a side-on crop.  Therefore the old ``near the middle`` rule is only a
    weak tie breaker.  A line already labelled ``net_candidate`` and carrying
    bright-floor support wins over an unrelated wall or court line.
    """

    first, second = player_points
    direction = second - first
    direction_length = float(np.linalg.norm(direction))
    if direction_length < 1e-6:
        return None, None, 0.0

    best: tuple[float, int, np.ndarray] | None = None
    for line in lines:
        if line.line_id in boundary_ids:
            continue
        midpoint_y = line.midpoint[1]
        if not 0.30 * height <= midpoint_y <= 0.70 * height:
            continue
        angle_distance = _angular_distance(line.angle_deg, axis_angle)
        if angle_distance < 8.0 or angle_distance > 78.0:
            continue
        equation = _coeff(line)
        value_first = float(np.dot(equation[:2], first) + equation[2])
        value_second = float(np.dot(equation[:2], second) + equation[2])
        denominator = value_first - value_second
        if abs(denominator) < 1e-7:
            continue
        ratio = value_first / denominator
        if not -0.06 <= ratio <= 1.06:
            continue
        intersection = first + ratio * direction
        middle_score = math.exp(-abs(ratio - 0.5) / 0.48)
        length_score = min(1.0, line.length / max(1.0, 0.16 * width))
        role_bonus = 1.05 if line.role == "net_candidate" else 0.0
        support_bonus = 0.32 * line.white_support + 0.20 * line.floor_support
        # Keep a small preference for a crossing near the segment, but never
        # let it outweigh the semantic net hint and line support.
        score = (
            1.55 * line.score
            + 0.24 * middle_score
            + 0.30 * length_score
            + role_bonus
            + support_bonus
        )
        if best is None or score > best[0]:
            best = (score, line.line_id, intersection)
    if best is None:
        return None, None, 0.0
    return best[1], best[2], float(best[0])


def _segment_parameter(line: Any, point: Sequence[float]) -> float:
    start = np.asarray((line.x1, line.y1), dtype=float)
    direction = np.asarray((line.x2 - line.x1, line.y2 - line.y1), dtype=float)
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-8:
        return 0.0
    return float(np.dot(np.asarray(point, dtype=float) - start, direction) / denominator)


def _pole_candidates(
    lines: Sequence[Any],
    net: Any,
    width: int,
    height: int,
) -> list[tuple[float, Any, np.ndarray, float]]:
    """Return vertical post hypotheses intersecting the selected net."""

    candidates: list[tuple[float, Any, np.ndarray, float]] = []
    for line in lines:
        if line.line_id == net.line_id:
            continue
        if not 62.0 <= line.angle_deg <= 118.0:
            continue
        # A short vertical Hough fragment is usually net mesh or a player
        # limb.  The two posts in this footage occupy a substantial vertical
        # span, so require a longer support before using it as a pole anchor.
        if line.length < max(95.0, 0.12 * height):
            continue
        point = _intersection(net, line)
        if point is None:
            continue
        t = _segment_parameter(net, point)
        # The Hough net segment can stop well before the far post because a
        # player or mesh hides the white tape.  Permit a moderate extrapolated
        # tail so the real post is not replaced by a short mesh support.
        if not -0.18 <= t <= 1.35:
            continue
        # A post should extend below the net.  This rejects vertical wall
        # edges that happen to cross the infinite net line above the floor.
        lower_endpoint = max((line.y1, line.y2))
        if lower_endpoint < point[1] + 0.20 * line.length:
            continue
        endpoint_score = math.exp(-min(abs(t), abs(1.0 - t)) / 0.30)
        vertical_score = math.exp(-_angular_distance(line.angle_deg, 90.0) / 14.0)
        score = (
            1.15 * line.score
            + 0.55 * line.white_support
            + 0.35 * line.floor_support
            + 0.80 * min(1.0, line.length / max(1.0, 0.28 * height))
            + 0.55 * endpoint_score
            + 0.75 * vertical_score
        )
        candidates.append((score, line, point, t))
    return sorted(candidates, key=lambda item: item[0], reverse=True)


def _choose_poles(
    candidates: Sequence[tuple[float, Any, np.ndarray, float]],
) -> tuple[tuple[int, int], np.ndarray, np.ndarray, float] | None:
    """Choose two separated posts, one near each end of the net."""

    if len(candidates) < 2:
        return None
    best: tuple[float, Any, Any, np.ndarray, np.ndarray] | None = None
    for first, second in itertools.combinations(candidates[:12], 2):
        if first[1].line_id == second[1].line_id:
            continue
        if abs(first[3] - second[3]) < 0.55:
            continue
        left, right = sorted((first, second), key=lambda item: item[3])
        separation_score = min(1.0, abs(right[3] - left[3]))
        endpoint_score = math.exp(-min(abs(left[3]), abs(1.0 - right[3])) / 0.20)
        score = left[0] + right[0] + 0.7 * separation_score + 0.7 * endpoint_score
        if best is None or score > best[0]:
            best = (score, left[1], right[1], left[2], right[2])
    if best is None:
        return None
    score, left_line, right_line, left_point, right_point = best
    return (
        (left_line.line_id, right_line.line_id),
        np.stack([left_point, right_point]),
        np.asarray([left_line, right_line], dtype=object),
        float(score),
    )


def _lower_endpoint(line: Any, top: Sequence[float]) -> np.ndarray:
    """Use the lower detected endpoint as a conservative post-foot proxy."""

    first = np.asarray((line.x1, line.y1), dtype=float)
    second = np.asarray((line.x2, line.y2), dtype=float)
    return first if first[1] >= second[1] else second


def _line_direction(line: Any) -> np.ndarray:
    direction = np.asarray((line.x2 - line.x1, line.y2 - line.y1), dtype=float)
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm >= 1e-8 else np.array((1.0, 0.0), dtype=float)


def _line_distance_to_point(line: Any, point: Sequence[float]) -> float:
    return _line_distance(line, point)


def _find_long_edge_candidates(
    lines: Sequence[Any],
    net_id: int,
    pole_ids: set[int],
    feet: np.ndarray,
    net_angle: float,
    width: int,
) -> tuple[Any | None, Any | None]:
    """Find floor lines close to the two post feet.

    The line detector intentionally keeps this permissive.  A visible line
    close to a post is much more useful than a globally high-scoring line from
    an adjacent court.  If no suitable line exists, the caller synthesizes a
    side using the best dominant floor direction.
    """

    pool = [
        line
        for line in lines
        if line.line_id not in pole_ids | {net_id}
        and line.length >= max(55.0, 0.07 * width)
        and _angular_distance(line.angle_deg, 90.0) >= 18.0
    ]
    choices: list[list[tuple[float, Any]]] = [[], []]
    for index, foot in enumerate(feet):
        for line in pool:
            distance = _line_distance_to_point(line, foot)
            if distance > 58.0:
                continue
            net_difference = _angular_distance(line.angle_deg, net_angle)
            # Court length lines can be close to either the dominant horizontal
            # family or a perspective diagonal.  Keep both, but penalize a
            # candidate that is almost the net itself.
            direction_score = max(0.0, 1.0 - max(0.0, 24.0 - net_difference) / 24.0)
            score = (
                1.3 * line.score
                + 0.50 * line.white_support
                + 0.30 * line.floor_support
                + 0.45 * min(1.0, line.length / max(1.0, 0.25 * width))
                + 0.50 * math.exp(-distance / 28.0)
                + 0.20 * direction_score
            )
            choices[index].append((score, line))
        choices[index].sort(key=lambda item: item[0], reverse=True)

    # Prefer a pair which is not the same physical line and which has a
    # similar orientation; adjacent courts normally produce a cleaner pair
    # under this condition.
    best: tuple[float, Any, Any] | None = None
    for score_left, left in choices[0][:10]:
        for score_right, right in choices[1][:10]:
            if left.line_id == right.line_id:
                continue
            angle_consistency = math.exp(-_angular_distance(left.angle_deg, right.angle_deg) / 18.0)
            score = score_left + score_right + 0.55 * angle_consistency
            if best is None or score > best[0]:
                best = (score, left, right)
    if best is None:
        return None, None
    return best[1], best[2]


def _construct_net_first_polygon(
    net_endpoints: np.ndarray,
    feet: np.ndarray,
    left_edge: Any | None,
    right_edge: Any | None,
    player_points: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, str, tuple[int | None, int | None], float] | None:
    """Construct the four visible sides from net and post-floor anchors.

    The target crop often contains only one side of the court.  We therefore
    build a four-sided *net-to-baseline court patch*: the net is side 1, two
    sides follow the post-guided long-edge directions, and the fourth side is
    the synthesized baseline.  Several orientations/scales are evaluated and
    the one containing the locked players wins.  This is still image-space
    detection, not a metric reconstruction.
    """

    if net_endpoints.shape != (2, 2):
        return None
    net_length = float(np.linalg.norm(net_endpoints[1] - net_endpoints[0]))
    if net_length < 40.0:
        return None

    directions: list[np.ndarray | None] = []
    for line in (left_edge, right_edge):
        directions.append(None if line is None else _line_direction(line))
    fallback = net_endpoints[1] - net_endpoints[0]
    fallback = np.array((-fallback[1], fallback[0]), dtype=float)
    fallback /= max(1e-8, float(np.linalg.norm(fallback)))
    if directions[0] is None:
        directions[0] = fallback.copy()
    if directions[1] is None:
        directions[1] = fallback.copy()

    # Try both signs and a physically plausible image-space court length.
    best: tuple[float, np.ndarray, str] | None = None
    for sign_left in (-1.0, 1.0):
        for sign_right in (-1.0, 1.0):
            d_left = sign_left * directions[0]
            d_right = sign_right * directions[1]
            for length_ratio in (0.72, 0.90, 1.10, 1.32, 1.55):
                q_left = net_endpoints[0] + d_left * net_length * length_ratio
                q_right = net_endpoints[1] + d_right * net_length * length_ratio
                polygon = np.stack([net_endpoints[0], net_endpoints[1], q_right, q_left])
                hull = cv2.convexHull(polygon.astype(np.float32)).reshape(-1, 2)
                if len(hull) != 4:
                    continue
                if (
                    np.min(polygon[:, 0]) < -0.10 * width
                    or np.max(polygon[:, 0]) > 1.10 * width
                    # Reject stadium rails/wall geometry above the playing
                    # surface.  The net itself may start higher, but the
                    # four outer court sides must stay in the green-floor ROI.
                    or np.min(polygon[:, 1]) < 0.30 * height
                    or np.max(polygon[:, 1]) > 1.10 * height
                ):
                    continue
                inlier_ratio = float(np.mean([_point_inside(polygon, point) for point in player_points]))
                area_ratio = abs(float(cv2.contourArea(polygon.astype(np.float32)))) / float(width * height)
                if not 0.015 <= area_ratio <= 0.88:
                    continue
                player_center = np.mean(player_points, axis=0)
                center_score = math.exp(-float(np.linalg.norm(np.mean(polygon, axis=0) - player_center)) / (0.65 * width))
                # Prefer the smallest patch that still includes both players;
                # this is important with parallel courts in the background.
                compactness = math.exp(-max(0.0, area_ratio - 0.16) / 0.18)
                score = 5.0 * inlier_ratio + 0.8 * center_score + 0.5 * compactness
                if best is None or score > best[0]:
                    method = "net_poles_visible" if left_edge is not None and right_edge is not None else "net_poles_synthesized_edges"
                    best = (score, polygon, method)

    # If the players are on opposite sides of the net, the net is an interior
    # divider rather than an outer boundary.  Build the four outer sides by
    # extending the two post-guided long-edge directions on both sides of the
    # net.  This is the important multi-court case: it keeps the net as an
    # anchor without drawing it as one of the red perimeter sides.
    player_left = player_points[int(np.argmin(player_points[:, 0]))]
    player_right = player_points[int(np.argmax(player_points[:, 0]))]
    extra_left = player_left - net_endpoints[0]
    extra_right = player_right - net_endpoints[1]
    extra_left /= max(1e-8, float(np.linalg.norm(extra_left)))
    extra_right /= max(1e-8, float(np.linalg.norm(extra_right)))
    full_directions_left = [directions[0], extra_left, fallback]
    full_directions_right = [directions[1], extra_right, fallback]
    for base_left in full_directions_left:
        for base_right in full_directions_right:
          for sign_left in (-1.0, 1.0):
            for sign_right in (-1.0, 1.0):
              d_left = sign_left * base_left
              d_right = sign_right * base_right
              for half_length_ratio in (0.62, 0.82, 1.05, 1.30, 1.55):
                half_length = net_length * half_length_ratio
                first_left = net_endpoints[0] - d_left * half_length
                first_right = net_endpoints[1] - d_right * half_length
                second_right = net_endpoints[1] + d_right * half_length
                second_left = net_endpoints[0] + d_left * half_length
                raw_polygon = np.stack([first_left, first_right, second_right, second_left])
                polygon = cv2.convexHull(raw_polygon.astype(np.float32)).reshape(-1, 2)
                if len(polygon) != 4:
                    continue
                if (
                    np.min(polygon[:, 0]) < -0.10 * width
                    or np.max(polygon[:, 0]) > 1.10 * width
                    or np.min(polygon[:, 1]) < 0.30 * height
                    or np.max(polygon[:, 1]) > 1.10 * height
                ):
                    continue
                inlier_ratio = float(np.mean([_point_inside(polygon, point) for point in player_points]))
                area_ratio = abs(float(cv2.contourArea(polygon.astype(np.float32)))) / float(width * height)
                if not 0.025 <= area_ratio <= 0.90:
                    continue
                player_center = np.mean(player_points, axis=0)
                center_score = math.exp(-float(np.linalg.norm(np.mean(polygon, axis=0) - player_center)) / (0.70 * width))
                compactness = math.exp(-max(0.0, area_ratio - 0.22) / 0.24)
                # A full-court hypothesis gets a bonus only when it really
                # contains both locked players; otherwise the half-court
                # patch above remains the safer result.
                score = 8.0 * inlier_ratio + 0.85 * center_score + 0.55 * compactness
                if inlier_ratio >= 1.0:
                    score += 1.5
                if best is None or score > best[0]:
                    method = "net_poles_visible_full_court" if left_edge is not None and right_edge is not None else "net_poles_synthesized_full_court"
                    best = (score, polygon, method)
    if best is None:
        return None
    polygon = best[1]
    method = best[2]
    current_inlier_ratio = float(np.mean([_point_inside(polygon, point) for point in player_points]))
    if current_inlier_ratio < 1.0:
        # A player can sit just outside the mathematically smallest patch
        # when the camera crop clips one long edge.  Expand only toward the
        # outlying player by a fraction of the detected net width, then reduce
        # the convex hull back to four sides.  This is a bounded floor-space
        # extension, not a new search through wall/background lines.
        expanded_points = [point.copy() for point in polygon]
        expanded_points.extend(point.copy() for point in net_endpoints)
        margin = 0.25 * net_length
        for point in player_points:
            if not _point_inside(polygon, point):
                if point[0] <= float(np.mean(polygon[:, 0])):
                    expanded_points.append(np.asarray((point[0] - margin, point[1]), dtype=float))
                else:
                    expanded_points.append(np.asarray((point[0] + margin, point[1]), dtype=float))
        hull = cv2.convexHull(np.asarray(expanded_points, dtype=np.float32)).reshape(-1, 2)
        safe_candidates: list[tuple[float, np.ndarray]] = []
        if len(hull) >= 4:
            for indices in itertools.combinations(range(len(hull)), 4):
                candidate = cv2.convexHull(hull[list(indices)].astype(np.float32)).reshape(-1, 2)
                if len(candidate) != 4:
                    continue
                if not all(_point_inside(candidate, point) for point in player_points):
                    continue
                if np.min(candidate[:, 1]) < 0.30 * height:
                    continue
                area = abs(float(cv2.contourArea(candidate.astype(np.float32))))
                safe_candidates.append((area, candidate))
        if safe_candidates:
            _, polygon = min(safe_candidates, key=lambda item: item[0])
            method = f"{method}_expanded_to_players"
    return polygon, method, (
        None if left_edge is None else left_edge.line_id,
        None if right_edge is None else right_edge.line_id,
    ), float(best[0])


def select_target_court(
    lines: Sequence[Any],
    player_points: Sequence[Sequence[float]],
    width: int,
    height: int,
    player_axis_angle_deg: float | None = None,
) -> SelectedCourt | None:
    """Select one four-sided court using net and post geometry first.

    The previous implementation optimized an arbitrary player-containing
    quadrilateral.  That is unsafe when several courts are visible: four
    strong lines can come from four different courts.  This implementation
    first finds a net that crosses the player connection, then finds two
    vertical post candidates at the net ends, and only then constructs the
    four-sided court patch from post-guided long-edge directions.
    """

    if len(lines) < 4:
        return None
    points = np.asarray(player_points, dtype=float).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) != 2:
        return None
    if player_axis_angle_deg is None:
        delta = points[1] - points[0]
        player_axis_angle_deg = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0

    ranked = sorted(lines, key=lambda line: (line.score, line.length), reverse=True)
    net_id, net_intersection, net_score = _select_net(
        lines,
        set(),
        points,
        player_axis_angle_deg,
        width,
        height,
    )
    if net_id is None or net_intersection is None:
        return None
    net_line = next((line for line in ranked if line.line_id == net_id), None)
    if net_line is None:
        return None

    pole_options = _pole_candidates(lines, net_line, width, height)
    pole_choice = _choose_poles(pole_options)
    if pole_choice is None:
        # A net without visible poles is still useful as a reference, but it
        # is not safe to claim four court sides.  The caller will keep the
        # all-candidate overlay and diagnostics will show target not selected.
        return None
    pole_ids, net_endpoints, pole_lines, pole_score = pole_choice
    feet = np.stack([
        _lower_endpoint(pole_lines[0], net_endpoints[0]),
        _lower_endpoint(pole_lines[1], net_endpoints[1]),
    ])
    left_edge, right_edge = _find_long_edge_candidates(
        lines,
        net_id=net_id,
        pole_ids=set(pole_ids),
        feet=feet,
        net_angle=net_line.angle_deg,
        width=width,
    )
    constructed = _construct_net_first_polygon(
        net_endpoints=net_endpoints,
        feet=feet,
        left_edge=left_edge,
        right_edge=right_edge,
        player_points=points,
        width=width,
        height=height,
    )
    if constructed is None:
        return None
    polygon, construction_method, long_edge_ids, construction_score = constructed
    player_inlier_ratio = float(np.mean([_point_inside(polygon, point) for point in points]))
    boundary_ids = (net_id, long_edge_ids[0] or -1, long_edge_ids[1] or -1, pole_ids[1])
    return SelectedCourt(
        boundary_line_ids=boundary_ids,
        vertices=polygon,
        score=float(0.55 * net_score + 0.35 * pole_score + construction_score),
        player_inlier_ratio=player_inlier_ratio,
        player_points=points.copy(),
        player_axis_angle_deg=float(player_axis_angle_deg),
        net_line_id=net_id,
        net_intersection=net_intersection,
        net_score=net_score,
        net_endpoint_points=net_endpoints,
        pole_line_ids=pole_ids,
        pole_foot_points=feet,
        long_edge_line_ids=long_edge_ids,
        construction_method=construction_method,
    )
