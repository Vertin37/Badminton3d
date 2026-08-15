"""Ground-plane geometry between tracked players and the selected net poles."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .pole_mapper import COURT_HALF_WIDTH_M, PoleMappedCourt


PLAYER_POLE_FIELDS = [
    "frame_id",
    "player_id",
    "pole_index",
    "pole_id",
    "player_image_x",
    "player_image_y",
    "player_ground_x_m",
    "player_ground_y_m",
    "pole_image_x",
    "pole_image_y",
    "pole_ground_x_m",
    "pole_ground_y_m",
    "image_distance_px",
    "image_bearing_deg",
    "ground_distance_m",
    "ground_bearing_deg",
    "pole_subtended_angle_deg",
    "metric_valid",
    "mapping",
    "mapping_fallback",
]


def _project_image_to_ground(point: Sequence[float], homography: np.ndarray) -> np.ndarray | None:
    """Project one image point through the court-plane inverse homography."""

    point = np.asarray(point, dtype=float).reshape(2)
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    projected = inverse @ np.asarray([point[0], point[1], 1.0], dtype=float)
    if abs(float(projected[2])) < 1e-8:
        return None
    result = projected[:2] / projected[2]
    return result if np.isfinite(result).all() else None


def _bearing_degrees(vector: np.ndarray) -> float | None:
    vector = np.asarray(vector, dtype=float).reshape(2)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8 or not np.isfinite(vector).all():
        return None
    return float(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))


def _subtended_angle_degrees(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=float).reshape(2)
    second = np.asarray(second, dtype=float).reshape(2)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        return None
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return float(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))


def compute_player_pole_geometry(
    court: PoleMappedCourt,
    player_points: Sequence[Sequence[float]],
    frame_id: int,
) -> list[dict[str, Any]]:
    """Return one metric row per player/pole pair for a stabilized court model.

    Ground coordinates use the existing court mapping: ``x`` is across the
    net, ``y`` is along the court, and the selected net is at ``y=0``.  The
    two standard pole locations are ``(-3.05, 0)`` and ``(3.05, 0)`` metres.
    Pixel metrics remain available when a frame only has a guarded fallback
    mapping or the inverse homography is not numerically usable.
    """

    points = np.asarray(player_points, dtype=float).reshape(-1, 2)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        return []
    pole_images = np.asarray(court.pole_feet, dtype=float)
    if pole_images.shape != (2, 2) or not np.isfinite(pole_images).all():
        return []

    pole_world = np.asarray(
        [[-COURT_HALF_WIDTH_M, 0.0], [COURT_HALF_WIDTH_M, 0.0]],
        dtype=float,
    )
    player_world = [_project_image_to_ground(point, court.homography_world_to_image) for point in points]
    mapping = str(court.diagnostics.get("mapping", "unknown"))
    mapping_fallback = bool(court.diagnostics.get("mapping_fallback", False))
    subtended_angles = [
        None
        if world_point is None
        else _subtended_angle_degrees(pole_world[0] - world_point, pole_world[1] - world_point)
        for world_point in player_world
    ]

    rows: list[dict[str, Any]] = []
    for player_id, (player_image, player_ground, subtended_angle) in enumerate(
        zip(points, player_world, subtended_angles)
    ):
        for pole_index, (pole_image, pole_ground) in enumerate(zip(pole_images, pole_world)):
            image_vector = pole_image - player_image
            image_distance = float(np.linalg.norm(image_vector))
            image_bearing = _bearing_degrees(image_vector)
            if player_ground is None:
                ground_vector = None
                ground_distance = None
                ground_bearing = None
            else:
                ground_vector = pole_ground - player_ground
                ground_distance = float(np.linalg.norm(ground_vector))
                ground_bearing = _bearing_degrees(ground_vector)
            rows.append(
                {
                    "frame_id": int(frame_id),
                    "player_id": int(player_id),
                    "pole_index": int(pole_index),
                    "pole_id": int(court.pole_ids[pole_index]),
                    "player_image_x": round(float(player_image[0]), 3),
                    "player_image_y": round(float(player_image[1]), 3),
                    "player_ground_x_m": None if player_ground is None else round(float(player_ground[0]), 5),
                    "player_ground_y_m": None if player_ground is None else round(float(player_ground[1]), 5),
                    "pole_image_x": round(float(pole_image[0]), 3),
                    "pole_image_y": round(float(pole_image[1]), 3),
                    "pole_ground_x_m": round(float(pole_ground[0]), 5),
                    "pole_ground_y_m": round(float(pole_ground[1]), 5),
                    "image_distance_px": round(image_distance, 4),
                    "image_bearing_deg": None if image_bearing is None else round(image_bearing, 4),
                    "ground_distance_m": None if ground_distance is None else round(ground_distance, 5),
                    "ground_bearing_deg": None if ground_bearing is None else round(ground_bearing, 4),
                    "pole_subtended_angle_deg": (
                        None if subtended_angle is None else round(subtended_angle, 4)
                    ),
                    "metric_valid": bool(player_ground is not None and ground_distance is not None),
                    "mapping": mapping,
                    "mapping_fallback": mapping_fallback,
                }
            )
    return rows
