import math

import numpy as np

from src.court_detection.player_pole_geometry import compute_player_pole_geometry
from src.court_detection.pole_mapper import PoleMappedCourt


def _identity_court() -> PoleMappedCourt:
    return PoleMappedCourt(
        pole_ids=(1, 2),
        pole_feet=np.asarray([[-3.05, 0.0], [3.05, 0.0]], dtype=float),
        player_intersection=np.asarray([0.0, 0.0]),
        player_intersection_ratio=0.5,
        net_line_id=1,
        net_points=np.asarray([[-3.05, 0.0], [3.05, 0.0]], dtype=float),
        net_score=1.0,
        vanishing_width=np.asarray([1.0, 0.0, 0.0]),
        vanishing_length=np.asarray([0.0, 1.0, 0.0]),
        homography_world_to_image=np.eye(3),
        image_corners=np.asarray([[-3.05, -6.7], [3.05, -6.7], [3.05, 6.7], [-3.05, 6.7]]),
        world_corners=np.asarray([[-3.05, -6.7], [3.05, -6.7], [3.05, 6.7], [-3.05, 6.7]]),
        score=1.0,
        diagnostics={"mapping": "test"},
    )


def test_player_pole_geometry_reports_ground_distance_and_angle():
    rows = compute_player_pole_geometry(
        _identity_court(),
        [[0.0, 3.0], [0.0, -3.0]],
        frame_id=7,
    )

    assert len(rows) == 4
    assert all(row["metric_valid"] for row in rows)
    assert math.isclose(rows[0]["ground_distance_m"], math.sqrt(3.05**2 + 3.0**2), rel_tol=1e-6)
    assert math.isclose(rows[0]["pole_subtended_angle_deg"], rows[1]["pole_subtended_angle_deg"], rel_tol=1e-6)
    assert rows[0]["frame_id"] == 7
