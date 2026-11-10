from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from src.court_detection.detector import CourtLine
from src.court_detection.line_mapper import (
    COURT_WORLD_CORNERS,
    COURT_X_LINES_M,
    COURT_Y_LINES_M,
    select_line_guided_court,
)


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))]
    projected = (homography @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def _court_line(
    line_id: int,
    first: np.ndarray,
    second: np.ndarray,
    family: str,
) -> CourtLine:
    delta = second - first
    return CourtLine(
        line_id=line_id,
        x1=int(round(float(first[0]))),
        y1=int(round(float(first[1]))),
        x2=int(round(float(second[0]))),
        y2=int(round(float(second[1]))),
        length=float(np.linalg.norm(delta)),
        angle_deg=float(math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0),
        score=0.92,
        white_support=0.96,
        edge_support=0.95,
        floor_support=0.90,
        family=family,
        role="court_line",
        role_confidence=0.9,
    )


class GroundLineMapperTests(unittest.TestCase):
    def test_recovers_metric_court_without_posts_or_net(self) -> None:
        width, height = 1280, 720
        true_corners = np.asarray(
            [[310.0, 155.0], [925.0, 168.0], [1120.0, 680.0], [105.0, 674.0]],
            dtype=np.float32,
        )
        true_h = cv2.getPerspectiveTransform(
            COURT_WORLD_CORNERS.astype(np.float32),
            true_corners,
        )
        frame = np.full((height, width, 3), (35, 115, 45), dtype=np.uint8)
        lines: list[CourtLine] = []
        line_id = 1
        for x in COURT_X_LINES_M:
            endpoints = _project(
                true_h,
                np.asarray([[x, -6.70], [x, 6.70]], dtype=np.float64),
            )
            cv2.line(frame, tuple(np.round(endpoints[0]).astype(int)), tuple(np.round(endpoints[1]).astype(int)), (245, 245, 245), 4)
            lines.append(_court_line(line_id, endpoints[0], endpoints[1], "length_court_family"))
            line_id += 1
        for y in COURT_Y_LINES_M:
            endpoints = _project(
                true_h,
                np.asarray([[-3.05, y], [3.05, y]], dtype=np.float64),
            )
            cv2.line(frame, tuple(np.round(endpoints[0]).astype(int)), tuple(np.round(endpoints[1]).astype(int)), (245, 245, 245), 4)
            lines.append(_court_line(line_id, endpoints[0], endpoints[1], "cross_court_family"))
            line_id += 1

        players = _project(
            true_h,
            np.asarray([[-0.6, -3.0], [0.8, 3.2]], dtype=np.float64),
        )
        recovered = select_line_guided_court(frame, lines, players, refine=True)

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertFalse(recovered.diagnostics["pole_evidence"])
        self.assertGreaterEqual(float(recovered.diagnostics["confidence"]), 0.65)
        corner_error = np.linalg.norm(recovered.image_corners - true_corners, axis=1)
        self.assertLess(float(np.median(corner_error)), 12.0)
        self.assertTrue(
            str(recovered.diagnostics["mapping"]).startswith("ground_plane_homography")
        )

    def test_rejects_sparse_background_rectangle(self) -> None:
        width, height = 960, 540
        frame = np.full((height, width, 3), (45, 55, 45), dtype=np.uint8)
        lines = [
            _court_line(1, np.asarray([180.0, 120.0]), np.asarray([780.0, 120.0]), "cross_court_family"),
            _court_line(2, np.asarray([120.0, 430.0]), np.asarray([840.0, 430.0]), "cross_court_family"),
            _court_line(3, np.asarray([180.0, 120.0]), np.asarray([120.0, 430.0]), "length_court_family"),
            _court_line(4, np.asarray([780.0, 120.0]), np.asarray([840.0, 430.0]), "length_court_family"),
        ]
        players = np.asarray([[390.0, 250.0], [570.0, 330.0]], dtype=np.float64)

        recovered = select_line_guided_court(frame, lines, players, refine=False)

        self.assertIsNone(recovered)


if __name__ == "__main__":
    unittest.main()
