from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.court_detection.pole_mapper import select_pole_guided_court


class PoleMapperTests(unittest.TestCase):
    def test_vertical_blue_posts_drive_physical_pnp_without_hough_net(self) -> None:
        image = np.full((800, 1280, 3), (70, 145, 55), dtype=np.uint8)
        cv2.rectangle(image, (130, 340), (150, 564), (255, 0, 0), -1)
        cv2.rectangle(image, (748, 380), (770, 528), (255, 0, 0), -1)
        player_points = np.array([[356.0, 463.0], [998.0, 624.0]], dtype=float)

        court = select_pole_guided_court(image, [], player_points)

        self.assertIsNotNone(court)
        assert court is not None
        self.assertIsNone(court.net_line_id)
        self.assertIn(
            court.diagnostics["mapping"],
            {
                "pnp_from_detected_vertical_posts",
                "ground_plane_homography_from_pole_feet_side_directions_and_cross_court_line",
            },
        )
        self.assertEqual(court.image_corners.shape, (4, 2))
        self.assertTrue(np.isfinite(court.image_corners).all())
        # The physical IPPE branch must preserve meaningful court depth; the
        # mirror branch collapses all four corners into a narrow strip.
        self.assertGreater(float(np.ptp(court.image_corners[:, 1])), 0.25 * image.shape[0])


if __name__ == "__main__":
    unittest.main()
