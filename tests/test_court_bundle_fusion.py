from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "work" / "build_gvhmr_two_player_court_bundle.py"
SPEC = importlib.util.spec_from_file_location("court_bundle_builder", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CourtBundleFusionTests(unittest.TestCase):
    def test_schema_v2_ground_mapping_does_not_flip_y_twice(self) -> None:
        conversion, reason = MODULE._ground_track_conversion(
            2,
            "ground_plane_homography_from_pole_feet_side_directions_and_cross_court_line",
        )

        self.assertTrue(np.array_equal(conversion, np.eye(2)))
        self.assertIn("share the same", reason)

    def test_legacy_ground_mapping_keeps_historical_y_flip(self) -> None:
        conversion, _ = MODULE._ground_track_conversion(0, "fallback_pnp_from_net_width_and_height")

        self.assertTrue(np.array_equal(conversion, np.diag([1.0, -1.0])))

    def test_contact_outlier_cannot_teleport_whole_body(self) -> None:
        frame_count = 12
        raw = np.zeros((frame_count, 2), dtype=np.float32)
        ground = np.zeros((frame_count, 2), dtype=np.float32)
        ground[5:, 0] = 1.4
        contact = np.ones(frame_count, dtype=np.float32)

        fused, metadata = MODULE._fuse_root_xy(
            raw,
            ground,
            contact,
            contact_threshold=0.6,
            max_measurement_residual_m=1.5,
            fps=30.0,
            max_contact_correction_speed_mps=1.5,
        )

        speed = np.linalg.norm(np.diff(fused, axis=0), axis=1) * 30.0
        self.assertLessEqual(float(speed.max()), 1.5 + 1e-5)
        self.assertGreater(metadata["rate_limited_ground_updates"], 0)

    def test_smooth_ground_drift_is_followed(self) -> None:
        frame_count = 31
        raw = np.zeros((frame_count, 2), dtype=np.float32)
        ground = np.zeros((frame_count, 2), dtype=np.float32)
        ground[:, 0] = np.linspace(0.0, 0.3, frame_count)
        contact = np.ones(frame_count, dtype=np.float32)

        fused, _ = MODULE._fuse_root_xy(
            raw,
            ground,
            contact,
            contact_threshold=0.6,
            max_measurement_residual_m=1.5,
            fps=30.0,
            max_contact_correction_speed_mps=1.5,
        )

        self.assertGreater(float(fused[-1, 0]), 0.20)
        self.assertLess(float(fused[-1, 0]), 0.31)


if __name__ == "__main__":
    unittest.main()
