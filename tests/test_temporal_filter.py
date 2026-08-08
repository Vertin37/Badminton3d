from __future__ import annotations

import unittest

import numpy as np

from src.pose_biomechanics.temporal_filter import (
    TemporalFilterConfig,
    filter_pose_arrays,
)


def _synthetic_pose(frame_count: int = 8) -> tuple[np.ndarray, np.ndarray]:
    raw = np.full((frame_count, 1, 133, 2), np.nan, dtype=float)
    scores = np.zeros((frame_count, 1, 133), dtype=float)
    body_offsets = {
        5: (-30, -50),
        6: (30, -50),
        11: (-25, 0),
        12: (25, 0),
        13: (-20, 55),
        14: (20, 55),
        15: (-25, 110),
        16: (25, 110),
    }
    for frame_index in range(frame_count):
        center = np.array([200.0 + frame_index * 2.0, 200.0])
        for keypoint_id, offset in body_offsets.items():
            raw[frame_index, 0, keypoint_id] = center + np.asarray(offset)
            scores[frame_index, 0, keypoint_id] = 0.95
    return raw, scores


class TemporalFilterTests(unittest.TestCase):
    def test_isolated_fast_keypoint_spike_is_rejected_and_interpolated(self) -> None:
        raw, scores = _synthetic_pose()
        for frame_index in range(raw.shape[0]):
            raw[frame_index, 0, 9] = [100.0 + frame_index * 5.0, 100.0]
            scores[frame_index, 0, 9] = 0.95
        raw[3, 0, 9] = [900.0, 100.0]

        result = filter_pose_arrays(raw, scores)

        self.assertEqual(int(result.outlier_rejected[3, 0, 9]), 1)
        self.assertEqual(result.status[3, 0, 9], "interpolated")
        self.assertLess(float(result.filtered_xy[3, 0, 9, 0]), 200.0)

    def test_fast_motion_is_not_flattened(self) -> None:
        raw, scores = _synthetic_pose(frame_count=4)
        for frame_index, x in enumerate([100.0, 200.0, 300.0, 400.0]):
            raw[frame_index, 0, 9] = [x, 100.0]
            scores[frame_index, 0, 9] = 0.95

        result = filter_pose_arrays(raw, scores)

        self.assertTrue(np.isfinite(result.filtered_xy[:, 0, 9]).all())
        self.assertGreater(float(result.filtered_xy[-1, 0, 9, 0]), 330.0)

    def test_low_confidence_gap_is_interpolated_but_long_gap_is_not_fabricated(self) -> None:
        raw, scores = _synthetic_pose(frame_count=10)
        for frame_index, x in enumerate([100.0 + frame_index * 10.0 for frame_index in range(10)]):
            raw[frame_index, 0, 5] = [x, 100.0]
            scores[frame_index, 0, 5] = 0.95
        raw[2, 0, 5] = [999.0, 100.0]
        scores[2, 0, 5] = 0.10
        raw[5:9, 0, 5] = np.nan
        scores[5:9, 0, 5] = 0.0

        result = filter_pose_arrays(
            raw,
            scores,
            TemporalFilterConfig(max_interpolation_gap=2, max_hold_gap=1),
        )

        self.assertEqual(result.status[2, 0, 5], "interpolated")
        self.assertTrue(np.isfinite(result.filtered_xy[2, 0, 5]).all())
        self.assertFalse(np.isfinite(result.filtered_xy[6, 0, 5]).any())

    def test_result_contains_comparison_data_and_coverage_stats(self) -> None:
        raw, scores = _synthetic_pose(frame_count=3)
        result = filter_pose_arrays(raw, scores)

        self.assertEqual(result.filtered_xy.shape, raw.shape)
        self.assertEqual(result.raw_valid.shape, scores.shape)
        self.assertEqual(result.stats["processed_frames"], 3)
        self.assertIn("coverage", result.stats["per_player"]["0"])


if __name__ == "__main__":
    unittest.main()
