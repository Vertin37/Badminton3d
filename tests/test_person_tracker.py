from __future__ import annotations

import unittest

import numpy as np

from src.pose_biomechanics.person_tracker import PersonTracker


def _pose(center_x: float, center_y: float, style: int) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((133, 2), dtype=float)
    scores = np.zeros(133, dtype=float)
    if style == 0:
        offsets = {
            5: (-20, -50), 6: (20, -50),
            11: (-15, 0), 12: (15, 0),
            13: (-30, 60), 14: (8, 60),
            15: (-38, 125), 16: (15, 125),
        }
    else:
        offsets = {
            5: (-22, -42), 6: (22, -42),
            11: (-17, 0), 12: (17, 0),
            13: (-8, 42), 14: (30, 70),
            15: (5, 90), 16: (48, 145),
        }
    for keypoint_id, (dx, dy) in offsets.items():
        keypoints[keypoint_id] = (center_x + dx, center_y + dy)
        scores[keypoint_id] = 0.95
    return keypoints, scores


class PersonTrackerTests(unittest.TestCase):
    def test_detection_order_swap_does_not_change_ids(self) -> None:
        tracker = PersonTracker()
        left = _pose(100, 200, 0)
        right = _pose(400, 200, 0)

        for frame_id in range(8):
            detections = [left, right] if frame_id % 2 == 0 else [right, left]
            result = tracker.update(
                np.stack([item[0] for item in detections]),
                np.stack([item[1] for item in detections]),
            )
            self.assertTrue(all(item.matched for item in result))
            self.assertLess(result[0].detection.center[0], 200)
            self.assertGreater(result[1].detection.center[0], 300)

    def test_crossing_uses_pose_shape_instead_of_left_right_position(self) -> None:
        tracker = PersonTracker()
        slot_markers: list[float] = []
        for frame_id in range(21):
            player_a = _pose(100 + 10 * frame_id, 200, 0)
            player_b = _pose(300 - 10 * frame_id, 200, 1)
            detections = [player_a, player_b] if frame_id % 2 == 0 else [player_b, player_a]
            result = tracker.update(
                np.stack([item[0] for item in detections]),
                np.stack([item[1] for item in detections]),
            )
            marker = result[0].detection.keypoints[13, 1] - result[0].detection.center[1]
            slot_markers.append(round(float(marker), 5))

        self.assertTrue(all(marker == slot_markers[0] for marker in slot_markers))

    def test_short_missing_does_not_reassign_the_other_player(self) -> None:
        tracker = PersonTracker()
        player_a = _pose(100, 200, 0)
        player_b = _pose(400, 200, 1)

        tracker.update(
            np.stack([player_a[0], player_b[0]]),
            np.stack([player_a[1], player_b[1]]),
        )
        one_person = tracker.update(
            np.expand_dims(player_a[0], 0),
            np.expand_dims(player_a[1], 0),
        )
        self.assertEqual(sum(item.matched for item in one_person), 1)
        matched = next(item for item in one_person if item.matched)
        missing = next(item for item in one_person if not item.matched)
        self.assertAlmostEqual(float(matched.detection.center[0]), 100.0, delta=8.0)
        self.assertEqual(matched.missed_frames, 0)
        self.assertEqual(missing.missed_frames, 1)


if __name__ == "__main__":
    unittest.main()
