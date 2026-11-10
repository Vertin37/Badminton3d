from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from run_court_detection import _load_player_targets


class CourtPlayerTargetTests(unittest.TestCase):
    def test_small_background_identity_takeover_is_not_used_for_court(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "pose.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "frame_id",
                        "player_id",
                        "keypoint_id",
                        "x",
                        "y",
                        "body_scale",
                    ),
                )
                writer.writeheader()
                for frame_id in range(1, 12):
                    for player_id in (0, 1):
                        scale = 210.0 if player_id == 0 else 125.0
                        x = 300.0 + 250.0 * player_id
                        if frame_id == 10 and player_id == 0:
                            scale = 62.0
                            x = 920.0
                        for keypoint_id, offset in ((15, -6.0), (16, 6.0)):
                            writer.writerow(
                                {
                                    "frame_id": frame_id,
                                    "player_id": player_id,
                                    "keypoint_id": keypoint_id,
                                    "x": x + offset,
                                    "y": 600.0,
                                    "body_scale": scale,
                                }
                            )

            frame_pairs, all_points, _ = _load_player_targets(csv_path)

        self.assertNotIn(10, frame_pairs)
        self.assertIn(9, frame_pairs)
        self.assertIn(11, frame_pairs)
        self.assertIsNotNone(all_points)
        self.assertLess(float(all_points[:, 0].max()), 900.0)


if __name__ == "__main__":
    unittest.main()
