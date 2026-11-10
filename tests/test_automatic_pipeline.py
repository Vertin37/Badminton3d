from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "work" / "run_badminton_3d_pipeline.py"
SPEC = importlib.util.spec_from_file_location("automatic_pipeline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AutomaticPipelineTests(unittest.TestCase):
    def test_depth_window_is_selected_from_motion_without_manual_frames(self) -> None:
        pose_csv = io.StringIO()
        writer = csv.writer(pose_csv)
        writer.writerow(
            ("frame_id", "player_id", "keypoint_id", "x", "y", "confidence", "body_scale")
        )
        for frame in range(1, 161):
            burst = max(0, min(frame - 90, 30)) * 6.0
            for player in (0, 1):
                for keypoint in (11, 12):
                    writer.writerow((frame, player, keypoint, 100 + burst, 200, 0.9, 100))
        pose_csv.seek(0)

        with mock.patch.object(Path, "open", return_value=pose_csv):
            start, end, metadata = MODULE._select_depth_window(Path("pose.csv"), 160, 50)

        self.assertLessEqual(start, 91)
        self.assertGreaterEqual(end, 120)
        self.assertEqual(end - start + 1, 50)
        self.assertEqual(metadata["method"], "highest_robust_normalized_pelvis_motion")

    def test_zero_depth_window_requests_full_video(self) -> None:
        start, end, metadata = MODULE._select_depth_window(Path("unused.csv"), 120, 0)
        self.assertEqual((start, end), (1, 120))
        self.assertEqual(metadata["method"], "full_video")

    def test_detector_focal_is_used_to_validate_court_calibration(self) -> None:
        world = [[-3.05, -6.7], [3.05, -6.7], [3.05, 6.7], [-3.05, 6.7]]
        image = [[-1530.861, 834.133], [-672.072, 1596.284], [883.803, 472.928], [490.288, 469.263]]
        payload = {
            "court_mapping_schema": 2,
            "width": 1280,
            "height": 800,
            "frames": [
                {
                    "frame_index": 10,
                    "pole_mapped_court": {
                        "world_corners": world,
                        "image_corners": image,
                        "diagnostics": {"approximate_camera_focal_px": 998.4},
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "court.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(MODULE._court_calibration_valid(path, 24))


if __name__ == "__main__":
    unittest.main()
