from __future__ import annotations

import unittest

import numpy as np

from src.pose_biomechanics.trajectory_fusion import (
    TrajectoryFusionConfig,
    fuse_court_trajectory,
    image_points_to_court_xy_at_height,
    project_court_points,
)


K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.94, -0.34117444],
        [0.0, 0.34117444, 0.94],
    ]
)
T = np.array([0.0, 0.0, 12.0])


class TrajectoryFusionTests(unittest.TestCase):
    def test_image_ray_round_trip_at_known_height(self) -> None:
        points = np.array([[-1.2, -2.0, 0.55], [0.4, 1.5, 0.8], [1.8, 3.0, 0.45]])
        pixels = project_court_points(points, K, R, T)
        recovered = image_points_to_court_xy_at_height(pixels, points[:, 2], K, R, T)
        np.testing.assert_allclose(recovered, points[:, :2], atol=1e-9)

    def test_single_frame_root_teleport_is_suppressed(self) -> None:
        frames = 90
        time = np.linspace(0.0, 1.0, frames)
        truth = np.column_stack([0.4 * time, -2.0 + 1.2 * time])
        height = np.full(frames, 0.55)
        xyz = np.column_stack([truth, height])
        pixels = project_court_points(xyz, K, R, T)
        raw = truth.copy()
        raw[44] += np.array([1.1, -0.9])
        ground = truth.copy()
        contact = np.ones(frames)

        result = fuse_court_trajectory(
            raw,
            height,
            pixels,
            np.ones(frames),
            ground,
            contact,
            K,
            R,
            T,
            30.0,
            6.1,
            13.4,
        )
        raw_max_step = np.max(np.linalg.norm(np.diff(raw, axis=0), axis=1))
        corrected_max_step = np.max(np.linalg.norm(np.diff(result.corrected_xy_m, axis=0), axis=1))

        self.assertLess(corrected_max_step, 0.12)
        self.assertLess(corrected_max_step, raw_max_step * 0.2)
        self.assertLess(float(result.raw_velocity_quality[44]), 0.2)

    def test_sustained_court_motion_is_retained(self) -> None:
        frames = 120
        truth = np.column_stack([np.linspace(-1.5, 1.5, frames), np.linspace(-3.0, 2.0, frames)])
        height = np.full(frames, 0.6)
        pixels = project_court_points(np.column_stack([truth, height]), K, R, T)
        result = fuse_court_trajectory(
            truth.copy(),
            height,
            pixels,
            np.ones(frames),
            np.full_like(truth, np.nan),
            np.zeros(frames),
            K,
            R,
            T,
            30.0,
            6.1,
            13.4,
        )

        displacement = result.corrected_xy_m[-1] - result.corrected_xy_m[0]
        np.testing.assert_allclose(displacement, truth[-1] - truth[0], atol=0.12)
        self.assertGreater(np.corrcoef(result.corrected_xy_m[:, 1], truth[:, 1])[0, 1], 0.999)

    def test_jump_height_changes_ray_target_without_changing_output_z(self) -> None:
        frames = 80
        xy = np.column_stack([np.full(frames, 0.4), np.full(frames, 1.5)])
        frame = np.arange(frames)
        height = 0.55 + 0.45 * np.exp(-0.5 * np.square((frame - 40) / 6.0))
        pixels = project_court_points(np.column_stack([xy, height]), K, R, T)
        result = fuse_court_trajectory(
            xy.copy(),
            height,
            pixels,
            np.ones(frames),
            np.full_like(xy, np.nan),
            np.zeros(frames),
            K,
            R,
            T,
            30.0,
            6.1,
            13.4,
            config=TrajectoryFusionConfig(max_correction_m=1.0),
        )

        np.testing.assert_allclose(result.aligned_pelvis_target_xy_m, xy, atol=1e-8)
        np.testing.assert_allclose(result.corrected_xy_m, xy, atol=1e-6)

    def test_wrong_ground_measurement_cluster_is_rejected(self) -> None:
        frames = 100
        truth = np.column_stack([np.linspace(-0.4, 0.4, frames), np.linspace(1.0, 2.0, frames)])
        height = np.full(frames, 0.55)
        pixels = project_court_points(np.column_stack([truth, height]), K, R, T)
        ground = truth.copy()
        ground[40:55] += np.array([2.5, -2.0])
        result = fuse_court_trajectory(
            truth.copy(),
            height,
            pixels,
            np.ones(frames),
            ground,
            np.ones(frames),
            K,
            R,
            T,
            30.0,
            6.1,
            13.4,
        )

        self.assertLess(float(np.max(np.linalg.norm(result.corrected_xy_m - truth, axis=1))), 0.15)
        self.assertLess(float(np.median(result.ground_quality[42:53])), 0.05)


if __name__ == "__main__":
    unittest.main()
