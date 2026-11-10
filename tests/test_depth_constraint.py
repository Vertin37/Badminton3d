from __future__ import annotations

import unittest

import numpy as np

from src.pose_biomechanics.depth_constraint import (
    DepthConstraintConfig,
    camera_depth_delta_to_court_xy,
    constrain_camera_depth,
    discover_player_ids,
    trajectory_metrics,
)


COURT_TO_CAMERA_R = np.array(
    [
        [0.4629041231, 0.8855139964, -0.0398087307],
        [-0.3073480283, 0.1182186353, -0.9442253670],
        [0.8314186444, -0.4493209504, -0.3268848746],
    ],
    dtype=float,
)


class DepthConstraintTests(unittest.TestCase):
    def test_discovers_all_bundle_players(self) -> None:
        keys = {
            "p0_root_court",
            "p0_vertices_court",
            "p1_root_court",
            "p1_vertices_court",
            "p2_root_court",
            "faces",
        }
        self.assertEqual(discover_player_ids(keys), [0, 1])

    def test_camera_depth_correction_stays_horizontal(self) -> None:
        requested = np.array([-0.4, 0.0, 0.25], dtype=float)
        xy = camera_depth_delta_to_court_xy(requested, COURT_TO_CAMERA_R)
        court_delta = np.column_stack([xy, np.zeros(len(xy))])
        achieved_camera_z = (court_delta @ COURT_TO_CAMERA_R.T)[:, 2]

        np.testing.assert_allclose(achieved_camera_z, requested, atol=1e-9)
        np.testing.assert_array_equal(court_delta[:, 2], np.zeros(len(requested)))

    def test_depth_spike_is_reduced_without_flattening_jump_height(self) -> None:
        fps = 30.0
        frame_count = 90
        frames = np.arange(frame_count)
        baseline_camera_depth = np.full(frame_count, 14.0)
        baseline_camera_depth[42:46] += np.array([0.4, 1.25, 1.15, 0.35])
        observed_depth = 7.0 + 0.015 * np.sin(frames / 4.0)
        root_height = 0.45 + 0.42 * np.exp(-0.5 * np.square((frames - 44) / 6.0))
        quality = np.ones(frame_count)

        result = constrain_camera_depth(
            baseline_camera_depth,
            observed_depth,
            quality,
            root_height,
            fps,
            DepthConstraintConfig(max_correction_m=1.5, max_correction_speed_mps=5.0),
        )
        baseline_root = np.column_stack([np.zeros(frame_count), np.zeros(frame_count), root_height])
        xy = camera_depth_delta_to_court_xy(result.applied_camera_depth_delta_m, COURT_TO_CAMERA_R)
        corrected_root = baseline_root.copy()
        corrected_root[:, :2] += xy
        metrics = trajectory_metrics(
            baseline_root,
            corrected_root,
            baseline_camera_depth,
            result.corrected_camera_depth_m,
            result.aligned_target_camera_depth_m,
            fps,
        )

        self.assertLess(
            metrics["corrected"]["depth_consistency"]["max_one_frame_change_m"],
            metrics["baseline"]["depth_consistency"]["max_one_frame_change_m"],
        )
        self.assertTrue(metrics["vertical_motion_preservation"]["exactly_preserved"])
        self.assertGreater(float(np.max(result.jump_score)), 0.4)

    def test_sustained_depth_travel_is_not_flattened(self) -> None:
        fps = 30.0
        frame_count = 120
        observed_depth = np.linspace(6.0, 8.0, frame_count)
        baseline_camera_depth = 11.0 + 1.3 * (observed_depth - observed_depth[0])
        baseline_camera_depth[55:58] += np.array([0.45, 1.1, 0.4])
        root_height = np.full(frame_count, 0.5)

        result = constrain_camera_depth(
            baseline_camera_depth,
            observed_depth,
            np.ones(frame_count),
            root_height,
            fps,
            DepthConstraintConfig(max_correction_m=1.5, max_correction_speed_mps=5.0),
        )

        self.assertGreater(result.corrected_camera_depth_m[-1] - result.corrected_camera_depth_m[0], 2.0)
        self.assertGreater(np.corrcoef(result.corrected_camera_depth_m, result.aligned_target_camera_depth_m)[0, 1], 0.98)
        self.assertLess(
            np.max(np.abs(np.diff(result.corrected_camera_depth_m))),
            np.max(np.abs(np.diff(baseline_camera_depth))),
        )

    def test_isolated_depth_observation_outlier_is_downweighted(self) -> None:
        frame_count = 60
        observed_depth = np.full(frame_count, 6.0)
        observed_depth[30] = 30.0
        baseline_camera_depth = np.full(frame_count, 13.0)
        result = constrain_camera_depth(
            baseline_camera_depth,
            observed_depth,
            np.ones(frame_count),
            np.full(frame_count, 0.5),
            30.0,
        )

        self.assertTrue(bool(result.observation_outlier[30]))
        self.assertLess(abs(result.cleaned_observation_m[30] - 6.0), 0.1)

    def test_continuous_foot_contact_strengthens_gvhmr_prior(self) -> None:
        frame_count = 90
        observed_depth = np.linspace(6.0, 7.0, frame_count)
        baseline_camera_depth = 12.0 + 0.8 * (observed_depth - observed_depth[0])
        baseline_camera_depth[42:47] += np.array([0.25, 0.65, 0.9, 0.65, 0.25])
        root_height = 0.5 + 0.25 * np.exp(-0.5 * np.square((np.arange(frame_count) - 44) / 5.0))
        config = DepthConstraintConfig(max_correction_m=1.5, max_correction_speed_mps=5.0)

        airborne = constrain_camera_depth(
            baseline_camera_depth,
            observed_depth,
            np.ones(frame_count),
            root_height,
            30.0,
            config,
            foot_stationary_confidence=np.zeros(frame_count),
        )
        grounded = constrain_camera_depth(
            baseline_camera_depth,
            observed_depth,
            np.ones(frame_count),
            root_height,
            30.0,
            config,
            foot_stationary_confidence=np.ones(frame_count),
        )

        self.assertLessEqual(
            float(np.mean(np.abs(grounded.applied_camera_depth_delta_m))),
            float(np.mean(np.abs(airborne.applied_camera_depth_delta_m))),
        )
        self.assertLess(float(np.max(grounded.jump_score)), float(np.max(airborne.jump_score)))


if __name__ == "__main__":
    unittest.main()
