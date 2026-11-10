from __future__ import annotations

import unittest

import numpy as np

from src.pose_biomechanics.foot_contact import (
    FootContactConfig,
    _project_correction_and_result_steps,
    refine_foot_contacts,
)


class FootContactTests(unittest.TestCase):
    def test_foot_lock_cannot_reintroduce_root_speed_spike(self) -> None:
        fps = 30.0
        root = np.zeros((20, 2), dtype=np.float64)
        root[:, 0] = np.arange(20) * (8.0 / fps)
        correction = np.zeros_like(root)
        correction[1::2, 0] = -0.25
        correction[2::2, 0] = 0.25

        limited = _project_correction_and_result_steps(
            root,
            correction,
            max_correction_norm=0.75,
            max_correction_step=8.0 / fps,
            max_result_step=8.0 / fps,
        )

        corrected_speed = np.linalg.norm(np.diff(root + limited, axis=0), axis=1) * fps
        self.assertLessEqual(float(corrected_speed.max()), 8.0 + 1e-5)
        self.assertLessEqual(float(np.linalg.norm(limited, axis=1).max()), 0.75 + 1e-5)

    def _synthetic(self, frames: int = 40) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
        vertices = np.zeros((frames, 8, 3), dtype=np.float32)
        left = np.arange(0, 4)
        right = np.arange(4, 8)
        vertices[:, left, 0] = -0.2
        vertices[:, right, 0] = 0.2
        vertices[:, :, 2] = 0.18
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 2] = 1.0
        return vertices, root, (left, right)

    def test_stationary_feet_are_grounded_and_stop_sliding(self) -> None:
        vertices, root, soles = self._synthetic()
        drift = np.linspace(0.0, 0.40, len(vertices))
        vertices[:, :, 1] += drift[:, None]
        root[:, 1] += drift
        confidence = np.ones((len(vertices), 2), dtype=np.float32)

        result = refine_foot_contacts(vertices, root, confidence, soles, fps=30.0)
        before = result.metrics["stationary_foot_xy_step_before_m"]["median"]
        after = result.metrics["stationary_foot_xy_step_after_m"]["median"]
        floor_after = result.metrics["support_sole_abs_height_after_m"]["median"]

        self.assertLess(after, before * 0.35)
        self.assertLess(floor_after, 0.01)
        np.testing.assert_allclose(
            result.corrected_vertices - result.corrected_root[:, None, :],
            vertices - root[:, None, :],
            atol=1e-6,
        )

    def test_airborne_sequence_is_not_forced_to_floor(self) -> None:
        vertices, root, soles = self._synthetic()
        confidence = np.zeros((len(vertices), 2), dtype=np.float32)
        result = refine_foot_contacts(vertices, root, confidence, soles, fps=30.0)
        np.testing.assert_allclose(result.correction_xyz_m, 0.0, atol=1e-6)

    def test_leg_motion_disables_false_contact_lock(self) -> None:
        vertices, root, soles = self._synthetic()
        vertices[:, soles[0], 0] += np.linspace(0.0, 1.5, len(vertices))[:, None]
        confidence = np.ones((len(vertices), 2), dtype=np.float32)
        result = refine_foot_contacts(
            vertices,
            root,
            confidence,
            soles,
            fps=30.0,
            config=FootContactConfig(stationary_local_speed_mps=0.5),
        )
        self.assertLess(float(result.stationary_scores[:, 0].max()), 0.1)

    def test_elevated_static_foot_is_not_treated_as_support(self) -> None:
        vertices, root, soles = self._synthetic()
        vertices[:, soles[1], 2] += 0.25
        confidence = np.ones((len(vertices), 2), dtype=np.float32)

        result = refine_foot_contacts(vertices, root, confidence, soles, fps=30.0)

        self.assertGreater(float(np.median(result.stationary_scores[:, 0])), 0.95)
        self.assertLess(float(np.max(result.stationary_scores[:, 1])), 0.02)
        self.assertTrue(np.all(result.support_foot == 0))

    def test_final_rigid_translation_stays_inside_court_margin(self) -> None:
        vertices, root, soles = self._synthetic()
        vertices[:, :, 0] += 3.20
        root[:, 0] += 3.20
        confidence = np.ones((len(vertices), 2), dtype=np.float32)

        result = refine_foot_contacts(
            vertices,
            root,
            confidence,
            soles,
            fps=30.0,
            court_half_extents_m=np.array([3.15, 6.80]),
        )

        self.assertLessEqual(float(np.abs(result.corrected_root[:, 0]).max()), 3.15 + 1e-6)
        np.testing.assert_allclose(
            result.corrected_vertices - result.corrected_root[:, None, :],
            vertices - root[:, None, :],
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
