"""Automatic badminton-court line detection utilities."""

from .detector import CourtDetectionResult, CourtDetectorConfig, CourtLine, CourtLineDetector
from .line_mapper import LineMappingConfig, score_court_homography, select_line_guided_court
from .pole_mapper import (
    PoleCandidate,
    PoleMappedCourt,
    annotate_pole_guided_court,
    detect_pole_candidates,
    recompute_pnp_ground_mapping,
    select_pole_guided_court,
)
from .player_pole_geometry import PLAYER_POLE_FIELDS, compute_player_pole_geometry
from .target_selector import SelectedCourt

__all__ = [
    "CourtDetectionResult",
    "CourtDetectorConfig",
    "CourtLine",
    "CourtLineDetector",
    "LineMappingConfig",
    "score_court_homography",
    "select_line_guided_court",
    "SelectedCourt",
    "PoleCandidate",
    "PoleMappedCourt",
    "annotate_pole_guided_court",
    "detect_pole_candidates",
    "recompute_pnp_ground_mapping",
    "select_pole_guided_court",
    "PLAYER_POLE_FIELDS",
    "compute_player_pole_geometry",
]
