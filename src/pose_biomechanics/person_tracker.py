"""Cross-frame identity tracking for two whole-body pose detections.

The pose detector returns an unordered list on every frame.  This module turns
that list into two persistent player slots without using the player's left or
right image position as an identity cue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Optional, Sequence

import numpy as np


# COCO WholeBody starts with the standard 17 body keypoints.  These are the
# torso/leg points that are normally more stable than wrists and fingers.
BODY_KEYPOINT_WEIGHTS = {
    5: 1.25,   # left shoulder
    6: 1.25,   # right shoulder
    11: 1.50,  # left hip
    12: 1.50,  # right hip
    13: 1.00,  # left knee
    14: 1.00,  # right knee
    15: 0.75,  # left ankle
    16: 0.75,  # right ankle
}


@dataclass
class PersonDetection:
    """One detector output plus the features used by the tracker."""

    keypoints: np.ndarray
    scores: np.ndarray
    source_detection_id: int
    center: np.ndarray
    scale: float
    quality: float
    visible_body_keypoints: int


@dataclass
class TrackingResult:
    """The result for one persistent player slot on one frame."""

    player_id: int
    detection: Optional[PersonDetection]
    matched: bool
    missed_frames: int
    predicted_center: np.ndarray
    match_cost: Optional[float]
    source_detection_id: Optional[int]


@dataclass
class _TrackState:
    player_id: int
    initialized: bool = False
    last_center: Optional[np.ndarray] = None
    predicted_center: Optional[np.ndarray] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    last_scale: float = 1.0
    last_quality: float = 0.0
    last_keypoints: Optional[np.ndarray] = None
    last_scores: Optional[np.ndarray] = None
    missed_frames: int = 0
    hits: int = 0
    age: int = 0
    latest_match_cost: Optional[float] = None


class PersonTracker:
    """Maintain persistent IDs for a fixed number of people.

    Matching uses all of the following signals:

    * a velocity prediction from the previous observations;
    * distance between weighted body centers;
    * normalized body-keypoint shape continuity;
    * a soft penalty for an implausible scale change.

    Unmatched tracks are kept alive for ``max_missed_frames``.  An unmatched
    track produces no pose rows for that frame, so downstream code can
    distinguish a brief miss from a newly assigned identity.
    """

    def __init__(
        self,
        num_players: int = 2,
        confidence_threshold: float = 0.30,
        min_body_keypoints: int = 4,
        max_missed_frames: int = 15,
        max_match_distance: float = 2.5,
    ) -> None:
        if num_players < 1:
            raise ValueError("num_players must be at least 1")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")

        self.num_players = num_players
        self.confidence_threshold = confidence_threshold
        self.min_body_keypoints = min_body_keypoints
        self.max_missed_frames = max_missed_frames
        self.max_match_distance = max_match_distance
        self.frame_index = 0
        self._tracks = [_TrackState(player_id=i) for i in range(num_players)]

    @property
    def tracks(self) -> Sequence[_TrackState]:
        """Read-only-by-convention state useful for diagnostics."""

        return tuple(self._tracks)

    def reset(self) -> None:
        self.frame_index = 0
        self._tracks = [_TrackState(player_id=i) for i in range(self.num_players)]

    def update(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> list[TrackingResult]:
        """Consume one detector frame and return one result per player slot."""

        self.frame_index += 1
        detections = self._build_detections(keypoints, scores)
        initialized_tracks = [track for track in self._tracks if track.initialized]
        assignments: dict[int, Optional[int]] = {}
        used_detection_ids: set[int] = set()

        if initialized_tracks and detections:
            assignment_list = self._best_assignment(initialized_tracks, detections)
            for track, detection_index in zip(initialized_tracks, assignment_list):
                assignments[track.player_id] = detection_index
                if detection_index is not None:
                    used_detection_ids.add(detection_index)

        for track in initialized_tracks:
            detection_index = assignments.get(track.player_id)
            if detection_index is None:
                track.latest_match_cost = None
                self._mark_missed(track)
                continue

            detection = detections[detection_index]
            cost = self._match_cost(track, detection)
            self._observe(track, detection)
            assignments[track.player_id] = detection_index
            track.latest_match_cost = cost

        # Empty slots are filled from the strongest unused candidates.  This
        # covers a one-person first frame and a player entering after startup.
        empty_tracks = [track for track in self._tracks if not track.initialized]
        unused_indices = [
            index for index in range(len(detections)) if index not in used_detection_ids
        ]
        for track in empty_tracks:
            if not unused_indices:
                break
            detection_index = max(
                unused_indices,
                key=lambda index: self._selection_score(detections[index]),
            )
            unused_indices.remove(detection_index)
            detection = detections[detection_index]
            self._observe(track, detection)
            assignments[track.player_id] = detection_index
            track.latest_match_cost = None

        results: list[TrackingResult] = []
        for track in self._tracks:
            detection_index = assignments.get(track.player_id)
            detection = (
                detections[detection_index]
                if detection_index is not None
                else None
            )
            predicted_center = (
                track.predicted_center.copy()
                if track.predicted_center is not None
                else np.array([math.nan, math.nan], dtype=float)
            )
            results.append(
                TrackingResult(
                    player_id=track.player_id,
                    detection=detection,
                    matched=detection is not None,
                    missed_frames=track.missed_frames,
                    predicted_center=predicted_center,
                    match_cost=track.latest_match_cost,
                    source_detection_id=(
                        detection.source_detection_id if detection is not None else None
                    ),
                )
            )

        return results

    def _build_detections(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> list[PersonDetection]:
        keypoints_array = np.asarray(keypoints, dtype=float)
        scores_array = np.asarray(scores, dtype=float)

        if keypoints_array.size == 0:
            return []
        if keypoints_array.ndim == 2:
            keypoints_array = keypoints_array[None, ...]
        if scores_array.ndim == 1:
            scores_array = scores_array[None, ...]
        if keypoints_array.ndim != 3 or keypoints_array.shape[2] < 2:
            raise ValueError("keypoints must have shape (people, keypoints, 2)")
        if scores_array.ndim != 2:
            raise ValueError("scores must have shape (people, keypoints)")

        detections: list[PersonDetection] = []
        person_count = min(keypoints_array.shape[0], scores_array.shape[0])
        for source_id in range(person_count):
            person_keypoints = keypoints_array[source_id]
            person_scores = scores_array[source_id]
            detection = self._summarize_detection(
                person_keypoints,
                person_scores,
                source_id,
            )
            if detection is not None:
                detections.append(detection)
        return detections

    def _summarize_detection(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        source_detection_id: int,
    ) -> Optional[PersonDetection]:
        usable_indices: list[int] = []
        usable_weights: list[float] = []
        for index, weight in BODY_KEYPOINT_WEIGHTS.items():
            if index >= len(keypoints) or index >= len(scores):
                continue
            point = keypoints[index, :2]
            score = float(scores[index])
            if (
                score >= self.confidence_threshold
                and np.isfinite(point).all()
            ):
                usable_indices.append(index)
                usable_weights.append(weight)

        if len(usable_indices) < self.min_body_keypoints:
            return None

        points = np.asarray([keypoints[index, :2] for index in usable_indices])
        weights = np.asarray(usable_weights, dtype=float)
        center = np.average(points, axis=0, weights=weights)
        extent = np.ptp(points, axis=0)
        scale = max(float(np.hypot(extent[0], extent[1])), 1.0)
        quality = float(
            np.average(
                [float(scores[index]) for index in usable_indices],
                weights=weights,
            )
        )

        return PersonDetection(
            keypoints=np.asarray(keypoints, dtype=float).copy(),
            scores=np.asarray(scores, dtype=float).copy(),
            source_detection_id=source_detection_id,
            center=np.asarray(center, dtype=float),
            scale=scale,
            quality=quality,
            visible_body_keypoints=len(usable_indices),
        )

    def _selection_score(self, detection: PersonDetection) -> float:
        # Scale is useful here because the video contains small background
        # detections.  Confidence only breaks ties; it does not define ID.
        return math.log1p(detection.scale) * (0.75 + 0.25 * detection.quality)

    def _best_assignment(
        self,
        tracks: Sequence[_TrackState],
        detections: Sequence[PersonDetection],
    ) -> list[Optional[int]]:
        """Find a minimum-cost one-to-one assignment for the two tracks."""

        if not tracks:
            return []

        costs = [
            [self._match_cost(track, detection) for detection in detections]
            for track in tracks
        ]
        unmatched_costs = [
            1.0 + min(0.35, track.missed_frames * 0.05) for track in tracks
        ]

        best_total = float("inf")
        best_assignment: list[Optional[int]] = [None] * len(tracks)

        def visit(
            track_index: int,
            used: set[int],
            current: list[Optional[int]],
            total: float,
        ) -> None:
            nonlocal best_total, best_assignment
            if total >= best_total:
                return
            if track_index == len(tracks):
                best_total = total
                best_assignment = current.copy()
                return

            # Leaving a track unmatched is preferable to a very distant
            # false-positive match.
            visit(
                track_index + 1,
                used,
                current + [None],
                total + unmatched_costs[track_index],
            )
            for detection_index, cost in enumerate(costs[track_index]):
                if detection_index in used or not math.isfinite(cost):
                    continue
                visit(
                    track_index + 1,
                    used | {detection_index},
                    current + [detection_index],
                    total + cost,
                )

        visit(0, set(), [], 0.0)
        return best_assignment

    def _match_cost(
        self,
        track: _TrackState,
        detection: PersonDetection,
    ) -> float:
        if not track.initialized or track.predicted_center is None:
            return float("inf")

        reference_scale = max(track.last_scale, detection.scale, 1.0)
        center_distance = float(
            np.linalg.norm(detection.center - track.predicted_center)
            / reference_scale
        )
        allowed_distance = self.max_match_distance + min(
            2.0,
            track.missed_frames * 0.45,
        )
        if center_distance > allowed_distance:
            return float("inf")

        shape_distance = self._shape_distance(track, detection)
        scale_distance = abs(math.log(max(detection.scale, 1.0) / max(track.last_scale, 1.0)))
        quality_penalty = max(0.0, track.last_quality - detection.quality)

        return (
            0.45 * center_distance
            + 0.35 * shape_distance
            + 0.15 * scale_distance
            + 0.05 * quality_penalty
        )

    def _shape_distance(
        self,
        track: _TrackState,
        detection: PersonDetection,
    ) -> float:
        if track.last_keypoints is None or track.last_center is None:
            return 0.0

        common_indices = [
            index
            for index in BODY_KEYPOINT_WEIGHTS
            if (
                index < len(track.last_keypoints)
                and index < len(track.last_scores)
                and index < len(detection.keypoints)
                and index < len(detection.scores)
                and track.last_scores[index] >= self.confidence_threshold
                and detection.scores[index] >= self.confidence_threshold
                and np.isfinite(track.last_keypoints[index, :2]).all()
                and np.isfinite(detection.keypoints[index, :2]).all()
            )
        ]
        if not common_indices:
            return 1.0

        old_scale = max(track.last_scale, 1.0)
        new_scale = max(detection.scale, 1.0)
        old_shape = np.asarray(
            [
                (track.last_keypoints[index, :2] - track.last_center) / old_scale
                for index in common_indices
            ]
        )
        new_shape = np.asarray(
            [
                (detection.keypoints[index, :2] - detection.center) / new_scale
                for index in common_indices
            ]
        )
        distances = np.linalg.norm(old_shape - new_shape, axis=1)
        return float(np.mean(distances))

    def _observe(self, track: _TrackState, detection: PersonDetection) -> None:
        if not track.initialized:
            track.initialized = True
            track.last_center = detection.center.copy()
            track.predicted_center = detection.center.copy()
            track.velocity = np.zeros(2, dtype=float)
            track.last_scale = detection.scale
            track.last_quality = detection.quality
            track.last_keypoints = detection.keypoints.copy()
            track.last_scores = detection.scores.copy()
            track.missed_frames = 0
            track.hits = 1
            track.age = 1
            return

        old_missed = track.missed_frames
        base_center = (
            track.predicted_center
            if track.predicted_center is not None
            else track.last_center
        )
        if base_center is not None:
            observed_step = (detection.center - base_center) / max(old_missed + 1, 1)
            track.velocity = 0.65 * track.velocity + 0.35 * observed_step

        track.last_center = detection.center.copy()
        track.predicted_center = detection.center.copy()
        track.last_scale = 0.70 * track.last_scale + 0.30 * detection.scale
        track.last_quality = 0.70 * track.last_quality + 0.30 * detection.quality
        track.last_keypoints = detection.keypoints.copy()
        track.last_scores = detection.scores.copy()
        track.missed_frames = 0
        track.hits += 1
        track.age += 1

    def _mark_missed(self, track: _TrackState) -> None:
        if not track.initialized:
            return
        track.missed_frames += 1
        track.age += 1
        if track.predicted_center is None:
            track.predicted_center = track.last_center.copy()
        else:
            track.predicted_center = track.predicted_center + track.velocity
        track.velocity *= 0.90
        if track.missed_frames > self.max_missed_frames:
            # A long disappearance cannot be safely extrapolated.  Retire the
            # slot so a later reappearance can be initialized explicitly,
            # while short gaps retain the original identity.
            track.initialized = False
            track.velocity = np.zeros(2, dtype=float)


def make_detection_arrays(
    detections: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of ``(keypoints, scores)`` pairs to detector arrays."""

    pairs = list(detections)
    if not pairs:
        return np.empty((0, 0, 2), dtype=float), np.empty((0, 0), dtype=float)
    return (
        np.stack([np.asarray(pair[0], dtype=float) for pair in pairs]),
        np.stack([np.asarray(pair[1], dtype=float) for pair in pairs]),
    )
