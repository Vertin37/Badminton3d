"""Robust temporal filtering for tracked 2-D pose keypoints.

This module intentionally sits after :mod:`person_tracker`: every filter
state is keyed by ``player_id`` and ``keypoint_id``.  It never changes a
player's identity and it does not require RTMLib, ONNX Runtime, OpenCV, or a
GPU.

The default filter is a One Euro filter.  Before a point reaches that filter,
low-confidence observations and isolated, implausible spikes are withheld.
Short gaps are filled only when they are bounded and small; long gaps remain
empty instead of being fabricated indefinitely.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


# COCO body indices used by PersonTracker.  These points provide a useful
# scale reference but are filtered independently like every other keypoint.
BODY_KEYPOINTS = frozenset({5, 6, 11, 12, 13, 14, 15, 16})

# WholeBody wrists and hand landmarks can move rapidly during a badminton
# swing.  They get more responsive One Euro parameters and a looser spike
# gate than the torso/leg points.
FAST_KEYPOINTS = frozenset({9, 10}) | frozenset(range(91, 133))


@dataclass(frozen=True)
class OneEuroParameters:
    """Parameters for one keypoint family.

    ``min_cutoff`` is the baseline smoothing cutoff in Hz.  ``beta`` raises
    the cutoff when the estimated speed increases, which preserves fast
    movement.  ``d_cutoff`` controls smoothing of the speed estimate.
    """

    min_cutoff: float
    beta: float
    d_cutoff: float = 1.0


@dataclass(frozen=True)
class TemporalFilterConfig:
    """Tunable settings for the robust One Euro pose filter."""

    fps: float = 25.0
    confidence_threshold: float = 0.35
    max_interpolation_gap: int = 3
    max_hold_gap: int = 2
    outlier_lookahead: int = 2
    stable_max_normalized_jump: float = 1.25
    fast_max_normalized_jump: float = 2.50
    stable_spike_deviation: float = 0.70
    fast_spike_deviation: float = 1.20
    stable_parameters: OneEuroParameters = OneEuroParameters(
        min_cutoff=1.10,
        beta=0.018,
        d_cutoff=1.0,
    )
    default_parameters: OneEuroParameters = OneEuroParameters(
        min_cutoff=1.30,
        beta=0.025,
        d_cutoff=1.0,
    )
    fast_parameters: OneEuroParameters = OneEuroParameters(
        min_cutoff=1.70,
        beta=0.040,
        d_cutoff=1.0,
    )

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 <= self.confidence_threshold <= 1.5:
            raise ValueError("confidence_threshold must be between 0 and 1.5")
        if self.max_interpolation_gap < 0 or self.max_hold_gap < 0:
            raise ValueError("gap limits cannot be negative")
        if self.outlier_lookahead < 1:
            raise ValueError("outlier_lookahead must be at least 1")

    def parameters_for(self, keypoint_id: int) -> OneEuroParameters:
        if keypoint_id in BODY_KEYPOINTS:
            return self.stable_parameters
        if keypoint_id in FAST_KEYPOINTS:
            return self.fast_parameters
        return self.default_parameters

    def max_jump_for(self, keypoint_id: int) -> float:
        if keypoint_id in BODY_KEYPOINTS:
            return self.stable_max_normalized_jump
        if keypoint_id in FAST_KEYPOINTS:
            return self.fast_max_normalized_jump
        return (self.stable_max_normalized_jump + self.fast_max_normalized_jump) / 2.0

    def spike_deviation_for(self, keypoint_id: int) -> float:
        if keypoint_id in BODY_KEYPOINTS:
            return self.stable_spike_deviation
        if keypoint_id in FAST_KEYPOINTS:
            return self.fast_spike_deviation
        return (self.stable_spike_deviation + self.fast_spike_deviation) / 2.0


class OneEuroFilter2D:
    """A two-dimensional One Euro filter with independent x/y channels."""

    def __init__(self, parameters: OneEuroParameters, fps: float) -> None:
        self.parameters = parameters
        self.fps = fps
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev = np.zeros(2, dtype=float)
        self._timestamp_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray:
        cutoff_array = np.asarray(cutoff, dtype=float)
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff_array, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = np.zeros(2, dtype=float)
        self._timestamp_prev = None

    def apply(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        value_array = np.asarray(value, dtype=float)
        if value_array.shape != (2,) or not np.isfinite(value_array).all():
            raise ValueError("OneEuroFilter2D expects a finite (2,) value")

        if self._x_prev is None:
            self._x_prev = value_array.copy()
            self._timestamp_prev = float(timestamp)
            return value_array.copy()

        if self._timestamp_prev is None:
            dt = 1.0 / self.fps
        else:
            dt = float(timestamp) - self._timestamp_prev
            if dt <= 0:
                dt = 1.0 / self.fps
        frequency = 1.0 / dt

        derivative = (value_array - self._x_prev) * frequency
        derivative_alpha = self._alpha(self.parameters.d_cutoff, dt)
        derivative_hat = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self._dx_prev
        )
        cutoff = self.parameters.min_cutoff + self.parameters.beta * np.abs(
            derivative_hat
        )
        value_alpha = self._alpha(cutoff, dt)
        value_hat = value_alpha * value_array + (1.0 - value_alpha) * self._x_prev

        self._x_prev = value_hat
        self._dx_prev = derivative_hat
        self._timestamp_prev = float(timestamp)
        return value_hat.copy()


@dataclass
class PoseCsvArrays:
    frame_ids: np.ndarray
    player_ids: np.ndarray
    keypoint_ids: np.ndarray
    raw_xy: np.ndarray
    scores: np.ndarray
    input_row_count: int


@dataclass
class TemporalFilterResult:
    filtered_xy: np.ndarray
    status: np.ndarray
    raw_valid: np.ndarray
    outlier_rejected: np.ndarray
    filter_applied: np.ndarray
    body_scales: np.ndarray
    stats: dict[str, Any]


def _float_or_nan(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return math.nan
    return float(value)


def load_tracked_pose_csv(csv_path: Path) -> PoseCsvArrays:
    """Load a fixed-ID CSV into dense frame/player/keypoint arrays.

    ``player_id`` is preferred, while ``person_id`` remains accepted for
    compatibility with older fixed-ID exports.  The input should already be
    the output of the identity tracker; this function does not assign IDs.
    """

    records: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    frame_values: set[int] = set()
    player_values: set[int] = set()
    keypoint_values: set[int] = set()

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = set(reader.fieldnames or [])
        if "frame_id" not in fields or "keypoint_id" not in fields:
            raise ValueError("tracked pose CSV must contain frame_id and keypoint_id")
        player_column = "player_id" if "player_id" in fields else "person_id"
        if player_column not in fields:
            raise ValueError("tracked pose CSV must contain player_id or person_id")

        input_row_count = 0
        for row in reader:
            frame_id = int(row["frame_id"])
            player_id = int(row[player_column])
            keypoint_id = int(row["keypoint_id"])
            x = _float_or_nan(row.get("x"))
            y = _float_or_nan(row.get("y"))
            confidence = _float_or_nan(row.get("confidence", 0.0))
            records[(frame_id, player_id, keypoint_id)] = (x, y, confidence)
            frame_values.add(frame_id)
            player_values.add(player_id)
            keypoint_values.add(keypoint_id)
            input_row_count += 1

    if not records:
        raise ValueError(f"tracked pose CSV has no data: {csv_path}")

    first_frame = min(frame_values)
    last_frame = max(frame_values)
    frame_ids = np.arange(first_frame, last_frame + 1, dtype=int)
    player_ids = np.asarray(sorted(player_values), dtype=int)
    max_keypoint_id = max(keypoint_values)
    keypoint_ids = np.arange(max_keypoint_id + 1, dtype=int)

    player_index = {int(value): index for index, value in enumerate(player_ids)}
    frame_index = {int(value): index for index, value in enumerate(frame_ids)}
    raw_xy = np.full(
        (len(frame_ids), len(player_ids), len(keypoint_ids), 2),
        math.nan,
        dtype=float,
    )
    scores = np.zeros(
        (len(frame_ids), len(player_ids), len(keypoint_ids)),
        dtype=float,
    )
    for (frame_id, player_id, keypoint_id), (x, y, confidence) in records.items():
        raw_xy[
            frame_index[frame_id],
            player_index[player_id],
            keypoint_id,
        ] = (x, y)
        scores[frame_index[frame_id], player_index[player_id], keypoint_id] = (
            0.0 if not math.isfinite(confidence) else confidence
        )

    return PoseCsvArrays(
        frame_ids=frame_ids,
        player_ids=player_ids,
        keypoint_ids=keypoint_ids,
        raw_xy=raw_xy,
        scores=scores,
        input_row_count=input_row_count,
    )


def _compute_body_scales(
    raw_xy: np.ndarray,
    raw_valid: np.ndarray,
) -> np.ndarray:
    frame_count, player_count, keypoint_count, _ = raw_xy.shape
    scales = np.full((frame_count, player_count), math.nan, dtype=float)
    for frame_index in range(frame_count):
        for player_index in range(player_count):
            indices = [
                keypoint_id
                for keypoint_id in BODY_KEYPOINTS
                if keypoint_id < keypoint_count
                and raw_valid[frame_index, player_index, keypoint_id]
            ]
            if len(indices) < 2:
                continue
            points = raw_xy[frame_index, player_index, indices]
            extent = np.ptp(points, axis=0)
            scale = float(np.hypot(extent[0], extent[1]))
            if math.isfinite(scale) and scale > 1.0:
                scales[frame_index, player_index] = scale

    # A brief body-point miss should not disable normalized jump checks.  Use
    # linear interpolation inside the observed range and edge holding outside
    # it; if a player's body scale is never available, use a conservative
    # global median from the other player/frames.
    finite_scales = scales[np.isfinite(scales)]
    fallback = float(np.median(finite_scales)) if finite_scales.size else 100.0
    frame_axis = np.arange(frame_count, dtype=float)
    for player_index in range(player_count):
        known = np.where(np.isfinite(scales[:, player_index]))[0]
        if known.size == 0:
            scales[:, player_index] = fallback
            continue
        scales[:, player_index] = np.interp(
            frame_axis,
            known.astype(float),
            scales[known, player_index],
        )
    return np.maximum(scales, 1.0)


def _remove_isolated_spikes(
    raw_xy: np.ndarray,
    raw_valid: np.ndarray,
    body_scales: np.ndarray,
    config: TemporalFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject only large points that disagree with both nearby observations."""

    clean_valid = raw_valid.copy()
    rejected = np.zeros(raw_valid.shape, dtype=bool)
    frame_count, player_count, keypoint_count, _ = raw_xy.shape

    for player_index in range(player_count):
        for keypoint_id in range(keypoint_count):
            kept_indices: list[int] = []
            for frame_index in range(frame_count):
                if not raw_valid[frame_index, player_index, keypoint_id]:
                    continue

                if kept_indices:
                    previous_index = kept_indices[-1]
                    elapsed_frames = frame_index - previous_index
                    step = float(
                        np.linalg.norm(
                            raw_xy[frame_index, player_index, keypoint_id]
                            - raw_xy[previous_index, player_index, keypoint_id]
                        )
                        / max(body_scales[frame_index, player_index], 1.0)
                        / max(elapsed_frames, 1)
                    )
                    next_candidates = np.where(
                        raw_valid[frame_index + 1 : frame_index + 1 + config.outlier_lookahead,
                                  player_index, keypoint_id]
                    )[0]
                    if (
                        step > config.max_jump_for(keypoint_id)
                        and next_candidates.size
                    ):
                        next_index = frame_index + 1 + int(next_candidates[0])
                        total_span = next_index - previous_index
                        expected = raw_xy[
                            previous_index, player_index, keypoint_id
                        ] + (
                            raw_xy[next_index, player_index, keypoint_id]
                            - raw_xy[previous_index, player_index, keypoint_id]
                        ) * (elapsed_frames / max(total_span, 1))
                        deviation = float(
                            np.linalg.norm(
                                raw_xy[frame_index, player_index, keypoint_id]
                                - expected
                            )
                            / max(body_scales[frame_index, player_index], 1.0)
                        )
                        if deviation > config.spike_deviation_for(keypoint_id):
                            clean_valid[frame_index, player_index, keypoint_id] = False
                            rejected[frame_index, player_index, keypoint_id] = True
                            continue

                kept_indices.append(frame_index)

    return clean_valid, rejected


def _fill_short_gaps(
    raw_xy: np.ndarray,
    clean_valid: np.ndarray,
    rejected: np.ndarray,
    config: TemporalFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count, player_count, keypoint_count, _ = raw_xy.shape
    filled = np.full_like(raw_xy, math.nan, dtype=float)
    status = np.full((frame_count, player_count, keypoint_count), "missing", dtype=object)
    finite_raw = np.isfinite(raw_xy).all(axis=-1)
    status[finite_raw] = "low_confidence"
    status[clean_valid] = "observed"
    status[rejected] = "rejected_outlier"
    filled[clean_valid] = raw_xy[clean_valid]

    for player_index in range(player_count):
        for keypoint_id in range(keypoint_count):
            observed = np.where(clean_valid[:, player_index, keypoint_id])[0]
            if observed.size == 0:
                continue

            first = int(observed[0])
            if first > 0 and first <= config.max_hold_gap:
                filled[:first, player_index, keypoint_id] = raw_xy[
                    first, player_index, keypoint_id
                ]
                status[:first, player_index, keypoint_id] = "held_next"

            for left, right in zip(observed[:-1], observed[1:]):
                gap = int(right - left - 1)
                if gap <= 0 or gap > config.max_interpolation_gap:
                    continue
                left_value = raw_xy[left, player_index, keypoint_id]
                right_value = raw_xy[right, player_index, keypoint_id]
                for offset in range(1, gap + 1):
                    fraction = offset / float(gap + 1)
                    frame_index = int(left + offset)
                    filled[frame_index, player_index, keypoint_id] = (
                        left_value * (1.0 - fraction) + right_value * fraction
                    )
                    status[frame_index, player_index, keypoint_id] = "interpolated"

            last = int(observed[-1])
            trailing_gap = frame_count - last - 1
            if trailing_gap > 0 and trailing_gap <= config.max_hold_gap:
                filled[last + 1 :, player_index, keypoint_id] = raw_xy[
                    last, player_index, keypoint_id
                ]
                status[last + 1 :, player_index, keypoint_id] = "held_previous"

    return filled, status


def filter_pose_arrays(
    raw_xy: np.ndarray,
    scores: np.ndarray,
    config: Optional[TemporalFilterConfig] = None,
) -> TemporalFilterResult:
    """Filter dense ``[frame, player, keypoint, xy]`` pose arrays."""

    config = config or TemporalFilterConfig()
    raw_xy = np.asarray(raw_xy, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if raw_xy.ndim != 4 or raw_xy.shape[-1] != 2:
        raise ValueError("raw_xy must have shape (frames, players, keypoints, 2)")
    if scores.shape != raw_xy.shape[:-1]:
        raise ValueError("scores must have shape (frames, players, keypoints)")

    raw_valid = np.isfinite(raw_xy).all(axis=-1) & (
        scores >= config.confidence_threshold
    )
    body_scales = _compute_body_scales(raw_xy, raw_valid)
    clean_valid, rejected = _remove_isolated_spikes(
        raw_xy,
        raw_valid,
        body_scales,
        config,
    )
    filter_input, status = _fill_short_gaps(
        raw_xy,
        clean_valid,
        rejected,
        config,
    )

    frame_count, player_count, keypoint_count, _ = raw_xy.shape
    filtered_xy = np.full_like(raw_xy, math.nan, dtype=float)
    filter_applied = np.zeros((frame_count, player_count, keypoint_count), dtype=bool)
    for player_index in range(player_count):
        for keypoint_id in range(keypoint_count):
            euro = OneEuroFilter2D(
                config.parameters_for(keypoint_id),
                fps=config.fps,
            )
            last_input_frame: Optional[int] = None
            for frame_index in range(frame_count):
                value = filter_input[frame_index, player_index, keypoint_id]
                if not np.isfinite(value).all():
                    continue
                if (
                    last_input_frame is not None
                    and frame_index - last_input_frame > config.max_hold_gap + 1
                ):
                    euro.reset()
                filtered_xy[frame_index, player_index, keypoint_id] = euro.apply(
                    value,
                    timestamp=frame_index / config.fps,
                )
                filter_applied[frame_index, player_index, keypoint_id] = True
                last_input_frame = frame_index

    stats: dict[str, Any] = {
        "processed_frames": int(frame_count),
        "player_count": int(player_count),
        "keypoint_count": int(keypoint_count),
        "input_valid_points": int(raw_valid.sum()),
        "rejected_outlier_points": int(rejected.sum()),
        "filter_input_points": int(clean_valid.sum()),
        "interpolated_points": int(np.count_nonzero(status == "interpolated")),
        "held_points": int(
            np.count_nonzero((status == "held_next") | (status == "held_previous"))
        ),
        "output_valid_points": int(filter_applied.sum()),
        "status_counts": {
            str(name): int(count) for name, count in Counter(status.ravel()).items()
        },
        "per_player": {},
        "config": asdict(config),
    }
    for player_index in range(player_count):
        total = frame_count * keypoint_count
        body_indices = [
            keypoint_id for keypoint_id in BODY_KEYPOINTS if keypoint_id < keypoint_count
        ]
        body_total = frame_count * len(body_indices)
        stats["per_player"][str(player_index)] = {
            "input_valid_points": int(raw_valid[:, player_index].sum()),
            "rejected_outlier_points": int(rejected[:, player_index].sum()),
            "interpolated_points": int(
                np.count_nonzero(status[:, player_index] == "interpolated")
            ),
            "held_points": int(
                np.count_nonzero(
                    (status[:, player_index] == "held_next")
                    | (status[:, player_index] == "held_previous")
                )
            ),
            "output_valid_points": int(filter_applied[:, player_index].sum()),
            "coverage": float(filter_applied[:, player_index].sum() / max(total, 1)),
            "body_coverage": float(
                filter_applied[:, player_index, body_indices].sum()
                / max(body_total, 1)
            ),
        }

    return TemporalFilterResult(
        filtered_xy=filtered_xy,
        status=status,
        raw_valid=raw_valid,
        outlier_rejected=rejected,
        filter_applied=filter_applied,
        body_scales=body_scales,
        stats=stats,
    )


FILTERED_CSV_FIELDS = [
    "frame_id",
    "player_id",
    "person_id",
    "keypoint_id",
    "x",
    "y",
    "confidence",
    "raw_x",
    "raw_y",
    "filter_status",
    "raw_valid",
    "outlier_rejected",
    "filter_applied",
    "body_scale",
]


def _csv_value(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else ""


def write_filtered_pose_csv(
    output_csv: Path,
    arrays: PoseCsvArrays,
    result: TemporalFilterResult,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(FILTERED_CSV_FIELDS)
        for frame_index, frame_id in enumerate(arrays.frame_ids):
            for player_index, player_id in enumerate(arrays.player_ids):
                for keypoint_id in arrays.keypoint_ids:
                    raw_value = arrays.raw_xy[frame_index, player_index, keypoint_id]
                    filtered_value = result.filtered_xy[
                        frame_index, player_index, keypoint_id
                    ]
                    writer.writerow(
                        [
                            int(frame_id),
                            int(player_id),
                            int(player_id),
                            int(keypoint_id),
                            _csv_value(filtered_value[0]),
                            _csv_value(filtered_value[1]),
                            float(
                                arrays.scores[frame_index, player_index, keypoint_id]
                            ),
                            _csv_value(raw_value[0]),
                            _csv_value(raw_value[1]),
                            str(
                                result.status[frame_index, player_index, keypoint_id]
                            ),
                            int(
                                result.raw_valid[frame_index, player_index, keypoint_id]
                            ),
                            int(
                                result.outlier_rejected[
                                    frame_index, player_index, keypoint_id
                                ]
                            ),
                            int(
                                result.filter_applied[
                                    frame_index, player_index, keypoint_id
                                ]
                            ),
                            float(result.body_scales[frame_index, player_index]),
                        ]
                    )


def filter_tracked_pose_csv(
    input_csv: Path,
    output_csv: Path,
    stats_json: Path,
    config: Optional[TemporalFilterConfig] = None,
) -> dict[str, Any]:
    """Filter a fixed-ID CSV and write a comparable stable CSV plus JSON stats."""

    arrays = load_tracked_pose_csv(input_csv)
    result = filter_pose_arrays(arrays.raw_xy, arrays.scores, config=config)
    result.stats["input_csv"] = str(Path(input_csv))
    result.stats["output_csv"] = str(Path(output_csv))
    result.stats["input_rows"] = int(arrays.input_row_count)
    result.stats["output_rows"] = int(
        len(arrays.frame_ids) * len(arrays.player_ids) * len(arrays.keypoint_ids)
    )
    write_filtered_pose_csv(output_csv, arrays, result)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    with Path(stats_json).open("w", encoding="utf-8") as json_file:
        json.dump(result.stats, json_file, ensure_ascii=False, indent=2)
    return result.stats
