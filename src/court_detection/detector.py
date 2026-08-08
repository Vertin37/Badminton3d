"""CPU-first badminton-court line detection.

This module deliberately stops at image-space structure detection.  It does not
estimate a homography, camera pose, 3-D coordinates, or player locations.

The detector combines three weak cues that are useful on indoor badminton
footage:

* Canny edges and probabilistic Hough lines;
* bright, low-saturation support along a candidate line (white court paint);
* the court's two dominant projective direction families and spatial spread.

The result is intentionally inspectable: every retained segment includes its
pixel endpoints, angle, length, score, family, and heuristic role.  The roles
(`net_candidate`, `boundary_candidate`, and `service_candidate`) are hints for
the next calibration stage, not a claim that a full court model was solved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .target_selector import SelectedCourt, select_target_court


def _angular_distance(a: float, b: float) -> float:
    """Return the smallest distance between two unoriented line angles."""

    distance = abs(float(a) - float(b)) % 180.0
    return min(distance, 180.0 - distance)


def _line_angle(line: Sequence[int | float]) -> float:
    x1, y1, x2, y2 = map(float, line)
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _line_length(line: Sequence[int | float]) -> float:
    x1, y1, x2, y2 = map(float, line)
    return math.hypot(x2 - x1, y2 - y1)


def _line_midpoint(line: Sequence[int | float]) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, line)
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _sample_segment(line: Sequence[int | float], count: int) -> np.ndarray:
    """Return integer pixel coordinates sampled uniformly on a segment."""

    x1, y1, x2, y2 = map(float, line)
    count = max(2, int(count))
    xs = np.linspace(x1, x2, count).round().astype(np.int32)
    ys = np.linspace(y1, y2, count).round().astype(np.int32)
    return np.stack([xs, ys], axis=1)


@dataclass(frozen=True)
class CourtDetectorConfig:
    """Tunable settings for the first automatic detector."""

    canny_low: int = 40
    canny_high: int = 140
    blur_kernel: int = 5
    roi_top_ratio: float = 0.27
    hough_threshold: int = 35
    min_line_length_ratio: float = 0.055
    max_line_gap: int = 24
    white_saturation_max: int = 125
    white_value_min: int = 135
    min_white_support: float = 0.07
    floor_hue_min: int = 35
    floor_hue_max: int = 90
    floor_saturation_min: int = 40
    floor_value_min: int = 35
    min_floor_support: float = 0.34
    min_candidate_score: float = 0.16
    merge_angle_tolerance: float = 4.0
    merge_distance_tolerance: float = 13.0
    merge_gap_tolerance: float = 80.0
    max_output_lines: int = 36


@dataclass
class CourtLine:
    """One retained image-space court-line segment."""

    line_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    length: float
    angle_deg: float
    score: float
    white_support: float
    edge_support: float
    floor_support: float = 0.0
    family: str = "unassigned"
    role: str = "court_line"
    role_confidence: float = 0.0

    @property
    def segment(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def midpoint(self) -> tuple[float, float]:
        return _line_midpoint(self.segment)

    def as_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "length_px": round(float(self.length), 3),
            "angle_deg": round(float(self.angle_deg), 3),
            "score": round(float(self.score), 5),
            "white_support": round(float(self.white_support), 5),
            "edge_support": round(float(self.edge_support), 5),
            "floor_support": round(float(self.floor_support), 5),
            "family": self.family,
            "role": self.role,
            "role_confidence": round(float(self.role_confidence), 5),
        }


@dataclass
class CourtDetectionResult:
    """Detection output for a single video frame."""

    frame_index: int
    width: int
    height: int
    lines: list[CourtLine]
    edges: np.ndarray = field(repr=False)
    white_mask: np.ndarray = field(repr=False)
    target_court: SelectedCourt | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        if self.target_court is None:
            final_lines = self.lines
        else:
            boundary_ids = set(self.target_court.boundary_line_ids)
            final_lines = [line for line in self.lines if line.line_id in boundary_ids]
        target_boundary_segments: list[dict[str, Any]] | None = None
        if self.target_court is not None:
            vertices = self.target_court.vertices
            target_boundary_segments = []
            for boundary_index, start in enumerate(vertices):
                end = vertices[(boundary_index + 1) % len(vertices)]
                target_boundary_segments.append(
                    {
                        "boundary_index": boundary_index + 1,
                        "x1": round(float(start[0]), 3),
                        "y1": round(float(start[1]), 3),
                        "x2": round(float(end[0]), 3),
                        "y2": round(float(end[1]), 3),
                        "synthetic_or_extended": True,
                    }
                )
        return {
            "frame_index": self.frame_index,
            "width": self.width,
            "height": self.height,
            "line_count": len(final_lines),
            "lines": [line.as_dict() for line in final_lines],
            "candidate_line_count": len(self.lines),
            "candidate_lines": [line.as_dict() for line in self.lines],
            "target_court": None if self.target_court is None else self.target_court.as_dict(),
            "target_boundary_segments": target_boundary_segments,
            "diagnostics": self.diagnostics,
        }


class CourtLineDetector:
    """Detect and annotate likely court lines in a BGR image."""

    # BGR colors used for the overlay.  They are deliberately high contrast
    # against the green court and remain readable when the image is resized.
    _ROLE_COLORS = {
        "net_candidate": (255, 0, 255),
        "boundary_candidate": (0, 0, 255),
        "service_candidate": (0, 255, 255),
        "court_line": (0, 220, 0),
    }

    def __init__(self, config: CourtDetectorConfig | None = None) -> None:
        self.config = config or CourtDetectorConfig()

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        target_player_points: Sequence[Sequence[float]] | None = None,
        target_axis_angle_deg: float | None = None,
    ) -> CourtDetectionResult:
        """Detect court-like line segments from one BGR frame."""

        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a non-empty BGR image with shape HxWx3")

        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kernel = self.config.blur_kernel
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("blur_kernel must be an odd integer >= 3")
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)

        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)
        roi_top = int(round(height * self.config.roi_top_ratio))
        roi_mask = np.zeros_like(edges)
        roi_mask[roi_top:, :] = 255
        edges = cv2.bitwise_and(edges, roi_mask)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # White paint is bright and much less saturated than the green floor.
        # The mask is used as a support cue, not as the only detector, because
        # compression and glare can make court paint slightly green/yellow.
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.config.white_value_min], dtype=np.uint8),
            np.array([180, self.config.white_saturation_max, 255], dtype=np.uint8),
        )
        white_mask = cv2.bitwise_and(white_mask, roi_mask)
        floor_mask = cv2.inRange(
            hsv,
            np.array(
                [self.config.floor_hue_min, self.config.floor_saturation_min, self.config.floor_value_min],
                dtype=np.uint8,
            ),
            np.array([self.config.floor_hue_max, 255, 255], dtype=np.uint8),
        )
        floor_mask = cv2.bitwise_and(floor_mask, roi_mask)

        min_line_length = max(40, int(round(width * self.config.min_line_length_ratio)))
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 720.0,
            threshold=self.config.hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=self.config.max_line_gap,
        )

        candidates: list[CourtLine] = []
        if raw_lines is not None:
            for raw_line in np.asarray(raw_lines).reshape(-1, 4):
                line = tuple(int(value) for value in raw_line)
                length = _line_length(line)
                if length < min_line_length:
                    continue
                white_support = self._line_support(white_mask, line, radius=2)
                edge_support = self._line_support(edges, line, radius=1)
                floor_support = self._line_support(floor_mask, line, radius=5)
                score = self._candidate_score(
                    line,
                    length,
                    white_support,
                    edge_support,
                    floor_support,
                    width,
                    height,
                )
                if (
                    white_support < self.config.min_white_support
                    or floor_support < self.config.min_floor_support
                    or score < self.config.min_candidate_score
                ):
                    continue
                candidates.append(
                    CourtLine(
                        line_id=-1,
                        x1=line[0],
                        y1=line[1],
                        x2=line[2],
                        y2=line[3],
                        length=length,
                        angle_deg=_line_angle(line),
                        score=score,
                        white_support=white_support,
                        edge_support=edge_support,
                        floor_support=floor_support,
                    )
                )

        candidates = self._deduplicate(candidates)
        candidates_before_limit = len(candidates)
        candidates = self._assign_families(candidates, width, height)
        candidates = self._assign_roles(candidates, width, height)
        candidates.sort(key=lambda item: (item.score, item.length), reverse=True)
        candidates = candidates[: self.config.max_output_lines]
        for line_id, line in enumerate(candidates, start=1):
            line.line_id = line_id

        target_court = None
        if target_player_points is not None:
            target_court = select_target_court(
                candidates,
                target_player_points,
                width=width,
                height=height,
                player_axis_angle_deg=target_axis_angle_deg,
            )

        diagnostics = {
            "roi_top_px": roi_top,
            "raw_hough_lines": 0 if raw_lines is None else int(len(raw_lines)),
            "candidate_lines_after_dedup_before_output_limit": candidates_before_limit,
            "output_lines": len(candidates),
            "opencv_version": cv2.__version__,
            "opencv_cuda_devices": self._cuda_device_count(),
            "processing": "CPU Canny + CPU HoughLinesP; no deep model loaded",
            "target_court_selected": target_court is not None,
        }
        return CourtDetectionResult(
            frame_index=int(frame_index),
            width=width,
            height=height,
            lines=candidates,
            edges=edges,
            white_mask=white_mask,
            target_court=target_court,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _cuda_device_count() -> int:
        try:
            return int(cv2.cuda.getCudaEnabledDeviceCount())
        except (AttributeError, cv2.error):
            return 0

    @staticmethod
    def _line_support(mask: np.ndarray, line: Sequence[int], radius: int) -> float:
        points = _sample_segment(line, max(24, int(_line_length(line) / 5)))
        height, width = mask.shape[:2]
        values: list[float] = []
        for x, y in points:
            x0, x1 = max(0, int(x) - radius), min(width, int(x) + radius + 1)
            y0, y1 = max(0, int(y) - radius), min(height, int(y) + radius + 1)
            patch = mask[y0:y1, x0:x1]
            values.append(float(np.count_nonzero(patch)) / max(1, patch.size))
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _candidate_score(
        line: Sequence[int],
        length: float,
        white_support: float,
        edge_support: float,
        floor_support: float,
        width: int,
        height: int,
    ) -> float:
        x1, y1, x2, y2 = line
        y_mean = 0.5 * (y1 + y2) / max(1, height)
        length_score = min(1.0, length / max(1.0, width * 0.45))
        # Court lines tend to be on the green playing surface.  A modest
        # preference for the lower part of the frame suppresses stadium rails
        # without hard-cropping the far court.
        floor_score = min(1.0, max(0.0, (y_mean - 0.22) / 0.58))
        return float(
            0.40 * white_support
            + 0.16 * edge_support
            + 0.28 * floor_support
            + 0.10 * length_score
            + 0.06 * floor_score
        )

    def _deduplicate(self, lines: Iterable[CourtLine]) -> list[CourtLine]:
        """Suppress repeated Hough pieces while keeping distinct parallel lines."""

        ordered = sorted(lines, key=lambda item: (item.score, item.length), reverse=True)
        kept: list[CourtLine] = []
        for candidate in ordered:
            if any(self._same_physical_segment(candidate, previous) for previous in kept):
                continue
            kept.append(candidate)
        return kept

    def _same_physical_segment(self, first: CourtLine, second: CourtLine) -> bool:
        if _angular_distance(first.angle_deg, second.angle_deg) > self.config.merge_angle_tolerance:
            return False
        a = np.asarray(first.segment, dtype=np.float32).reshape(2, 2)
        b = np.asarray(second.segment, dtype=np.float32).reshape(2, 2)
        direction = a[1] - a[0]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return False
        direction /= norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        center_distance = abs(float(np.dot(np.mean(a, axis=0) - np.mean(b, axis=0), normal)))
        if center_distance > self.config.merge_distance_tolerance:
            return False
        projections_a = np.dot(a, direction)
        projections_b = np.dot(b, direction)
        gap = max(float(np.min(projections_a)), float(np.min(projections_b))) - min(
            float(np.max(projections_a)), float(np.max(projections_b))
        )
        return gap <= self.config.merge_gap_tolerance

    def _assign_families(self, lines: list[CourtLine], width: int, height: int) -> list[CourtLine]:
        if not lines:
            return lines
        # Weighted angle histogram.  Using unoriented angles avoids treating a
        # line and the same line in the reverse direction as different groups.
        bin_size = 10.0
        histogram = np.zeros(18, dtype=np.float64)
        for line in lines:
            histogram[int(line.angle_deg // bin_size) % 18] += line.length * max(0.1, line.score)
        primary_bin = int(np.argmax(histogram))
        primary_angle = (primary_bin + 0.5) * bin_size

        eligible = [
            (value, index)
            for index, value in enumerate(histogram)
            if _angular_distance((index + 0.5) * bin_size, primary_angle) >= 30.0
        ]
        secondary_angle = None
        if eligible:
            _, secondary_bin = max(eligible)
            secondary_angle = (secondary_bin + 0.5) * bin_size

        for line in lines:
            primary_distance = _angular_distance(line.angle_deg, primary_angle)
            secondary_distance = (
                _angular_distance(line.angle_deg, secondary_angle) if secondary_angle is not None else 999.0
            )
            if primary_distance <= min(secondary_distance, 18.0):
                line.family = "cross_court_family"
            elif secondary_angle is not None and secondary_distance < 25.0:
                line.family = "length_court_family"
            else:
                line.family = "other"
        return lines

    def _assign_roles(self, lines: list[CourtLine], width: int, height: int) -> list[CourtLine]:
        if not lines:
            return lines
        non_other = [line for line in lines if line.family != "other"]
        cross = [line for line in non_other if line.family == "cross_court_family"]

        # The net is usually the longest bright line crossing the middle of the
        # playing surface, and is often not in the dominant horizontal family
        # for a corner/side camera.  This is only a candidate label.
        net_candidates = [
            line
            for line in non_other
            if (
                0.36 * height <= line.midpoint[1] <= 0.68 * height
                and (
                    line.length >= 0.10 * width
                    if line.family == "length_court_family"
                    else line.length >= 0.22 * width
                )
            )
        ]
        if net_candidates:
            preferred_net_candidates = [
                line for line in net_candidates if line.family == "length_court_family"
            ]
            if preferred_net_candidates:
                net_candidates = preferred_net_candidates
            net = max(
                net_candidates,
                key=lambda line: line.length
                * max(0.1, line.score)
                * (1.0 if line.family == "length_court_family" else 0.85)
                * (1.15 if 0.25 * width <= line.midpoint[0] <= 0.82 * width else 0.85),
            )
            net.role = "net_candidate"
            net.role_confidence = self._role_confidence(
                net,
                width=width,
                height=height,
                length_ratio=0.22,
                center_bonus=True,
            )

        # At least two long lines from the dominant family are useful for the
        # next calibration step.  The outermost candidates are marked as
        # boundaries; interior candidates are service-line hints.
        remaining_cross = [line for line in cross if line.role == "court_line"]
        remaining_cross.sort(key=lambda line: line.midpoint[1])
        if remaining_cross:
            boundary_count = min(2, len(remaining_cross))
            selected_boundaries = remaining_cross[:boundary_count] + remaining_cross[-boundary_count:]
            seen: set[int] = set()
            for line in selected_boundaries:
                identity = id(line)
                if identity in seen:
                    continue
                seen.add(identity)
                line.role = "boundary_candidate"
                line.role_confidence = self._role_confidence(
                    line,
                    width=width,
                    height=height,
                    length_ratio=0.12,
                    center_bonus=False,
                )

            for line in remaining_cross:
                if line.role == "court_line" and line.length >= 0.12 * width:
                    line.role = "service_candidate"
                    line.role_confidence = (
                        self._role_confidence(
                            line,
                            width=width,
                            height=height,
                            length_ratio=0.12,
                            center_bonus=False,
                        )
                        * 0.85
                    )
        return lines

    @staticmethod
    def _role_confidence(
        line: CourtLine,
        width: int,
        height: int,
        length_ratio: float,
        center_bonus: bool,
    ) -> float:
        length_score = min(1.0, line.length / max(1.0, length_ratio * width))
        center_score = 0.2 if center_bonus and 0.32 <= line.midpoint[1] / max(1, height) <= 0.78 else 0.0
        return float(min(1.0, 0.55 * line.score + 0.35 * length_score + center_score))

    def annotate(
        self,
        frame_bgr: np.ndarray,
        result: CourtDetectionResult,
        show_ids: bool = True,
        target_only: bool = False,
    ) -> np.ndarray:
        """Draw either all candidates or only the player-selected court."""

        output = frame_bgr.copy()
        if target_only and result.target_court is not None:
            court = result.target_court
            vertices = np.round(court.vertices).astype(np.int32)
            vertices[:, 0] = np.clip(vertices[:, 0], 0, result.width - 1)
            vertices[:, 1] = np.clip(vertices[:, 1], 0, result.height - 1)
            cv2.polylines(output, [vertices.reshape(-1, 1, 2)], True, (0, 0, 255), 4, cv2.LINE_AA)
            for point, color in zip(court.player_points, ((0, 165, 255), (255, 180, 0))):
                cv2.circle(output, tuple(np.round(point).astype(int)), 7, color, -1, cv2.LINE_AA)
            cv2.line(
                output,
                tuple(np.round(court.player_points[0]).astype(int)),
                tuple(np.round(court.player_points[1]).astype(int)),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if court.net_endpoint_points is not None:
                for point in court.net_endpoint_points:
                    cv2.circle(
                        output,
                        tuple(np.round(point).astype(int)),
                        8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            if court.pole_foot_points is not None:
                for point in court.pole_foot_points:
                    cv2.circle(
                        output,
                        tuple(np.round(point).astype(int)),
                        7,
                        (255, 120, 0),
                        2,
                        cv2.LINE_AA,
                    )
            if court.pole_line_ids is not None:
                for pole_id in court.pole_line_ids:
                    pole = next((line for line in result.lines if line.line_id == pole_id), None)
                    if pole is not None:
                        cv2.line(output, (pole.x1, pole.y1), (pole.x2, pole.y2), (255, 120, 0), 3, cv2.LINE_AA)
            if court.net_line_id is not None:
                net = next((line for line in result.lines if line.line_id == court.net_line_id), None)
                if net is not None:
                    if court.net_endpoint_points is not None:
                        start, end = [tuple(np.round(point).astype(int)) for point in court.net_endpoint_points]
                    else:
                        net_points = self._line_polygon_intersections(net, court.vertices)
                        if len(net_points) >= 2:
                            start, end = net_points[0], net_points[1]
                        else:
                            start, end = (net.x1, net.y1), (net.x2, net.y2)
                    cv2.line(output, start, end, (255, 0, 255), 4, cv2.LINE_AA)
                    if show_ids:
                        cv2.putText(
                            output,
                            f"NET L{net.line_id}",
                            (net.x1 + 6, max(18, net.y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
            if show_ids:
                for index, point in enumerate(vertices, start=1):
                    cv2.putText(
                        output,
                        f"B{index}",
                        (int(point[0]) + 6, int(point[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            title = f"Target court: 4 sides | net + posts | {court.construction_method}"
            cv2.rectangle(output, (8, 8), (min(result.width - 8, 610), 38), (0, 0, 0), -1)
            cv2.putText(output, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
            return output

        for line in result.lines:
            color = self._ROLE_COLORS.get(line.role, self._ROLE_COLORS["court_line"])
            thickness = 4 if line.role == "net_candidate" else 2
            cv2.line(output, (line.x1, line.y1), (line.x2, line.y2), color, thickness, cv2.LINE_AA)
            if show_ids:
                label = f"L{line.line_id} {line.role.replace('_candidate', '')} {line.score:.2f}"
                x, y = map(int, line.midpoint)
                cv2.putText(output, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        title = f"Court lines: {len(result.lines)} | CPU Canny + HoughLinesP"
        cv2.rectangle(output, (8, 8), (min(result.width - 8, 560), 38), (0, 0, 0), -1)
        cv2.putText(output, title, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        return output

    @staticmethod
    def _line_polygon_intersections(line: CourtLine, polygon: np.ndarray) -> list[tuple[int, int]]:
        """Extend a detected net segment only across the selected court polygon."""

        equation = np.array(
            [line.y1 - line.y2, line.x2 - line.x1, line.x1 * line.y2 - line.x2 * line.y1],
            dtype=float,
        )
        norm = float(np.linalg.norm(equation[:2]))
        if norm < 1e-8:
            return []
        equation /= norm
        intersections: list[np.ndarray] = []
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            value_first = float(np.dot(equation[:2], first) + equation[2])
            value_second = float(np.dot(equation[:2], second) + equation[2])
            if value_first * value_second > 0.0:
                continue
            denominator = value_first - value_second
            ratio = 0.0 if abs(denominator) < 1e-8 else value_first / denominator
            if -1e-6 <= ratio <= 1.0 + 1e-6:
                point = first + ratio * (second - first)
                if not any(np.linalg.norm(point - previous) < 2.0 for previous in intersections):
                    intersections.append(point)
        return [tuple(np.round(point).astype(int)) for point in intersections[:2]]

    @staticmethod
    def edge_preview(result: CourtDetectionResult) -> np.ndarray:
        """Return a BGR preview of the Canny input and retained line segments."""

        preview = cv2.cvtColor(result.edges, cv2.COLOR_GRAY2BGR)
        for line in result.lines:
            color = (0, 220, 0)
            if line.role == "net_candidate":
                color = (255, 0, 255)
            elif line.role == "boundary_candidate":
                color = (0, 0, 255)
            elif line.role == "service_candidate":
                color = (0, 255, 255)
            cv2.line(preview, (line.x1, line.y1), (line.x2, line.y2), color, 2, cv2.LINE_AA)
        return preview
