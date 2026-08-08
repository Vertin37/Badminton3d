"""Render a fixed-ID filtered pose CSV over its source video.

This is a visualization-only step.  It reads coordinates already produced by
the detector/filter and never imports RTMLib, ONNX Runtime, or CUDA.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_DIR / "examples" / "data" / "test.mp4"
DEFAULT_CSV = PROJECT_DIR / "outputs" / "pose_test" / "pose_data_stable.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "pose_test" / "analyzed_video_stable.mp4"

# COCO body skeleton (keypoints 0..16).  WholeBody hand/face landmarks are
# intentionally not drawn here so the comparison video remains readable.
BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _number(value: str) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load_points(csv_path: Path) -> dict[int, dict[int, dict[int, dict[str, object]]]]:
    points: dict[int, dict[int, dict[int, dict[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"frame_id", "player_id", "keypoint_id", "x", "y"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"filtered pose CSV missing columns: {sorted(missing)}")
        for row in reader:
            frame_id = int(row["frame_id"])
            player_id = int(row["player_id"])
            keypoint_id = int(row["keypoint_id"])
            points[frame_id][player_id][keypoint_id] = {
                "filtered": (_number(row["x"]), _number(row["y"])),
                "raw": (_number(row.get("raw_x", "")), _number(row.get("raw_y", ""))),
                "status": row.get("filter_status", "observed"),
            }
    return points


def _point(
    frame_points: dict[int, dict[str, object]],
    keypoint_id: int,
    kind: str,
) -> Optional[tuple[int, int]]:
    item = frame_points.get(keypoint_id)
    if item is None:
        return None
    x, y = item[kind]
    if x is None or y is None:
        return None
    return int(round(float(x))), int(round(float(y)))


def _draw_player(
    frame,
    player_id: int,
    points: dict[int, dict[str, object]],
    color: tuple[int, int, int],
    show_raw: bool,
) -> None:
    import cv2

    if show_raw:
        for keypoint_id in range(17):
            raw_point = _point(points, keypoint_id, "raw")
            if raw_point is not None:
                cv2.circle(frame, raw_point, 3, (150, 150, 150), -1, cv2.LINE_AA)
        for start, end in BODY_EDGES:
            start_point = _point(points, start, "raw")
            end_point = _point(points, end, "raw")
            if start_point is not None and end_point is not None:
                cv2.line(frame, start_point, end_point, (120, 120, 120), 1, cv2.LINE_AA)

    for start, end in BODY_EDGES:
        start_point = _point(points, start, "filtered")
        end_point = _point(points, end, "filtered")
        if start_point is not None and end_point is not None:
            cv2.line(frame, start_point, end_point, color, 2, cv2.LINE_AA)
    for keypoint_id in range(17):
        filtered_point = _point(points, keypoint_id, "filtered")
        if filtered_point is not None:
            cv2.circle(frame, filtered_point, 4, color, -1, cv2.LINE_AA)

    center = _point(points, 0, "filtered") or _point(points, 5, "filtered")
    if center is not None:
        cv2.putText(
            frame,
            f"P{player_id} stable",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def render_video(
    source_video: Path,
    filtered_csv: Path,
    output_video: Path,
    show_raw: bool = True,
) -> None:
    import cv2

    points = _load_points(filtered_csv)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source video: {source_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot create output video: {output_video}")

    colors = {0: (0, 165, 255), 1: (255, 220, 0)}  # BGR orange/cyan.
    frame_id = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_id += 1
            for player_id, player_points in points.get(frame_id, {}).items():
                _draw_player(
                    frame,
                    player_id,
                    player_points,
                    colors.get(player_id, (0, 255, 0)),
                    show_raw=show_raw,
                )
            cv2.putText(
                frame,
                "Filtered pose (One Euro) | gray=raw, color=stable",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    print(f"stable pose video: {output_video}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render filtered pose CSV over video")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--filtered-only",
        action="store_true",
        help="hide gray raw points and show only the filtered skeleton",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    render_video(args.video, args.csv, args.output, show_raw=not args.filtered_only)
