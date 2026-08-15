from __future__ import annotations

import argparse
import csv
import ctypes
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from src.pose_biomechanics.person_tracker import PersonTracker, TrackingResult
from src.pose_biomechanics.temporal_filter import (
    TemporalFilterConfig,
    filter_tracked_pose_csv,
)


PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_DIR / "examples" / "data" / "test.mp4"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "pose_test"

CONFIDENCE_THRESHOLD = 0.3
TRACKED_CSV_NAME = "pose_data_tracked.csv"
RAW_CSV_NAME = "pose_data_raw.csv"
DEBUG_CSV_NAME = "tracking_debug.csv"
STABLE_CSV_NAME = "pose_data_stable.csv"
FILTER_STATS_NAME = "temporal_filter_stats.json"
TRACKED_VIDEO_NAME = "analyzed_video_tracked.mp4"
DEBUG_VIDEO_NAME = "tracking_debug.mp4"

# Keep Windows DLL-directory handles alive for the whole process.  ONNX
# Runtime 1.23.2's Windows preload list omits cuDNN's tensor-IR engine DLL,
# while cuDNN 9.25 loads it lazily on the first convolution.
_CUDA_DLL_DIR_HANDLES: list[object] = []

TRACKED_CSV_FIELDS = [
    "frame_id",
    "player_id",
    "person_id",  # Compatibility alias for older downstream scripts.
    "keypoint_id",
    "x",
    "y",
    "confidence",
]
RAW_CSV_FIELDS = [
    "frame_id",
    "person_id",
    "keypoint_id",
    "x",
    "y",
    "confidence",
]
DEBUG_CSV_FIELDS = [
    "frame_id",
    "player_id",
    "matched",
    "source_detection_id",
    "missed_frames",
    "center_x",
    "center_y",
    "predicted_x",
    "predicted_y",
    "body_scale",
    "match_cost",
    "candidate_count",
    "visible_body_keypoints",
]


def _make_tracker() -> PersonTracker:
    return PersonTracker(
        num_players=2,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        min_body_keypoints=4,
        max_missed_frames=15,
    )


def _write_tracked_rows(
    writer: csv.writer,
    frame_id: int,
    results: Iterable[TrackingResult],
) -> None:
    for result in results:
        detection = result.detection
        if detection is None:
            continue
        for keypoint_id, point in enumerate(detection.keypoints):
            confidence = (
                float(detection.scores[keypoint_id])
                if keypoint_id < len(detection.scores)
                else 0.0
            )
            writer.writerow(
                [
                    frame_id,
                    result.player_id,
                    result.player_id,
                    keypoint_id,
                    float(point[0]),
                    float(point[1]),
                    confidence,
                ]
            )


def _write_raw_rows(
    writer: csv.writer,
    frame_id: int,
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> None:
    for person_id, person_keypoints in enumerate(keypoints):
        person_scores = scores[person_id]
        for keypoint_id, point in enumerate(person_keypoints):
            confidence = (
                float(person_scores[keypoint_id])
                if keypoint_id < len(person_scores)
                else 0.0
            )
            writer.writerow(
                [
                    frame_id,
                    person_id,
                    keypoint_id,
                    float(point[0]),
                    float(point[1]),
                    confidence,
                ]
            )


def _write_debug_rows(
    writer: csv.writer,
    frame_id: int,
    results: Iterable[TrackingResult],
    candidate_count: int,
) -> None:
    for result in results:
        detection = result.detection
        if detection is None:
            center_x = center_y = ""
            body_scale = ""
            visible_body_keypoints = ""
        else:
            center_x = float(detection.center[0])
            center_y = float(detection.center[1])
            body_scale = float(detection.scale)
            visible_body_keypoints = detection.visible_body_keypoints

        predicted_x = float(result.predicted_center[0])
        predicted_y = float(result.predicted_center[1])
        writer.writerow(
            [
                frame_id,
                result.player_id,
                int(result.matched),
                "" if result.source_detection_id is None else result.source_detection_id,
                result.missed_frames,
                center_x,
                center_y,
                predicted_x,
                predicted_y,
                body_scale,
                "" if result.match_cost is None else float(result.match_cost),
                candidate_count,
                visible_body_keypoints,
            ]
        )


def _load_pose_csv(
    csv_path: Path,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, int]], int, int]:
    """Load detector CSV rows without requiring RTMLib or a GPU.

    The returned mapping is ``frame_id -> (keypoints, scores, raw_count)``.
    ``raw_count`` deliberately counts all source detections, including ones
    that the tracker later rejects for insufficient body support.
    """

    rows_by_frame: dict[int, dict[int, dict[int, tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    max_keypoint_id = -1

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头：{csv_path}")
        if "frame_id" not in reader.fieldnames:
            raise ValueError("CSV 缺少 frame_id 列")
        person_column = (
            "person_id"
            if "person_id" in reader.fieldnames
            else "player_id"
            if "player_id" in reader.fieldnames
            else None
        )
        if person_column is None:
            raise ValueError("CSV 缺少 person_id/player_id 列")

        for row in reader:
            frame_id = int(row["frame_id"])
            source_person_id = int(row[person_column])
            keypoint_id = int(row["keypoint_id"])
            x = float(row["x"])
            y = float(row["y"])
            confidence = float(row["confidence"])
            rows_by_frame[frame_id][source_person_id][keypoint_id] = (
                x,
                y,
                confidence,
            )
            max_keypoint_id = max(max_keypoint_id, keypoint_id)

    if not rows_by_frame:
        return {}, 0, max_keypoint_id + 1

    keypoint_count = max_keypoint_id + 1
    frames: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for frame_id, people in rows_by_frame.items():
        keypoints = np.zeros(
            (len(people), keypoint_count, 2),
            dtype=float,
        )
        scores = np.zeros((len(people), keypoint_count), dtype=float)
        for output_index, source_person_id in enumerate(sorted(people)):
            for keypoint_id, (x, y, confidence) in people[source_person_id].items():
                keypoints[output_index, keypoint_id] = (x, y)
                scores[output_index, keypoint_id] = confidence
        frames[frame_id] = (keypoints, scores, len(people))

    return frames, max(frames), keypoint_count


def _draw_tracking_labels(
    frame: np.ndarray,
    results: Iterable[TrackingResult],
) -> np.ndarray:
    import cv2

    colors = [(255, 160, 0), (0, 210, 255)]  # BGR: orange, yellow.
    output = frame
    for result in results:
        color = colors[result.player_id % len(colors)]
        if result.matched and result.detection is not None:
            center = result.detection.center
            x, y = int(round(center[0])), int(round(center[1]))
            label = f"P{result.player_id}"
            cv2.circle(output, (x, y), 8, color, -1, cv2.LINE_AA)
            cv2.putText(
                output,
                label,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
        elif np.isfinite(result.predicted_center).all():
            x, y = (
                int(round(result.predicted_center[0])),
                int(round(result.predicted_center[1])),
            )
            cv2.drawMarker(
                output,
                (x, y),
                color,
                cv2.MARKER_CROSS,
                24,
                2,
            )
            cv2.putText(
                output,
                f"P{result.player_id} MISS {result.missed_frames}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
    return output


def _write_debug_video(
    source_video: Path,
    output_video: Path,
    results_by_frame: dict[int, list[TrackingResult]],
) -> None:
    import cv2

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开用于标注的源视频：{source_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建跟踪标注视频：{output_video}")

    frame_id = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_id += 1
            frame_results = results_by_frame.get(frame_id, [])
            _draw_tracking_labels(frame, frame_results)
            cv2.putText(
                frame,
                f"Tracking frame: {frame_id}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def run_temporal_filter(
    input_csv: Path,
    output_dir: Path,
    fps: float = 25.0,
) -> None:
    """Run CPU-only temporal filtering after fixed-ID tracking."""

    stable_csv = output_dir / STABLE_CSV_NAME
    stats_json = output_dir / FILTER_STATS_NAME
    stats = filter_tracked_pose_csv(
        input_csv=input_csv,
        output_csv=stable_csv,
        stats_json=stats_json,
        config=TemporalFilterConfig(fps=fps),
    )
    print(f"时序滤波完成（One Euro，CPU）：{stable_csv}")
    print(f"滤波统计：{stats_json}")
    print(
        "滤波点统计："
        f"输入有效 {stats['input_valid_points']}，"
        f"拒绝异常 {stats['rejected_outlier_points']}，"
        f"插值 {stats['interpolated_points']}，"
        f"保持 {stats['held_points']}，"
        f"输出有效 {stats['output_valid_points']}"
    )


def track_existing_csv(
    input_csv: Path,
    output_dir: Path,
    source_video: Optional[Path] = None,
) -> None:
    """Track an existing pose CSV; this path never imports RTMLib/CUDA."""

    frames, last_frame_id, _ = _load_pose_csv(input_csv)
    if not frames:
        raise ValueError(f"CSV 没有可处理的数据：{input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tracked_csv = output_dir / TRACKED_CSV_NAME
    debug_csv = output_dir / DEBUG_CSV_NAME

    tracker = _make_tracker()
    results_by_frame: dict[int, list[TrackingResult]] = {}
    observed_frames = [0, 0]
    longest_misses = [0, 0]
    candidate_counts: list[int] = []

    with (
        tracked_csv.open("w", encoding="utf-8-sig", newline="") as tracked_file,
        debug_csv.open("w", encoding="utf-8-sig", newline="") as debug_file,
    ):
        tracked_writer = csv.writer(tracked_file)
        debug_writer = csv.writer(debug_file)
        tracked_writer.writerow(TRACKED_CSV_FIELDS)
        debug_writer.writerow(DEBUG_CSV_FIELDS)

        for frame_id in range(1, last_frame_id + 1):
            keypoints, scores, candidate_count = frames.get(
                frame_id,
                (
                    np.empty((0, 0, 2), dtype=float),
                    np.empty((0, 0), dtype=float),
                    0,
                ),
            )
            results = tracker.update(keypoints, scores)
            results_by_frame[frame_id] = results
            candidate_counts.append(candidate_count)
            _write_tracked_rows(tracked_writer, frame_id, results)
            _write_debug_rows(debug_writer, frame_id, results, candidate_count)

            for result in results:
                if result.matched:
                    observed_frames[result.player_id] += 1
                longest_misses[result.player_id] = max(
                    longest_misses[result.player_id],
                    result.missed_frames,
                )

    if source_video is None:
        candidate_video = output_dir / "analyzed_video.mp4"
        source_video = candidate_video if candidate_video.exists() else VIDEO_PATH
    debug_video = output_dir / DEBUG_VIDEO_NAME
    if source_video.exists():
        try:
            _write_debug_video(source_video, debug_video, results_by_frame)
        except ModuleNotFoundError as error:
            if error.name != "cv2":
                raise
            print("当前 Python 没有 OpenCV，已跳过视频标注；CSV/统计仍已完成。")

    print(f"离线跟踪完成，共处理 {last_frame_id} 帧。")
    print(f"固定 ID CSV：{tracked_csv}")
    print(f"跟踪统计 CSV：{debug_csv}")
    if debug_video.exists():
        print(f"ID 标注视频：{debug_video}")
    print(
        "候选人数范围："
        f"{min(candidate_counts)}–{max(candidate_counts)}；"
        f"P0/P1 有效帧：{observed_frames[0]}/{observed_frames[1]}；"
        f"最长连续漏检：{longest_misses[0]}/{longest_misses[1]} 帧"
    )
    run_temporal_filter(tracked_csv, output_dir)


def _select_device(requested: str, providers: list[str]) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "已要求 CUDA，但 ONNX Runtime 没有 CUDAExecutionProvider。"
            )
        return "cuda"
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"


def _prepare_cuda_runtime(ort: object) -> None:
    """Make the bundled CUDA/cuDNN DLLs discoverable before RTMLib loads."""

    site_packages = Path(ort.__file__).resolve().parents[1]
    nvidia_root = site_packages / "nvidia"
    dll_dirs = [
        nvidia_root / "cublas" / "bin",
        nvidia_root / "cuda_runtime" / "bin",
        nvidia_root / "cuda_nvrtc" / "bin",
        nvidia_root / "nvjitlink" / "bin",
        nvidia_root / "cudnn" / "bin",
    ]
    for dll_dir in dll_dirs:
        if not dll_dir.is_dir():
            continue
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            _CUDA_DLL_DIR_HANDLES.append(add_dll_directory(str(dll_dir)))

    # Preserve the standard ONNX Runtime preload path used by the project.
    preload_dlls = getattr(ort, "preload_dlls", None)
    if preload_dlls is not None:
        preload_dlls(directory="")

    # ORT 1.23.2 does not preload this cuDNN 9 engine DLL on Windows, but
    # cuDNN may require it when building a convolution frontend graph.
    tensor_ir_dll = nvidia_root / "cudnn" / "bin" / "cudnn_engines_tensor_ir64_9.dll"
    if tensor_ir_dll.is_file():
        ctypes.CDLL(str(tensor_ir_dll))


def run_video_inference(
    video_path: Path,
    output_dir: Path,
    requested_device: str,
    display: bool,
) -> None:
    """Run RTMLib inference and assign stable IDs to its detections."""

    import cv2

    # Keep all RTMLib/ONNX Runtime imports here so --from-csv is guaranteed to
    # be a CPU-only data operation.
    import onnxruntime as ort

    providers = ort.get_available_providers()
    device = _select_device(requested_device, providers)
    if device == "cuda":
        _prepare_cuda_runtime(ort)

    from rtmlib import Wholebody, draw_skeleton

    print(f"ONNX Runtime Providers：{providers}")
    print(f"当前推理设备：{device}")
    pose_model = Wholebody(
        mode="lightweight",
        backend="onnxruntime",
        device=device,
        to_openpose=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0

    output_dir.mkdir(parents=True, exist_ok=True)
    tracked_video = output_dir / TRACKED_VIDEO_NAME
    tracked_csv = output_dir / TRACKED_CSV_NAME
    raw_csv = output_dir / RAW_CSV_NAME
    debug_csv = output_dir / DEBUG_CSV_NAME

    writer = cv2.VideoWriter(
        str(tracked_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频：{tracked_video}")

    tracker = _make_tracker()
    frame_id = 0
    try:
        with (
            tracked_csv.open("w", encoding="utf-8-sig", newline="") as tracked_file,
            raw_csv.open("w", encoding="utf-8-sig", newline="") as raw_file,
            debug_csv.open("w", encoding="utf-8-sig", newline="") as debug_file,
        ):
            tracked_writer = csv.writer(tracked_file)
            raw_writer = csv.writer(raw_file)
            debug_writer = csv.writer(debug_file)
            tracked_writer.writerow(TRACKED_CSV_FIELDS)
            raw_writer.writerow(RAW_CSV_FIELDS)
            debug_writer.writerow(DEBUG_CSV_FIELDS)

            print("开始姿态识别与双人 ID 跟踪，按 q 可以提前结束。")
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_id += 1

                keypoints, scores = pose_model(frame)
                keypoints = np.asarray(keypoints)
                scores = np.asarray(scores)
                results = tracker.update(keypoints, scores)

                _write_raw_rows(raw_writer, frame_id, keypoints, scores)
                _write_tracked_rows(tracked_writer, frame_id, results)
                _write_debug_rows(debug_writer, frame_id, results, len(keypoints))

                result_frame = draw_skeleton(
                    frame.copy(),
                    keypoints,
                    scores,
                    kpt_thr=CONFIDENCE_THRESHOLD,
                    openpose_skeleton=False,
                )
                _draw_tracking_labels(result_frame, results)
                cv2.putText(
                    result_frame,
                    f"Frame: {frame_id}/{frame_total}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(result_frame)

                if display:
                    cv2.imshow("Badminton Pose Tracking", result_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("用户提前结束分析。")
                        break
                if frame_id % 50 == 0:
                    print(f"已处理 {frame_id}/{frame_total} 帧")
    finally:
        cap.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    print("分析完成。")
    print(f"跟踪视频：{tracked_video}")
    print(f"固定 ID CSV：{tracked_csv}")
    print(f"原始检测 CSV：{raw_csv}")
    print(f"跟踪统计 CSV：{debug_csv}")
    run_temporal_filter(tracked_csv, output_dir, fps=float(fps))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RTMLib 姿态提取与双人持续 ID 跟踪。"
    )
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="只读取已有姿态 CSV 做离线跟踪；不会导入 RTMLib，也不会使用 GPU。",
    )
    parser.add_argument(
        "--filter-from-csv",
        type=Path,
        default=None,
        help="只读取已有固定 ID CSV 做 One Euro 时序滤波；不会导入 RTMLib，也不会使用 GPU。",
    )
    parser.add_argument(
        "--filter-fps",
        type=float,
        default=25.0,
        help="离线时序滤波使用的帧率，默认 25 FPS。",
    )
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="仅对视频推理模式生效。",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="视频推理时不显示窗口。",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.from_csv is not None and args.filter_from_csv is not None:
        raise SystemExit("--from-csv 与 --filter-from-csv 不能同时使用")
    if args.filter_from_csv is not None:
        input_csv = args.filter_from_csv
        if not input_csv.is_absolute():
            input_csv = PROJECT_DIR / input_csv
        run_temporal_filter(input_csv, args.output_dir, fps=args.filter_fps)
        return
    if args.from_csv is not None:
        input_csv = args.from_csv
        if not input_csv.is_absolute():
            input_csv = PROJECT_DIR / input_csv
        track_existing_csv(input_csv, args.output_dir, args.video)
        return

    run_video_inference(
        video_path=args.video,
        output_dir=args.output_dir,
        requested_device=args.device,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
