"""Run official Metric Video Depth Anything on an arbitrary video interval.

This is a thin, Windows-native adapter around the unmodified official vendor
repository. It writes a dense metric depth sequence and preview to a new output
directory; downstream person ROI extraction remains model-independent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
VENDOR = PROJECT / "work" / "video_depth_anything" / "vendor"
DEFAULT_CHECKPOINT = VENDOR / "checkpoints" / "metric_video_depth_anything_vits.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official temporally consistent metric video depth inference")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=1, help="1-based inclusive")
    parser.add_argument("--end-frame", type=int, help="1-based inclusive; defaults to the last frame")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-size", type=int, default=392, help="Must be practical for available VRAM")
    parser.add_argument("--max-res", type=int, default=1280)
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def _prepare_output(output_dir: Path, inputs: list[Path]) -> None:
    resolved = output_dir.resolve()
    if resolved in {path.resolve().parent for path in inputs}:
        raise ValueError("Output directory must not be an input directory")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def _read_interval(
    video: Path,
    start: int,
    end: int | None,
    max_res: int,
) -> tuple[np.ndarray, float, int, int, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Video FPS is unavailable")
    selected_end = frame_count if end is None else end
    if not (1 <= start <= selected_end <= frame_count):
        capture.release()
        raise ValueError(f"Invalid frame interval [{start}, {selected_end}] for {frame_count} frames")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    frames: list[np.ndarray] = []
    for frame_id in range(start, selected_end + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not read frame {frame_id}")
        if max_res > 0 and max(frame.shape[:2]) > max_res:
            scale = max_res / max(frame.shape[:2])
            width = int(round(frame.shape[1] * scale / 2.0) * 2)
            height = int(round(frame.shape[0] * scale / 2.0) * 2)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames), fps, selected_end, source_width, source_height


def _save_depth_preview(depths: np.ndarray, path: Path, fps: float) -> None:
    """Write a Windows-native preview without requiring an external ffmpeg."""
    valid = depths[np.isfinite(depths) & (depths > 0)]
    if valid.size == 0:
        raise ValueError("Depth sequence has no positive finite values")
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        high = low + 1e-6
    height, width = depths.shape[1:]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create depth preview: {path}")
    for depth in depths:
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
        gray = np.round(normalized * 255.0).astype(np.uint8)
        writer.write(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO))
    writer.release()


def run(args: argparse.Namespace) -> None:
    args.video = args.video.resolve()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.input_size < 224 or args.input_size % 14 != 0:
        raise ValueError("--input-size must be at least 224 and divisible by 14")
    if not torch.cuda.is_available():
        raise RuntimeError("Metric Video Depth Anything requires CUDA in this project")
    _prepare_output(args.output_dir, [args.video, args.checkpoint])

    frames, fps, selected_end, source_width, source_height = _read_interval(
        args.video, args.start_frame, args.end_frame, args.max_res
    )
    sys.path.insert(0, str(VENDOR))
    from video_depth_anything.video_depth import VideoDepthAnything  # noqa: PLC0415

    model = VideoDepthAnything(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        metric=True,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        depths, output_fps = model.infer_video_depth(
            frames,
            fps,
            input_size=args.input_size,
            device="cuda",
            fp32=args.fp32,
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "VDA ran out of VRAM. Re-run in a new output directory with --input-size 336 or 280."
        ) from exc

    depths = np.asarray(depths, dtype=np.float32)
    raw_path = args.output_dir / "depth_raw.npy"
    np.save(raw_path, depths)
    preview_path = args.output_dir / "depth_preview.mp4"
    _save_depth_preview(depths, preview_path, output_fps)

    finite = np.isfinite(depths) & (depths > 0)
    quality = {
        "backend": "official Metric Video Depth Anything Small",
        "vendor_commit": "4f5ae23172ba60fd7bc11ef671cca678842c7072",
        "checkpoint": str(args.checkpoint),
        "source_video": str(args.video),
        "selected_start_frame_1based": args.start_frame,
        "selected_end_frame_1based": selected_end,
        "frame_count": int(len(depths)),
        "fps": float(output_fps),
        "depth_shape": list(depths.shape),
        "source_width": source_width,
        "source_height": source_height,
        "depth_width": int(depths.shape[2]),
        "depth_height": int(depths.shape[1]),
        "input_size": args.input_size,
        "fp32": bool(args.fp32),
        "valid_ratio": float(finite.mean()),
        "positive_depth_p01_m": float(np.percentile(depths[finite], 1)),
        "positive_depth_median_m": float(np.median(depths[finite])),
        "positive_depth_p99_m": float(np.percentile(depths[finite], 99)),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_allocated_mb": int(torch.cuda.max_memory_allocated() / 1024**2),
        "outputs": {
            "depth_raw": raw_path.name,
            "depth_preview": preview_path.name,
        },
    }
    (args.output_dir / "depth_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
