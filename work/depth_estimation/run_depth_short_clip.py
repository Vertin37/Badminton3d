"""Run the isolated monocular-depth Phase 1 on a short Windows clip.

The script intentionally has no fusion logic. It writes only inside the supplied
experiment output directory and keeps the original video and GVHMR bundle read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True, help="1-based inclusive")
    parser.add_argument("--end-frame", type=int, required=True, help="1-based inclusive")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_frame < 1 or args.end_frame < args.start_frame:
        raise ValueError("Invalid 1-based frame range")
    for path in (args.video, args.model, args.vendor):
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "depth_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.vendor.resolve()))

    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.end_frame > total_frames:
        raise ValueError(f"Requested frame {args.end_frame}, video has {total_frames}")

    frame_count = args.end_frame - args.start_frame + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame - 1)
    first_ok, first_frame = cap.read()
    if not first_ok or first_frame is None:
        cap.release()
        raise RuntimeError("Could not read the first selected frame")
    if first_frame.shape[:2] != (height, width):
        height, width = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame - 1)

    model = YOLO(str(args.model))
    if getattr(model, "task", None) != "depth":
        cap.release()
        raise RuntimeError(f"Verified model task is not depth: {getattr(model, 'task', None)!r}")

    raw_path = args.output_dir / "depth_raw.npy"
    raw = np.lib.format.open_memmap(
        raw_path, mode="w+", dtype=np.float32, shape=(frame_count, height, width)
    )
    preview_path = args.output_dir / "depth_preview.mp4"
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        raw._mmap.close()
        raw_path.unlink(missing_ok=True)
        cap.release()
        raise RuntimeError(f"Could not create preview video: {preview_path}")

    frame_stats: list[dict[str, float | int]] = []
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    try:
        with torch.inference_mode():
            for offset in range(frame_count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read selected frame {args.start_frame + offset}")
                results = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    device=args.device,
                    batch=1,
                    verbose=False,
                    stream=False,
                )
                if not results or getattr(results[0], "depth", None) is None:
                    raise RuntimeError("Depth result is missing result.depth")
                depth_obj = results[0].depth
                depth_data = getattr(depth_obj, "data", depth_obj)
                if hasattr(depth_data, "detach"):
                    depth = depth_data.detach().float().cpu().numpy()
                else:
                    depth = np.asarray(depth_data, dtype=np.float32)
                depth = np.squeeze(depth).astype(np.float32, copy=False)
                if depth.ndim != 2:
                    raise RuntimeError(f"Unexpected depth shape: {depth.shape}")
                if depth.shape != (height, width):
                    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
                raw[offset] = depth

                finite = np.isfinite(depth)
                positive = finite & (depth > 0)
                vals = depth[positive]
                if vals.size:
                    p2, p98 = np.percentile(vals, [2, 98])
                    vmin, vmax = float(vals.min()), float(vals.max())
                    median = float(np.median(vals))
                else:
                    p2 = p98 = vmin = vmax = median = float("nan")
                if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
                    p2, p98 = 0.0, 1.0
                scaled = np.clip((depth - p2) / (p98 - p2), 0.0, 1.0)
                scaled[~finite] = 0.0
                depth_u8 = (scaled * 255.0).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
                overlay = cv2.addWeighted(frame, 0.45, depth_color, 0.55, 0.0)
                cv2.putText(
                    overlay,
                    f"depth  frame {args.start_frame + offset}  p2={p2:.2f} p98={p98:.2f}",
                    (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(np.hstack([frame, overlay]))
                frame_stats.append(
                    {
                        "frame_id": args.start_frame + offset,
                        "valid_ratio": float(positive.mean()),
                        "finite_ratio": float(finite.mean()),
                        "positive_count": int(vals.size),
                        "min": vmin,
                        "max": vmax,
                        "median": median,
                        "p2": float(p2),
                        "p98": float(p98),
                    }
                )
    finally:
        writer.release()
        cap.release()
        raw.flush()
        raw._mmap.close()

    quality = {
        "input_video": str(args.video.resolve()),
        "total_input_frames": total_frames,
        "selected_start_frame_1based": args.start_frame,
        "selected_end_frame_1based": args.end_frame,
        "selected_frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "model": str(args.model.resolve()),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": args.device,
        "imgsz": args.imgsz,
        "raw_depth_file": str(raw_path.resolve()),
        "preview_file": str(preview_path.resolve()),
        "gpu_peak_allocated_mb": (
            int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if cuda_available else None
        ),
        "frames": frame_stats,
    }
    (args.output_dir / "depth_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: quality[k] for k in quality if k != "frames"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
