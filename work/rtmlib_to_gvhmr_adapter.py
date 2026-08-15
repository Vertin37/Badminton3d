"""Build GVHMR preprocessing tensors from the existing BadmintonPose RTMLib CSV.

This is an experiment-only adapter. It does not modify GVHMR core code or the
existing 2D pipeline. RTMLib Wholebody emits 133 landmarks; the first 17 are
the COCO17 body landmarks expected by GVHMR's ``kp2d`` input.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy


PROJECT = Path(r"D:\Projects\BadmintonPose")
DEFAULT_GVHMR_ROOT = PROJECT / "work" / "GVHMR"
DEFAULT_CSV = PROJECT / "outputs" / "test2_analysis" / "pose_data_stable.csv"
DEFAULT_VIDEO = PROJECT / "examples" / "data" / "test2.mp4"
DEFAULT_CLIP = PROJECT / "outputs" / "world_hmr_test" / "test2_f120_240.mp4"
DEFAULT_SIMPLEVO = PROJECT / "outputs" / "world_hmr_test" / "simplevo_trajectory.npz"
DEFAULT_OUTPUT = PROJECT / "outputs" / "world_hmr_test" / "gvhmr_rtmlib_p0"

COCO17 = 17


def _read_rtmlib_csv(csv_path: Path) -> dict[int, dict[int, dict[int, tuple[float, float, float]]]]:
    rows: dict[int, dict[int, dict[int, tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_id", "keypoint_id", "x", "y", "confidence"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV 缺少字段，需要 {sorted(required)}: {csv_path}")
        person_field = "player_id" if "player_id" in reader.fieldnames else "person_id"
        if person_field not in reader.fieldnames:
            raise ValueError("CSV 缺少 player_id/person_id 字段")
        for row in reader:
            frame_id = int(row["frame_id"])
            player_id = int(row[person_field])
            keypoint_id = int(row["keypoint_id"])
            if keypoint_id >= COCO17:
                continue
            try:
                x = float(row["x"])
                y = float(row["y"])
                confidence = float(row["confidence"])
            except (TypeError, ValueError):
                # Stable tracking keeps rows for held/missing landmarks. Keep
                # the landmark slot but mark it as unobserved.
                x = y = 0.0
                confidence = 0.0
            rows[frame_id][player_id][keypoint_id] = (x, y, confidence)
    return rows


def _interp_column(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interpolate one time-series column, with edge hold."""
    if valid.all():
        return values
    indices = np.arange(len(values), dtype=np.float64)
    if not valid.any():
        raise ValueError("RTMLib CSV 没有可用的 COCO17 点")
    return np.interp(indices, indices[valid], values[valid])


def _load_clip_keypoints(
    rows: dict[int, dict[int, dict[int, tuple[float, float, float]]]],
    player_id: int,
    start_frame0: int,
    length: int,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Return kp2d (L,17,3) and xyxy boxes (L,4) in image coordinates."""
    kp = np.zeros((length, COCO17, 3), dtype=np.float32)
    kp[:, :, 2] = 0.0
    # run_pose_test numbers its first video frame as frame_id=1
    for local_idx in range(length):
        frame_id = start_frame0 + local_idx + 1
        points = rows.get(frame_id, {}).get(player_id, {})
        for keypoint_id, (x, y, confidence) in points.items():
            kp[local_idx, keypoint_id] = (x, y, confidence)

    # Fill missing coordinates temporally but preserve zero confidence for the
    # original missing observations. This mirrors a tracker hold without
    # pretending that an interpolated point was directly observed.
    for keypoint_id in range(COCO17):
        for axis in (0, 1):
            valid = np.isfinite(kp[:, keypoint_id, axis]) & (kp[:, keypoint_id, 2] > 0.05)
            if valid.any():
                kp[:, keypoint_id, axis] = _interp_column(kp[:, keypoint_id, axis], valid)

    xyxy = np.zeros((length, 4), dtype=np.float32)
    xyxy[:] = np.nan
    valid_frame_count = 0
    for local_idx in range(length):
        visible = kp[local_idx, :, 2] > 0.05
        if visible.sum() >= 3:
            xs = kp[local_idx, visible, 0]
            ys = kp[local_idx, visible, 1]
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            # Keep a conservative body margin; GVHMR applies its own 1.2x
            # enlargement when converting xyxy -> center/size.
            pad_x = max(8.0, 0.08 * (x2 - x1))
            pad_y = max(8.0, 0.08 * (y2 - y1))
            xyxy[local_idx] = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
            valid_frame_count += 1

    for axis in range(4):
        valid = np.isfinite(xyxy[:, axis])
        xyxy[:, axis] = _interp_column(xyxy[:, axis], valid)
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height - 1)
    xyxy[:, 2] = np.maximum(xyxy[:, 2], xyxy[:, 0] + 32)
    xyxy[:, 3] = np.maximum(xyxy[:, 3], xyxy[:, 1] + 32)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, width - 1)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, height - 1)

    return (
        torch.from_numpy(kp),
        get_bbx_xys_from_xyxy(torch.from_numpy(xyxy), base_enlarge=1.2).float(),
        {"valid_bbox_frames": valid_frame_count, "total_frames": length},
    )


def _save_slam_if_available(simplevo_npz: Path, output_path: Path) -> bool:
    if not simplevo_npz.exists():
        return False
    data = np.load(simplevo_npz)
    if "T_w2c" not in data:
        return False
    trajectory = np.asarray(data["T_w2c"])
    torch.save(trajectory, output_path)
    return True


def build_adapter(
    csv_path: Path,
    video_path: Path,
    clip_path: Path,
    output_root: Path,
    simplevo_npz: Path,
    player_id: int,
    start_frame0: int,
    end_frame0: int,
) -> None:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开短片: {clip_path}")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    expected_length = end_frame0 - start_frame0
    if length != expected_length:
        raise ValueError(f"短片帧数为 {length}，但期望 {expected_length}: {clip_path}")

    rows = _read_rtmlib_csv(csv_path)
    kp2d, bbx_xys, stats = _load_clip_keypoints(
        rows, player_id, start_frame0, length, width, height
    )

    preprocess_dir = output_root / Path(clip_path).stem / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"bbx_xyxy": None, "bbx_xys": bbx_xys},
        preprocess_dir / "bbx.pt",
    )
    torch.save(kp2d, preprocess_dir / "vitpose.pt")
    torch.save(kp2d, preprocess_dir / "rtmlib_kp2d.pt")
    slam_saved = _save_slam_if_available(simplevo_npz, preprocess_dir / "slam_results.pt")

    metadata = {
        "method": "GVHMR with existing BadmintonPose RTMLib COCO17 keypoints",
        "official_vitpose_bypassed": True,
        "source_csv": str(csv_path),
        "source_video": str(video_path),
        "clip_video": str(clip_path),
        "player_id": player_id,
        "source_frame_range_zero_based_end_exclusive": [start_frame0, end_frame0],
        "csv_frame_mapping": "clip frame i uses CSV frame_id=start_frame0+i+1",
        "shape_kp2d": list(kp2d.shape),
        "shape_bbx_xys": list(bbx_xys.shape),
        "simplevo_cached": slam_saved,
        **stats,
        "note": "kp2d is RTMLib Wholebody first 17 COCO keypoints; HMR2 image features are still required.",
    }
    (preprocess_dir / "rtmlib_adapter_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"preprocess_dir": str(preprocess_dir), **metadata}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simplevo", type=Path, default=DEFAULT_SIMPLEVO)
    parser.add_argument("--player-id", type=int, default=0)
    parser.add_argument("--start-frame0", type=int, default=120)
    parser.add_argument("--end-frame0", type=int, default=240)
    args = parser.parse_args()
    build_adapter(
        csv_path=args.csv,
        video_path=args.video,
        clip_path=args.clip,
        output_root=args.output_root,
        simplevo_npz=args.simplevo,
        player_id=args.player_id,
        start_frame0=args.start_frame0,
        end_frame0=args.end_frame0,
    )


if __name__ == "__main__":
    main()
