"""Extract robust torso/pelvis depth observations for both tracked players."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BODY_KPS = (5, 6, 11, 12)  # COCO-17: left/right shoulder and left/right hip.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-csv", type=Path, required=True)
    parser.add_argument("--depth-raw", type=Path, required=True)
    parser.add_argument("--depth-quality", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def finite_confidence(values):
    import numpy as np

    return np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def main() -> None:
    args = parse_args()
    for path in (args.pose_csv, args.depth_raw, args.depth_quality):
        if not path.exists():
            raise FileNotFoundError(path)

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    quality = json.loads(args.depth_quality.read_text(encoding="utf-8"))
    start = int(quality["selected_start_frame_1based"])
    end = int(quality["selected_end_frame_1based"])
    depth = np.load(args.depth_raw, mmap_mode="r")
    if depth.ndim != 3 or depth.shape[0] != end - start + 1:
        raise ValueError(f"Depth array shape does not match quality manifest: {depth.shape}")
    _, image_height, image_width = depth.shape
    source_width = int(quality.get("source_width", image_width))
    source_height = int(quality.get("source_height", image_height))
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Invalid source dimensions in depth quality manifest")
    scale_x = image_width / source_width
    scale_y = image_height / source_height

    pose = pd.read_csv(args.pose_csv)
    pose = pose[pose.frame_id.between(start, end)].copy()
    pose = pose[pose.player_id.isin([0, 1])]
    if pose.empty:
        raise ValueError("No pose rows for selected frames and players")

    rows: list[dict[str, float | int | bool | str]] = []
    for frame_id in range(start, end + 1):
        frame_pose = pose[pose.frame_id == frame_id]
        depth_map = np.asarray(depth[frame_id - start], dtype=np.float32)
        for player_id in (0, 1):
            g = frame_pose[frame_pose.player_id == player_id]
            kps = g[g.keypoint_id.isin(BODY_KPS)].set_index("keypoint_id")
            valid_kps = []
            for kp in BODY_KPS:
                if kp in kps.index:
                    row = kps.loc[kp]
                    x = float(row.x) * scale_x
                    y = float(row.y) * scale_y
                    conf = float(finite_confidence(np.asarray([row.confidence]))[0])
                    if np.isfinite(x) and np.isfinite(y) and conf >= 0.2:
                        valid_kps.append((x, y, conf))

            scale_values = g.body_scale.to_numpy(dtype=float) if "body_scale" in g else np.array([])
            scale_values = scale_values[np.isfinite(scale_values) & (scale_values > 1)]
            source_body_scale = float(np.median(scale_values)) if scale_values.size else 100.0
            body_scale = source_body_scale * 0.5 * (scale_x + scale_y)
            if len(valid_kps) >= 2:
                xs = np.asarray([v[0] for v in valid_kps], dtype=float)
                ys = np.asarray([v[1] for v in valid_kps], dtype=float)
                x_span = max(float(xs.max() - xs.min()), 0.35 * body_scale, 20.0)
                y_span = max(float(ys.max() - ys.min()), 0.45 * body_scale, 30.0)
                x1 = max(0, int(np.floor(xs.min() - 0.18 * x_span)))
                x2 = min(image_width, int(np.ceil(xs.max() + 0.18 * x_span)))
                y1 = max(0, int(np.floor(ys.min() - 0.15 * y_span)))
                y2 = min(image_height, int(np.ceil(ys.max() + 0.18 * y_span)))
                kp_conf = float(np.mean([v[2] for v in valid_kps]) * len(valid_kps) / len(BODY_KPS))
                missing_flag = len(valid_kps) < len(BODY_KPS)
            else:
                center_x = float(g.x.median()) * scale_x if not g.empty else float("nan")
                center_y = float(g.y.median()) * scale_y if not g.empty else float("nan")
                half_w = max(0.25 * body_scale, 25.0)
                half_h = max(0.35 * body_scale, 35.0)
                x1 = max(0, int(np.floor(center_x - half_w))) if np.isfinite(center_x) else 0
                x2 = min(image_width, int(np.ceil(center_x + half_w))) if np.isfinite(center_x) else 0
                y1 = max(0, int(np.floor(center_y - half_h))) if np.isfinite(center_y) else 0
                y2 = min(image_height, int(np.ceil(center_y + half_h))) if np.isfinite(center_y) else 0
                kp_conf = 0.0
                missing_flag = True

            if x2 > x1 and y2 > y1:
                roi = depth_map[y1:y2, x1:x2]
                finite = np.isfinite(roi) & (roi > 0)
                vals = roi[finite]
            else:
                roi = np.empty((0, 0), dtype=np.float32)
                finite = np.empty((0, 0), dtype=bool)
                vals = np.empty(0, dtype=np.float32)

            valid_ratio = float(finite.mean()) if finite.size else 0.0
            if vals.size:
                vals = np.asarray(vals, dtype=np.float32)
                q05, q95 = np.percentile(vals, [5, 95]) if vals.size >= 4 else (vals.min(), vals.max())
                trimmed = vals[(vals >= q05) & (vals <= q95)]
                if not trimmed.size:
                    trimmed = vals
                median = float(np.median(vals))
                trimmed_mean = float(np.mean(trimmed))
                q25, q75 = np.percentile(vals, [25, 75])
                iqr = float(q75 - q25)
                min_depth = float(vals.min())
                max_depth = float(vals.max())
            else:
                median = trimmed_mean = iqr = min_depth = max_depth = float("nan")

            roi_confidence = float(kp_conf * min(max(valid_ratio, 0.0), 1.0))
            rows.append(
                {
                    "frame_id": frame_id,
                    "player_id": player_id,
                    "roi_type": "torso_pelvis",
                    "roi_x1": x1,
                    "roi_y1": y1,
                    "roi_x2": x2,
                    "roi_y2": y2,
                    "valid_pixel_count": int(vals.size),
                    "valid_ratio": valid_ratio,
                    "depth_median": median,
                    "depth_trimmed_mean": trimmed_mean,
                    "depth_iqr": iqr,
                    "depth_min": min_depth,
                    "depth_max": max_depth,
                    "roi_confidence": roi_confidence,
                    "keypoint_confidence": kp_conf,
                    "valid_body_keypoint_count": len(valid_kps),
                    "occlusion_or_missing_flag": bool(missing_flag or valid_ratio < 0.5),
                }
            )

    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False, float_format="%.8f")
    summary = {
        "input_pose_csv": str(args.pose_csv.resolve()),
        "input_depth_raw": str(args.depth_raw.resolve()),
        "frame_start": start,
        "frame_end": end,
        "frame_count": end - start + 1,
        "source_dimensions": [source_width, source_height],
        "depth_dimensions": [image_width, image_height],
        "pose_to_depth_scale": [scale_x, scale_y],
        "players": {},
        "output_csv": str(args.output_csv.resolve()),
    }
    for player_id, g in out.groupby("player_id"):
        valid = np.isfinite(g.depth_median.to_numpy()) & (g.roi_confidence.to_numpy() >= 0.2)
        summary["players"][str(int(player_id))] = {
            "rows": int(len(g)),
            "usable_rows": int(valid.sum()),
            "usable_ratio": float(valid.mean()),
            "median_valid_ratio": float(g.valid_ratio.median()),
            "median_depth_m": float(g.depth_median.median()),
            "median_iqr_m": float(g.depth_iqr.median()),
            "missing_or_occluded_rows": int(g.occlusion_or_missing_flag.sum()),
        }
    summary_path = args.output_csv.with_name("person_depth_quality.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
