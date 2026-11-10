"""Align person ROI depth observations with the read-only GVHMR camera depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person-depth", type=Path, required=True)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def fit_line(x, y):
    import numpy as np

    if x.size < 3:
        raise ValueError("At least three valid observations are required for alignment")
    scale, offset = np.polyfit(x, y, 1)
    pred = scale * x + offset
    residual = y - pred
    return float(scale), float(offset), residual


def main() -> None:
    args = parse_args()
    for path in (args.person_depth, args.baseline_npz, args.scene_metadata):
        if not path.exists():
            raise FileNotFoundError(path)

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    person = pd.read_csv(args.person_depth).sort_values(["player_id", "frame_id"])
    meta = json.loads(args.scene_metadata.read_text(encoding="utf-8"))
    calibration = meta["calibration"]
    rotation = np.asarray(calibration["court_to_camera_R"], dtype=np.float64)
    translation = np.asarray(calibration["court_to_camera_t"], dtype=np.float64)
    baseline = np.load(args.baseline_npz, allow_pickle=False)
    fps = float(np.asarray(baseline["fps"]).reshape(()))

    start = int(person.frame_id.min())
    end = int(person.frame_id.max())
    rows: list[dict[str, float | int | bool]] = []
    summary: dict[str, object] = {
        "person_depth_csv": str(args.person_depth.resolve()),
        "baseline_npz": str(args.baseline_npz.resolve()),
        "scene_metadata": str(args.scene_metadata.resolve()),
        "frame_start": start,
        "frame_end": end,
        "frame_count": end - start + 1,
        "fps": fps,
        "alignment_equation": "root_camera_z_m = scale * depth_median + offset",
        "camera_transform": {
            "court_to_camera_R": rotation.tolist(),
            "court_to_camera_t": translation.tolist(),
            "rotation_orthogonality_error": float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))),
            "rotation_determinant": float(np.linalg.det(rotation)),
        },
        "players": {},
    }

    for player_id in (0, 1):
        g = person[person.player_id == player_id].copy().sort_values("frame_id")
        if len(g) != end - start + 1:
            raise ValueError(f"Player {player_id} does not have one row per selected frame")
        root_court = np.asarray(
            baseline[f"p{player_id}_root_court"][start - 1 : end], dtype=np.float64
        )
        root_camera = root_court @ rotation.T + translation
        observed = g.depth_median.to_numpy(dtype=float)
        trimmed = g.depth_trimmed_mean.to_numpy(dtype=float)
        roi_conf = g.roi_confidence.to_numpy(dtype=float)
        finite = np.isfinite(observed) & np.isfinite(root_camera[:, 2]) & (observed > 0)
        scale, offset, residual_valid = fit_line(observed[finite], root_camera[finite, 2])
        aligned = scale * observed + offset
        residual = root_camera[:, 2] - aligned
        model_delta = np.diff(observed)
        root_delta = np.diff(root_camera[:, 2])
        residual_delta = np.diff(residual)
        abs_residual = np.abs(residual[finite])
        corr = float(np.corrcoef(observed[finite], root_camera[finite, 2])[0, 1])

        contact = np.asarray(baseline[f"p{player_id}_foot_static_confidence"][start - 1 : end])
        for index, (_, row) in enumerate(g.iterrows()):
            rows.append(
                {
                    "frame_id": int(row.frame_id),
                    "player_id": player_id,
                    "depth_median": float(observed[index]) if np.isfinite(observed[index]) else float("nan"),
                    "depth_trimmed_mean": float(trimmed[index]) if np.isfinite(trimmed[index]) else float("nan"),
                    "depth_iqr": float(row.depth_iqr) if np.isfinite(row.depth_iqr) else float("nan"),
                    "valid_ratio": float(row.valid_ratio),
                    "roi_confidence": float(roi_conf[index]),
                    "root_court_x_m": float(root_court[index, 0]),
                    "root_court_y_m": float(root_court[index, 1]),
                    "root_court_z_m": float(root_court[index, 2]),
                    "root_camera_x_m": float(root_camera[index, 0]),
                    "root_camera_y_m": float(root_camera[index, 1]),
                    "root_camera_z_m": float(root_camera[index, 2]),
                    "aligned_depth_m": float(aligned[index]) if np.isfinite(aligned[index]) else float("nan"),
                    "alignment_residual_m": float(residual[index]) if np.isfinite(residual[index]) else float("nan"),
                    "foot_static_confidence": float(contact[index]),
                    "airborne_weight": float(np.clip(1.0 - contact[index], 0.0, 1.0)),
                    "valid_alignment": bool(finite[index]),
                }
            )

        summary["players"][str(player_id)] = {
            "valid_rows": int(finite.sum()),
            "valid_ratio": float(finite.mean()),
            "correlation": corr,
            "scale": scale,
            "offset_m": offset,
            "observed_depth_range_m": [float(np.nanmin(observed)), float(np.nanmax(observed))],
            "root_camera_depth_range_m": [float(root_camera[:, 2].min()), float(root_camera[:, 2].max())],
            "median_abs_residual_m": float(np.median(abs_residual)),
            "rmse_residual_m": float(np.sqrt(np.mean(residual[finite] ** 2))),
            "p95_abs_residual_m": float(np.percentile(abs_residual, 95)),
            "median_abs_observed_delta_m": float(np.median(np.abs(model_delta))),
            "median_abs_root_delta_m": float(np.median(np.abs(root_delta))),
            "median_abs_residual_delta_m": float(np.median(np.abs(residual_delta))),
            "p95_abs_residual_delta_m": float(np.percentile(np.abs(residual_delta), 95)),
            "median_foot_static_confidence": float(np.median(contact)),
            "low_contact_rows": int((contact < 0.6).sum()),
        }

    output = pd.DataFrame(rows).sort_values(["frame_id", "player_id"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, float_format="%.8f")
    summary_path = args.output_csv.with_name("depth_alignment_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
