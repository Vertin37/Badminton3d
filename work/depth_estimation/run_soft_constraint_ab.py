"""Build a minimal, reversible A/B trajectory comparison.

A is the read-only GVHMR short slice. B applies only a low-frequency depth
drift correction along the calibrated camera viewing ray. The original full
GVHMR bundle is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-csv", type=Path, required=True)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=15)
    return parser.parse_args()


def moving_average(values, window: int):
    import numpy as np

    if window < 3 or window % 2 == 0:
        raise ValueError("smooth-window must be an odd integer >= 3")
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def jerk_stats(points, fps: float) -> dict[str, float]:
    import numpy as np

    if len(points) < 4:
        return {"mean_mps3": float("nan"), "median_mps3": float("nan"), "max_mps3": float("nan")}
    jerk = np.diff(points, n=3, axis=0) * fps**3
    norms = np.linalg.norm(jerk, axis=1)
    return {
        "mean_mps3": float(np.mean(norms)),
        "median_mps3": float(np.median(norms)),
        "max_mps3": float(np.max(norms)),
    }


def court_stats(points, width: float, length: float) -> dict[str, float]:
    import numpy as np

    half_w, half_l = width / 2.0, length / 2.0
    inside = (
        (np.abs(points[:, 0]) <= half_w)
        & (np.abs(points[:, 1]) <= half_l)
    )
    violation = np.maximum(np.abs(points[:, :2]) - np.array([half_w, half_l]), 0.0)
    return {
        "inside_xy_ratio": float(np.mean(inside)),
        "max_xy_boundary_violation_m": float(np.max(violation)),
        "median_xy_boundary_violation_m": float(np.median(np.linalg.norm(violation, axis=1))),
    }


def ground_stats(vertices, foot_confidence):
    import numpy as np

    min_z = np.min(vertices[:, :, 2], axis=1)
    contact = foot_confidence >= 0.6
    selected = min_z[contact] if np.any(contact) else min_z
    return {
        "contact_frame_count": int(contact.sum()),
        "contact_median_abs_min_vertex_z_m": float(np.median(np.abs(selected))),
        "contact_median_min_vertex_z_m": float(np.median(selected)),
        "contact_ground_below_minus_005_ratio": float(np.mean(selected < -0.05)),
        "all_frame_min_vertex_z_range_m": [float(min_z.min()), float(min_z.max())],
    }


def main() -> None:
    args = parse_args()
    for path in (args.alignment_csv, args.baseline_npz, args.scene_metadata):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    a_dir = args.output_dir / "ab_results" / "A_gvhmr_only"
    b_dir = args.output_dir / "ab_results" / "B_depth_soft_constraint"
    a_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    alignment = pd.read_csv(args.alignment_csv).sort_values(["frame_id", "player_id"])
    meta = json.loads(args.scene_metadata.read_text(encoding="utf-8"))
    calibration = meta["calibration"]
    rotation = np.asarray(calibration["court_to_camera_R"], dtype=np.float64)
    width = float(np.asarray(np.load(args.baseline_npz, allow_pickle=False)["court_width_m"]).reshape(()))
    length = float(np.asarray(np.load(args.baseline_npz, allow_pickle=False)["court_length_m"]).reshape(()))
    baseline = np.load(args.baseline_npz, allow_pickle=False)
    fps = float(np.asarray(baseline["fps"]).reshape(()))
    start = int(alignment.frame_id.min())
    end = int(alignment.frame_id.max())
    frames = np.arange(start, end + 1)

    outputs: dict[str, dict[str, object]] = {}
    metric_rows: list[dict[str, object]] = []
    for player_id in (0, 1):
        g = alignment[alignment.player_id == player_id].sort_values("frame_id")
        if not np.array_equal(g.frame_id.to_numpy(), frames):
            raise ValueError(f"Alignment rows are not contiguous for player {player_id}")
        root_a = np.asarray(baseline[f"p{player_id}_root_court"][start - 1 : end], dtype=np.float32)
        vertices_a = np.asarray(baseline[f"p{player_id}_vertices_court"][start - 1 : end], dtype=np.float32)
        foot = np.asarray(baseline[f"p{player_id}_foot_static_confidence"][start - 1 : end], dtype=np.float64)
        observed = g.depth_median.to_numpy(dtype=float)
        aligned = g.aligned_depth_m.to_numpy(dtype=float)
        roi_conf = np.clip(g.roi_confidence.to_numpy(dtype=float), 0.0, 1.0)
        iqr = np.maximum(g.depth_iqr.to_numpy(dtype=float), 0.0)
        depth_quality = roi_conf * np.clip(1.0 - iqr / np.maximum(observed, 1.0), 0.15, 1.0)
        depth_quality[~np.isfinite(depth_quality)] = 0.0
        airborne = np.clip(1.0 - foot, 0.0, 1.0)
        # Depth receives more weight in airborne frames, while the existing
        # ground/court baseline dominates when foot contact confidence is high.
        weight = np.clip((0.20 + 0.65 * airborne) * depth_quality, 0.0, 0.85)
        aligned_smooth = moving_average(aligned, args.smooth_window)
        root_camera_a = root_a @ rotation.T + np.asarray(calibration["court_to_camera_t"], dtype=np.float64)
        delta_z = np.clip(aligned_smooth - root_camera_a[:, 2], -0.75, 0.75)
        applied_delta_z = weight * delta_z
        # Realize the requested camera-depth change using court-plane motion
        # only. This keeps every court Z value (and therefore jump height)
        # exactly unchanged.
        camera_depth_axis_xy = rotation[2, :2]
        camera_depth_axis_xy_norm_sq = float(camera_depth_axis_xy @ camera_depth_axis_xy)
        if camera_depth_axis_xy_norm_sq <= 1e-8:
            raise ValueError("Camera depth axis has no usable court-plane projection")
        delta_court = np.zeros((len(applied_delta_z), 3), dtype=np.float64)
        delta_court[:, :2] = (
            applied_delta_z[:, None]
            * camera_depth_axis_xy[None, :]
            / camera_depth_axis_xy_norm_sq
        )
        root_b = root_a + delta_court.astype(np.float32)
        vertices_b = vertices_a + delta_court[:, None, :].astype(np.float32)
        root_camera_b = root_b @ rotation.T + np.asarray(calibration["court_to_camera_t"], dtype=np.float64)

        a_file = a_dir / f"p{player_id}_trajectory_bundle.npz"
        b_file = b_dir / f"p{player_id}_trajectory_bundle.npz"
        np.savez_compressed(
            a_file,
            root_court=root_a,
            vertices_court=vertices_a,
            faces=baseline["faces"],
            fps=np.float32(fps),
            player_id=np.int32(player_id),
            frame_ids=frames,
            source_baseline=str(args.baseline_npz.resolve()),
            variant="A_gvhmr_only",
        )
        np.savez_compressed(
            b_file,
            root_court=root_b,
            vertices_court=vertices_b,
            faces=baseline["faces"],
            fps=np.float32(fps),
            player_id=np.int32(player_id),
            frame_ids=frames,
            source_baseline=str(args.baseline_npz.resolve()),
            variant="B_depth_soft_constraint_low_frequency",
            applied_camera_delta_z_m=applied_delta_z,
            depth_weight=weight,
        )

        depth_error_a = root_camera_a[:, 2] - aligned_smooth
        depth_error_b = root_camera_b[:, 2] - aligned_smooth
        for variant, roots, verts, root_cam, depth_error in (
            ("A_gvhmr_only", root_a, vertices_a, root_camera_a, depth_error_a),
            ("B_depth_soft_constraint", root_b, vertices_b, root_camera_b, depth_error_b),
        ):
            metric_rows.append(
                {
                    "variant": variant,
                    "player_id": player_id,
                    "temporal_jerk": jerk_stats(roots, fps),
                    "depth_consistency": {
                        "median_abs_error_m": float(np.median(np.abs(depth_error))),
                        "rmse_error_m": float(np.sqrt(np.mean(depth_error**2))),
                        "p95_abs_error_m": float(np.percentile(np.abs(depth_error), 95)),
                    },
                    "ground_consistency": ground_stats(verts, foot),
                    "court_consistency": court_stats(roots, width, length),
                    "median_applied_camera_delta_z_m": float(np.median(np.abs(applied_delta_z)))
                    if variant.startswith("B")
                    else 0.0,
                    "median_depth_weight": float(np.median(weight)) if variant.startswith("B") else 0.0,
                }
            )
        outputs[str(player_id)] = {
            "player_id": player_id,
            "applied_camera_delta_z_range_m": [float(applied_delta_z.min()), float(applied_delta_z.max())],
            "median_depth_weight": float(np.median(weight)),
            "airborne_weight_range": [float(airborne.min()), float(airborne.max())],
            "alignment_depth_smoothed_range_m": [float(aligned_smooth.min()), float(aligned_smooth.max())],
            "A_file": str(a_file.resolve()),
            "B_file": str(b_file.resolve()),
        }

    # Compact visual acceptance aid: court-plane path plus camera-depth trace.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = {0: "tab:orange", 1: "tab:cyan"}
    for player_id in (0, 1):
        g = alignment[alignment.player_id == player_id].sort_values("frame_id")
        root_a = np.asarray(baseline[f"p{player_id}_root_court"][start - 1 : end], dtype=np.float32)
        b_file = b_dir / f"p{player_id}_trajectory_bundle.npz"
        root_b = np.load(b_file, allow_pickle=False)["root_court"]
        axes[0].plot(root_a[:, 0], root_a[:, 1], "--", color=colors[player_id], alpha=0.65, label=f"P{player_id} A")
        axes[0].plot(root_b[:, 0], root_b[:, 1], "-", color=colors[player_id], label=f"P{player_id} B")
        axes[1].plot(frames, g.root_camera_z_m, "--", color=colors[player_id], alpha=0.65, label=f"P{player_id} GVHMR")
        axes[1].plot(frames, g.aligned_depth_m, ":", color=colors[player_id], alpha=0.85, label=f"P{player_id} aligned depth")
    axes[0].set_title("Court-plane root trajectory")
    axes[0].set_xlabel("court X (m)")
    axes[0].set_ylabel("court Y (m)")
    axes[0].legend(fontsize=8)
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].set_title("Camera-depth alignment")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("camera depth Z (m)")
    axes[1].legend(fontsize=8)
    fig.savefig(args.output_dir / "diagnostics" / "ab_trajectory_alignment.png", dpi=150)
    plt.close(fig)

    metrics = {
        "input_alignment_csv": str(args.alignment_csv.resolve()),
        "source_baseline_npz": str(args.baseline_npz.resolve()),
        "frame_start": start,
        "frame_end": end,
        "frame_count": int(len(frames)),
        "fps": fps,
        "court_width_m": width,
        "court_length_m": length,
        "fusion": {
            "method": "scale_offset_aligned_depth_plus_low_frequency_drift_correction",
            "smooth_window_frames": args.smooth_window,
            "delta_camera_axis": "camera_z_only",
            "weight_rule": "clip((0.20 + 0.65 * airborne_weight) * depth_quality, 0, 0.85)",
            "airborne_weight": "clip(1 - GVHMR foot_static_confidence, 0, 1)",
            "hard_root_z_assignment": False,
        },
        "players": outputs,
        "comparisons": metric_rows,
        "A_files": [
            str((args.output_dir / "ab_results" / "A_gvhmr_only" / f"p{player_id}_trajectory_bundle.npz").resolve())
            for player_id in (0, 1)
        ],
        "B_files": [
            str((args.output_dir / "ab_results" / "B_depth_soft_constraint" / f"p{player_id}_trajectory_bundle.npz").resolve())
            for player_id in (0, 1)
        ],
        "diagnostic_plot": str((args.output_dir / "diagnostics" / "ab_trajectory_alignment.png").resolve()),
    }
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "ab_summary.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
