"""Apply continuous SMPL-X sole locking and court-floor alignment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.pose_biomechanics.foot_contact import (  # noqa: E402
    FootContactConfig,
    refine_foot_contacts,
    smplx_sole_vertex_indices,
)


DEFAULT_SMPLX = PROJECT / "work" / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce planted-foot skating and align support soles to court Z=0")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smplx-model", type=Path, default=DEFAULT_SMPLX)
    parser.add_argument("--scene-metadata", type=Path)
    parser.add_argument("--min-contact-confidence", type=float)
    parser.add_argument("--stationary-local-speed-mps", type=float)
    parser.add_argument("--max-stationary-height-gap-m", type=float)
    parser.add_argument("--foot-lock-weight", type=float)
    parser.add_argument("--floor-contact-weight", type=float)
    parser.add_argument("--max-xy-correction-m", type=float)
    parser.add_argument("--max-z-correction-m", type=float)
    parser.add_argument("--max-correction-speed-mps", type=float)
    parser.add_argument("--max-root-xy-speed-mps", type=float)
    parser.add_argument("--max-root-z-speed-mps", type=float)
    parser.add_argument("--court-margin-m", type=float)
    return parser.parse_args()


def _config(args: argparse.Namespace) -> FootContactConfig:
    config = FootContactConfig()
    overrides = {
        "min_contact_confidence": args.min_contact_confidence,
        "stationary_local_speed_mps": args.stationary_local_speed_mps,
        "max_stationary_height_gap_m": args.max_stationary_height_gap_m,
        "foot_lock_weight": args.foot_lock_weight,
        "floor_contact_weight": args.floor_contact_weight,
        "max_xy_correction_m": args.max_xy_correction_m,
        "max_z_correction_m": args.max_z_correction_m,
        "max_correction_speed_mps": args.max_correction_speed_mps,
        "max_root_xy_speed_mps": args.max_root_xy_speed_mps,
        "max_root_z_speed_mps": args.max_root_z_speed_mps,
        "court_margin_m": args.court_margin_m,
    }
    values = {key: value for key, value in overrides.items() if value is not None}
    if values:
        config = replace(config, **values)
    if not (0.0 <= config.min_contact_confidence < 1.0):
        raise ValueError("Minimum contact confidence must be in [0, 1)")
    positive = (
        config.stationary_local_speed_mps,
        config.max_stationary_height_gap_m,
        config.foot_lock_weight,
        config.floor_contact_weight,
        config.max_xy_correction_m,
        config.max_z_correction_m,
        config.max_correction_speed_mps,
        config.max_root_xy_speed_mps,
        config.max_root_z_speed_mps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Foot-contact weights and limits must be positive")
    if config.court_margin_m < 0:
        raise ValueError("Court margin must be non-negative")
    return config


def _prepare_output(output_dir: Path, inputs: list[Path]) -> None:
    resolved = output_dir.resolve()
    if resolved in {path.resolve().parent for path in inputs}:
        raise ValueError("Output directory must not be an input directory")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def _players(keys: list[str]) -> list[int]:
    roots = {int(match.group(1)) for key in keys if (match := re.fullmatch(r"p(\d+)_root_court", key))}
    vertices = {
        int(match.group(1)) for key in keys if (match := re.fullmatch(r"p(\d+)_vertices_court", key))
    }
    return sorted(roots & vertices)


def run(args: argparse.Namespace) -> None:
    config = _config(args)
    args.bundle = args.bundle.resolve()
    args.output_dir = args.output_dir.resolve()
    args.smplx_model = args.smplx_model.resolve()
    input_paths = [args.bundle, args.smplx_model]
    if args.scene_metadata is not None:
        args.scene_metadata = args.scene_metadata.resolve()
        input_paths.append(args.scene_metadata)
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    _prepare_output(args.output_dir, input_paths)

    source = np.load(args.bundle, allow_pickle=False)
    players = _players(source.files)
    if not players:
        raise ValueError("Bundle has no pN_root_court/pN_vertices_court pairs")
    body = np.load(args.smplx_model)
    sole_indices = smplx_sole_vertex_indices(body["weights"], body["v_template"], config)
    fps = float(source["fps"])
    court_half_extents = None
    if "court_width_m" in source.files and "court_length_m" in source.files:
        court_half_extents = np.array(
            [
                0.5 * float(source["court_width_m"]) + config.court_margin_m,
                0.5 * float(source["court_length_m"]) + config.court_margin_m,
            ],
            dtype=np.float64,
        )
    output = {key: source[key] for key in source.files}
    metrics: dict[str, dict] = {}
    rows: list[dict] = []

    for player_id in players:
        vertices = source[f"p{player_id}_vertices_court"]
        root = source[f"p{player_id}_root_court"]
        lr_key = f"p{player_id}_foot_static_confidence_lr"
        fallback = lr_key not in source.files
        if fallback:
            scalar = np.asarray(source[f"p{player_id}_foot_static_confidence"], dtype=np.float32)
            confidence_lr = np.repeat(scalar[:, None], 2, axis=1)
        else:
            confidence_lr = source[lr_key]
        result = refine_foot_contacts(
            vertices,
            root,
            confidence_lr,
            sole_indices,
            fps,
            config,
            court_half_extents_m=court_half_extents,
        )
        output[f"p{player_id}_vertices_court"] = result.corrected_vertices
        output[f"p{player_id}_root_court"] = result.corrected_root
        output[f"p{player_id}_foot_contact_correction_xyz_m"] = result.correction_xyz_m
        output[f"p{player_id}_foot_stationary_score_lr"] = result.stationary_scores
        output[f"p{player_id}_support_foot"] = result.support_foot
        output[f"p{player_id}_sole_points_before"] = result.sole_points_before
        output[f"p{player_id}_sole_points_after"] = result.sole_points_after
        metrics[str(player_id)] = {
            **result.metrics,
            "left_right_confidence_fallback": fallback,
            "max_xy_correction_m": float(np.linalg.norm(result.correction_xyz_m[:, :2], axis=1).max()),
            "max_abs_z_correction_m": float(np.abs(result.correction_xyz_m[:, 2]).max()),
            "local_smplx_pose_preserved": True,
        }
        if court_half_extents is not None:
            inside = (
                (np.abs(result.corrected_root[:, 0]) <= court_half_extents[0] + 1e-6)
                & (np.abs(result.corrected_root[:, 1]) <= court_half_extents[1] + 1e-6)
            )
            metrics[str(player_id)]["court_inside_ratio_with_margin"] = float(np.mean(inside))
        for frame in range(len(vertices)):
            support = int(result.support_foot[frame])
            rows.append(
                {
                    "frame_id": frame + 1,
                    "player_id": player_id,
                    "correction_x_m": result.correction_xyz_m[frame, 0],
                    "correction_y_m": result.correction_xyz_m[frame, 1],
                    "correction_z_m": result.correction_xyz_m[frame, 2],
                    "left_stationary_score": result.stationary_scores[frame, 0],
                    "right_stationary_score": result.stationary_scores[frame, 1],
                    "support_foot": support,
                    "support_sole_z_before_m": result.sole_points_before[frame, support, 2],
                    "support_sole_z_after_m": result.sole_points_after[frame, support, 2],
                }
            )

    output["foot_contact_refinement_version"] = np.asarray("continuous-v3")
    bundle_path = args.output_dir / "foot_contact_refined_smplx_court_animation.npz"
    np.savez_compressed(bundle_path, **output)
    pd.DataFrame(rows).to_csv(args.output_dir / "foot_contact_diagnostics.csv", index=False, float_format="%.8f")
    report = {
        "source_bundle": str(args.bundle),
        "output_bundle": str(bundle_path),
        "players": players,
        "fps": fps,
        "configuration": asdict(config),
        "player_metrics": metrics,
        "safety": {
            "source_overwritten": False,
            "rigid_root_translation_only": True,
            "local_smplx_pose_modified": False,
            "airborne_frames_forced_to_floor": False,
        },
    }
    (args.output_dir / "foot_contact_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.scene_metadata is not None:
        metadata = json.loads(args.scene_metadata.read_text(encoding="utf-8"))
        metadata["foot_contact_refinement"] = report
        metadata["bundle"] = bundle_path.name
        (args.output_dir / "scene_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run(parse_args())
