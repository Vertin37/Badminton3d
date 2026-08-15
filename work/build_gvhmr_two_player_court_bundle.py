"""Build a common court-coordinate animation bundle from native GVHMR meshes.

Each player is reconstructed independently by GVHMR, so their ``global``
coordinates are separate local reference frames.  This tool correctly merges
the two players through their *camera-coordinate* SMPL-X meshes and one shared
camera-to-court transform recovered from the already validated court
homography.  It never fits individual limbs to 2D points.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT = Path(r"D:\Projects\BadmintonPose")
GVHMR_ROOT = PROJECT / "work" / "GVHMR"
COURT_JSON = PROJECT / "outputs" / "test2_analysis" / "stable_geometry" / "court_detection_lines.json"
COURT_TRACK_CSV = PROJECT / "outputs" / "test2_analysis" / "stable_geometry" / "player_pole_geometry.csv"
P0_RESULT = PROJECT / "outputs" / "world_hmr_native_mesh_test" / "test2_p0_f120_240" / "1_hmr4d_results_native.pt"
P1_RESULT = PROJECT / "outputs" / "world_hmr_native_mesh_test" / "test2_p1_f120_240" / "1_hmr4d_results_native.pt"
DEFAULT_OUTPUT = PROJECT / "outputs" / "world_hmr_native_mesh_test" / "two_player_court_f120_240"

# The court corners only constrain a plane.  OpenCV's planar PnP convention
# chose the mathematically valid normal pointing *down*, which placed the
# camera underneath the floor and inverted every SMPL-X body.  Keep the
# existing validated X/Y court coordinates and resolve only this normal
# ambiguity: final Z is physical height above the court.  This is a single
# scene-coordinate conversion, not an animation or limb correction.
COURT_BASIS_RESOLUTION = np.diag([1.0, 1.0, -1.0])

# The stable 2D tracker predates the corner-PnP calibration and names the
# across-net direction in the opposite sense.  Its tracked image positions
# are correct; only the exported court-Y sign differs.  Reconcile that once
# at the interface, before it is fused with the PnP/GVHMR trajectory.
TRACK_XY_TO_PNP_COURT = np.diag([1.0, -1.0])


def _load_court_camera(anchor_frame: int, focal_mm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return the planar-PnP camera pose plus its physical Z-up resolution."""
    sys.path.insert(0, str(GVHMR_ROOT))
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor

    payload = json.loads(COURT_JSON.read_text(encoding="utf-8"))
    frame = next((item for item in payload["frames"] if item["frame_index"] == anchor_frame), None)
    if frame is None or frame.get("pole_mapped_court") is None:
        raise ValueError(f"No pole-mapped court calibration for frame {anchor_frame}")
    mapped = frame["pole_mapped_court"]
    object_points = np.c_[np.asarray(mapped["world_corners"], dtype=np.float64), np.zeros(4)]
    image_points = np.asarray(mapped["image_corners"], dtype=np.float64)
    k = create_camera_sensor(payload["width"], payload["height"], focal_mm)[2].cpu().numpy().astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        k,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP could not recover the shared court camera")
    r_court_to_cam, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, k, None)
    reprojection = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    camera_center_planar = (-r_court_to_cam.T @ tvec).reshape(3)
    # For final court points q, planar coordinates are S @ q, so the physical
    # camera extrinsic is R @ S and its center is S @ C_planar.
    r_final_court_to_cam = r_court_to_cam @ COURT_BASIS_RESOLUTION
    camera_center_final = COURT_BASIS_RESOLUTION @ camera_center_planar
    screen_up_final = r_final_court_to_cam.T @ np.array([0.0, -1.0, 0.0])
    camera_forward_final = r_final_court_to_cam.T @ np.array([0.0, 0.0, 1.0])
    screen_up_final /= np.linalg.norm(screen_up_final)
    camera_forward_final /= np.linalg.norm(camera_forward_final)
    metadata = {
        "anchor_frame_zero_based": anchor_frame,
        "court_world_corners_m": object_points.tolist(),
        "court_image_corners_px": image_points.tolist(),
        "K": k.tolist(),
        "reprojection_error_px": reprojection.tolist(),
        "reprojection_rms_px": float(np.sqrt(np.mean(reprojection**2))),
        "planar_court_to_camera_R": r_court_to_cam.tolist(),
        "court_to_camera_t": tvec.reshape(3).tolist(),
        "planar_camera_center_in_court_m": camera_center_planar.tolist(),
        "court_basis_resolution": {
            "matrix": COURT_BASIS_RESOLUTION.tolist(),
            "reason": "planar PnP returned the camera below the Z=0 court plane; Z was resolved to physical upward height while retaining the existing court X/Y homography",
        },
        "court_to_camera_R": r_final_court_to_cam.tolist(),
        "camera_center_in_court_m": camera_center_final.tolist(),
        "source_matched_view": {
            "camera_forward": camera_forward_final.tolist(),
            "screen_up": screen_up_final.tolist(),
        },
    }
    return r_court_to_cam, tvec.reshape(3), k, metadata


def _camera_to_court(points: np.ndarray, r_court_to_cam: np.ndarray, t_court_to_cam: np.ndarray) -> np.ndarray:
    """Map camera points to the final court frame (X/Y floor, Z physical up)."""
    planar_court = (points - t_court_to_cam.reshape(1, 1, 3)) @ r_court_to_cam
    return planar_court @ COURT_BASIS_RESOLUTION


def _load_native_incam_vertices(result_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not result_path.exists():
        raise FileNotFoundError(f"Native GVHMR result missing: {result_path}")
    result = torch.load(result_path, map_location="cpu")
    params = result["smpl_params_incam"]
    from hmr4d.utils.smplx_utils import make_smplx

    model = make_smplx("supermotion").eval().cuda()
    params_gpu = {key: value.cuda(non_blocking=True) for key, value in params.items()}
    with torch.no_grad():
        vertices = model(**params_gpu).vertices.detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    root = params["transl"].numpy().astype(np.float32)
    return vertices, faces, root


def _load_ground_measurements(source_start_frame0: int, frames: int) -> dict[int, np.ndarray]:
    """Read stable ground tracks and align their court axes to corner PnP."""
    if not COURT_TRACK_CSV.exists():
        raise FileNotFoundError(f"Stable court track missing: {COURT_TRACK_CSV}")
    tracks = {0: np.full((frames, 2), np.nan, dtype=np.float32), 1: np.full((frames, 2), np.nan, dtype=np.float32)}
    with COURT_TRACK_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_index = int(row["frame_id"]) - 1
            local_index = frame_index - source_start_frame0
            player_id = int(row["player_id"])
            if not (0 <= local_index < frames and player_id in tracks and row["pole_index"] == "0"):
                continue
            if row["metric_valid"] != "True":
                continue
            tracker_xy = np.array(
                [float(row["player_ground_x_m"]), float(row["player_ground_y_m"])], dtype=np.float32
            )
            tracks[player_id][local_index] = TRACK_XY_TO_PNP_COURT @ tracker_xy
    return tracks


def _load_foot_static_confidence(result_path: Path) -> np.ndarray:
    """Use GVHMR's learned foot-contact confidence, not a hand-written jump rule."""
    native_mesh = result_path.parent / "2_smplx_mesh_global.npz"
    if not native_mesh.exists():
        raise FileNotFoundError(f"Native GVHMR confidence output missing: {native_mesh}")
    confidence = np.load(native_mesh)["static_confidence"]
    if confidence.ndim != 2 or confidence.shape[1] < 4:
        raise ValueError(f"Unexpected static-confidence shape: {confidence.shape}")
    return confidence[:, :4].max(axis=1).astype(np.float32)


def _fuse_root_xy(
    raw_root_xy: np.ndarray,
    ground_xy: np.ndarray,
    foot_static_confidence: np.ndarray,
    contact_threshold: float,
    max_measurement_residual_m: float,
) -> tuple[np.ndarray, dict]:
    """Fuse court contacts with GVHMR motion; preserve airborne HMR translation.

    A ground measurement is admitted only when GVHMR itself predicts a static
    foot.  Between contacts, the root advances solely by GVHMR's native 3D
    displacement.  The correction is a rigid whole-body XY translation.
    """
    frame_count = len(raw_root_xy)
    if len(ground_xy) != frame_count or len(foot_static_confidence) != frame_count:
        raise ValueError("Root, court measurement and contact confidence lengths must match")
    inside_court = (np.abs(ground_xy[:, 0]) <= 3.2) & (np.abs(ground_xy[:, 1]) <= 6.8)
    candidate = np.isfinite(ground_xy).all(axis=1) & inside_court & (foot_static_confidence >= contact_threshold)
    candidate_indices = np.flatnonzero(candidate)
    if len(candidate_indices) == 0:
        raise RuntimeError("No learned foot-contact frames are available for court root alignment")

    seed = int(candidate_indices[0])
    fused = np.empty_like(raw_root_xy, dtype=np.float32)
    fused[seed] = ground_xy[seed]
    accepted = np.zeros(frame_count, dtype=bool)
    accepted[seed] = True
    residuals = []

    # Propagate from the first grounded frame.  Airborne frames have no court
    # measurement update, which prevents an elevated ankle from being forced
    # onto the ground plane.
    for index in range(seed + 1, frame_count):
        predicted = fused[index - 1] + (raw_root_xy[index] - raw_root_xy[index - 1])
        fused[index] = predicted
        if not candidate[index]:
            continue
        residual = float(np.linalg.norm(ground_xy[index] - predicted))
        if residual > max_measurement_residual_m:
            continue
        # Higher learned contact confidence gets a stronger measurement update.
        alpha = 0.30 + 0.55 * (foot_static_confidence[index] - contact_threshold) / (1.0 - contact_threshold)
        alpha = float(np.clip(alpha, 0.30, 0.85))
        fused[index] = (1.0 - alpha) * predicted + alpha * ground_xy[index]
        accepted[index] = True
        residuals.append(residual)

    # Fill leading airborne frames by reversing GVHMR's relative displacement
    # from the first learned-grounded anchor.
    for index in range(seed - 1, -1, -1):
        fused[index] = fused[index + 1] - (raw_root_xy[index + 1] - raw_root_xy[index])

    metadata = {
        "contact_threshold": contact_threshold,
        "max_measurement_residual_m": max_measurement_residual_m,
        "candidate_contact_frames": int(candidate.sum()),
        "accepted_ground_updates": int(accepted.sum()),
        "seed_frame": seed,
        "accepted_measurement_residual_median": float(np.median(residuals)) if residuals else None,
        "accepted_measurement_residual_max": float(np.max(residuals)) if residuals else None,
        "airborne_or_untrusted_frames": int((~accepted).sum()),
    }
    return fused, metadata


def build(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to regenerate SMPL-X mesh vertices")
    model_file = GVHMR_ROOT / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not model_file.exists():
        raise FileNotFoundError(f"Official SMPL-X file missing: {model_file}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.replace_existing_bundle:
        raise FileExistsError(f"Refusing to overwrite existing bundle: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(GVHMR_ROOT)
    r, t, _, calibration = _load_court_camera(args.anchor_frame0, args.f_mm)
    p0_cam, faces, p0_root_cam = _load_native_incam_vertices(args.p0_result)
    p1_cam, p1_faces, p1_root_cam = _load_native_incam_vertices(args.p1_result)
    if len(p0_cam) != len(p1_cam):
        raise ValueError(f"Frame mismatch: p0={len(p0_cam)}, p1={len(p1_cam)}")
    if not np.array_equal(faces, p1_faces):
        raise ValueError("The two SMPL-X meshes do not share the same face topology")

    # A reflection across the PnP plane reverses the mesh winding.  Restore
    # outward normals once for the entire mesh topology; vertex motion and
    # every articulated pose remain exactly the native GVHMR result.
    faces = faces[:, [0, 2, 1]]

    p0_court = _camera_to_court(p0_cam, r, t).astype(np.float32)
    p1_court = _camera_to_court(p1_cam, r, t).astype(np.float32)
    p0_root_court = _camera_to_court(p0_root_cam[:, None, :], r, t)[:, 0].astype(np.float32)
    p1_root_court = _camera_to_court(p1_root_cam[:, None, :], r, t)[:, 0].astype(np.float32)

    tracks = _load_ground_measurements(args.source_start_frame0, len(p0_court))
    p0_contact = _load_foot_static_confidence(args.p0_result)
    p1_contact = _load_foot_static_confidence(args.p1_result)
    p0_fused_xy, p0_fusion = _fuse_root_xy(
        p0_root_court[:, :2], tracks[0], p0_contact, args.contact_threshold, args.max_contact_residual_m
    )
    p1_fused_xy, p1_fusion = _fuse_root_xy(
        p1_root_court[:, :2], tracks[1], p1_contact, args.contact_threshold, args.max_contact_residual_m
    )
    p0_delta_xy = p0_fused_xy - p0_root_court[:, :2]
    p1_delta_xy = p1_fused_xy - p1_root_court[:, :2]
    p0_court[:, :, :2] += p0_delta_xy[:, None, :]
    p1_court[:, :, :2] += p1_delta_xy[:, None, :]
    p0_root_court[:, :2] = p0_fused_xy
    p1_root_court[:, :2] = p1_fused_xy

    # GVHMR's camera-space body translation has an unknown absolute vertical
    # zero.  Estimate *one shared* court-ground offset from the robust lower
    # envelope of both mesh sequences.  This is a rigid scene calibration, not
    # a per-frame root correction and it leaves every limb's 3D motion intact.
    raw_floor_samples = np.concatenate((p0_court[:, :, 2].min(axis=1), p1_court[:, :, 2].min(axis=1)))
    shared_ground_offset = -float(np.median(raw_floor_samples))
    p0_court[:, :, 2] += shared_ground_offset
    p1_court[:, :, 2] += shared_ground_offset
    p0_root_court[:, 2] += shared_ground_offset
    p1_root_court[:, 2] += shared_ground_offset

    floor_stats = {
        "p0_vertex_z_median_m": float(np.median(p0_court[:, :, 2].min(axis=1))),
        "p0_vertex_z_min_m": float(p0_court[:, :, 2].min()),
        "p0_vertex_z_max_m": float(p0_court[:, :, 2].max()),
        "p1_vertex_z_median_m": float(np.median(p1_court[:, :, 2].min(axis=1))),
        "p1_vertex_z_min_m": float(p1_court[:, :, 2].min()),
        "p1_vertex_z_max_m": float(p1_court[:, :, 2].max()),
        "shared_ground_offset_m": shared_ground_offset,
        "height_correction_applied": True,
        "height_correction": "one rigid vertical translation estimated from the median lowest mesh vertex across both players and all frames",
    }
    view = calibration["source_matched_view"]
    np.savez_compressed(
        args.output_dir / "two_player_smplx_court_animation.npz",
        p0_vertices_court=p0_court,
        p1_vertices_court=p1_court,
        faces=faces,
        p0_root_court=p0_root_court,
        p1_root_court=p1_root_court,
        p0_ground_measurements_court=tracks[0],
        p1_ground_measurements_court=tracks[1],
        p0_foot_static_confidence=p0_contact,
        p1_foot_static_confidence=p1_contact,
        court_width_m=np.float32(6.1),
        court_length_m=np.float32(13.4),
        net_height_m=np.float32(1.524),
        fps=np.float32(30.0),
        initial_view_lookat=np.asarray([0.0, 0.0, 0.75], dtype=np.float32),
        initial_view_front=np.asarray(view["camera_forward"], dtype=np.float32),
        initial_view_up=np.asarray(view["screen_up"], dtype=np.float32),
    )
    metadata = {
        "method": "Native GVHMR incam SMPL-X meshes transformed once through shared court PnP calibration with the planar Z-normal resolved to physical upward height",
        "uses_custom_3d_skeleton": False,
        "uses_per_frame_limb_fitting": False,
        "root_translation_fusion": {
            "description": "Ground contacts from the existing court tracker are admitted only when GVHMR predicts a static foot; all other frames, including airborne frames, propagate GVHMR's native root displacement.",
            "rigid_whole_body_xy_only": True,
            "court_track_xy_conversion": {
                "matrix": TRACK_XY_TO_PNP_COURT.tolist(),
                "reason": "the original stable ground tracker exports the across-net Y direction opposite to the court-corner PnP convention; this conversion aligns the two already-existing coordinate systems without changing any 2D detections",
            },
            "p0": p0_fusion,
            "p1": p1_fusion,
        },
        "source_native_results": {"p0": str(args.p0_result), "p1": str(args.p1_result)},
        "official_smplx_model": str(model_file),
        "frames": int(len(p0_court)),
        "vertices_per_player_per_frame": int(p0_court.shape[1]),
        "faces": int(len(faces)),
        "calibration": calibration,
        "floor_validation": floor_stats,
        "bundle": "two_player_smplx_court_animation.npz",
    }
    (args.output_dir / "scene_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-result", type=Path, default=P0_RESULT)
    parser.add_argument("--p1-result", type=Path, default=P1_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor-frame0", type=int, default=224)
    parser.add_argument("--source-start-frame0", type=int, default=120)
    parser.add_argument("--f-mm", type=int, default=24)
    parser.add_argument("--contact-threshold", type=float, default=0.60)
    parser.add_argument("--max-contact-residual-m", type=float, default=1.50)
    parser.add_argument("--replace-existing-bundle", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
