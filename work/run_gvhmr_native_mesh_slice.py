"""Run GVHMR's native post-processing and export an official SMPL-X mesh slice.

This runner deliberately does *not* rebuild 3D pose from RTMLib joints.  It
reuses the cached RTMLib-conditioned GVHMR inputs, executes the official
GVHMR ``postproc=True`` path, and applies its predicted SMPL-X parameters to
the licensed SMPL-X model.  It is a Windows-oriented validation tool and does
not modify GVHMR core code or the stable 2D BadmintonPose pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_module


PROJECT = Path(r"D:\Projects\BadmintonPose")
GVHMR_ROOT = PROJECT / "work" / "GVHMR"
SOURCE_VIDEO = PROJECT / "examples" / "data" / "test2.mp4"
SOURCE_CACHE_ROOT = PROJECT / "outputs" / "world_hmr_full_test2" / "gvhmr_rtmlib_p0" / "test2"
DEFAULT_OUTPUT = PROJECT / "outputs" / "world_hmr_native_mesh_test" / "test2_p0_f120_240"


def _load_tensor(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing cached GVHMR input: {path}")
    return torch.load(path, map_location="cpu")


def _slice_cached_data(cache_root: Path, start: int, end: int, f_mm: int) -> tuple[dict, int, int]:
    """Load and slice existing official-demo cache without copying a video."""
    sys.path.insert(0, str(GVHMR_ROOT))
    from hmr4d.utils.geo.hmr_cam import create_camera_sensor
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.video_io_utils import get_video_lwh

    length, width, height = get_video_lwh(SOURCE_VIDEO)
    if not (0 <= start < end <= length):
        raise ValueError(f"Invalid frame range [{start}, {end}) for {length}-frame video")

    preprocess = cache_root / "preprocess"
    bbx_xys = _load_tensor(preprocess / "bbx.pt")["bbx_xys"]
    kp2d = _load_tensor(preprocess / "vitpose.pt")
    f_imgseq = _load_tensor(preprocess / "vit_features.pt")
    slam = _load_tensor(preprocess / "slam_results.pt")

    expected = length
    for label, tensor in (("bbx_xys", bbx_xys), ("kp2d", kp2d), ("f_imgseq", f_imgseq), ("slam", slam)):
        if len(tensor) != expected:
            raise ValueError(f"Cached {label} has {len(tensor)} frames; expected {expected}")

    # This is the exact SimpleVO branch used by the official demo.  Camera
    # rotation is intentionally retained; this clip is not treated as a
    # synthetic static-camera sequence.
    r_w2c = torch.as_tensor(slam[:, :3, :3]).float()
    cam_angvel = compute_cam_angvel(r_w2c)
    k_fullimg = create_camera_sensor(width, height, f_mm)[2].repeat(end - start, 1, 1)

    data = {
        "length": torch.tensor(end - start),
        "bbx_xys": bbx_xys[start:end].float(),
        "kp2d": kp2d[start:end].float(),
        "K_fullimg": k_fullimg.float(),
        "cam_angvel": cam_angvel[start:end].float(),
        "f_imgseq": f_imgseq[start:end].float(),
    }
    return data, width, height


def _make_model():
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=["video_name=native_mesh_slice"])

    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    return model.eval().cuda(), cfg


def _to_cpu_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in params.items()}


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a single standard PLY frame for free inspection in Open3D/MeshLab."""
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        np.savetxt(handle, vertices, fmt="%.7f %.7f %.7f")
        np.savetxt(handle, faces, fmt="3 %d %d %d")


def _write_root_csv(path: Path, root: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("clip_frame", "root_x", "root_y", "root_z"))
        for index, xyz in enumerate(root):
            writer.writerow((index, float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _render_camera_overlay(
    path: Path,
    vertices_incam: torch.Tensor,
    faces: np.ndarray,
    k_fullimg: torch.Tensor,
    start: int,
    width: int,
    height: int,
) -> None:
    """Render native SMPL-X vertices over the original video; no synthetic skeleton."""
    from hmr4d.utils.vis.renderer import Renderer

    # SMPL-X has 20,908 faces, which can exceed PyTorch3D's automatic coarse
    # rasterizer bin at 1280x720.  The naive mode is slower but guarantees a
    # complete mesh image for this short visual validation.
    renderer = Renderer(width, height, device="cuda", faces=faces, K=k_fullimg[0], bin_size=0)
    capture = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {SOURCE_VIDEO}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create overlay video: {path}")

    try:
        for frame_index in range(len(vertices_incam)):
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Source video ended at local frame {frame_index}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb = renderer.render_mesh(vertices_incam[frame_index], frame_rgb, [55, 190, 255])
            writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        capture.release()


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GVHMR native mesh validation requires CUDA, but CUDA is unavailable")
    model_file = GVHMR_ROOT / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not model_file.exists():
        raise FileNotFoundError(f"Official SMPL-X model is missing: {model_file}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(GVHMR_ROOT)
    data, width, height = _slice_cached_data(args.cache_root, args.start_frame0, args.end_frame0, args.f_mm)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, cfg = _make_model()

    # DemoPL.predict is the official inference entry.  Unlike the deleted
    # validation runner, it always calls Pipeline.forward(..., postproc=True).
    with torch.no_grad():
        prediction = model.predict(data, static_cam=False)

    params_global = _to_cpu_params(prediction["smpl_params_global"])
    params_incam = _to_cpu_params(prediction["smpl_params_incam"])
    static_confidence = torch.sigmoid(prediction["net_outputs"]["static_conf_logits"][0]).detach().cpu()

    from hmr4d.utils.smplx_utils import make_smplx

    smplx_model = make_smplx("supermotion").eval().cuda()
    global_gpu = {key: value.cuda(non_blocking=True) for key, value in params_global.items()}
    incam_gpu = {key: value.cuda(non_blocking=True) for key, value in params_incam.items()}
    with torch.no_grad():
        mesh_global = smplx_model(**global_gpu)
        vertices_global = mesh_global.vertices.detach().cpu().float()
        smplx_joints_global = mesh_global.joints.detach().cpu().float()
        vertices_incam = smplx_model(**incam_gpu).vertices.detach().cpu().float()
        joints22_global = model.pipeline.endecoder.fk_v2(
            **{key: value[None] for key, value in global_gpu.items()}
        )[0].detach().cpu().float()
    faces = np.asarray(smplx_model.faces, dtype=np.int32)

    torch.save(
        {
            "smpl_params_global": params_global,
            "smpl_params_incam": params_incam,
            "K_fullimg": data["K_fullimg"].cpu(),
            "method": "GVHMR official DemoPL.predict with postproc=True; official SMPL-X v1.1 mesh",
            "official_postproc": True,
            "body_model": str(model_file),
        },
        args.output_dir / "1_hmr4d_results_native.pt",
    )
    np.savez_compressed(
        args.output_dir / "2_smplx_mesh_global.npz",
        vertices_global=vertices_global.numpy(),
        smplx_joints_global=smplx_joints_global.numpy(),
        joints22_global=joints22_global.numpy(),
        faces=faces,
        static_confidence=static_confidence.numpy(),
    )
    root = params_global["transl"].numpy()
    _write_root_csv(args.output_dir / "3_root_global_trajectory.csv", root)

    ply_frames = sorted({0, len(vertices_global) // 3, (2 * len(vertices_global)) // 3, len(vertices_global) - 1})
    ply_dir = args.output_dir / "mesh_frames_ply"
    ply_dir.mkdir()
    for frame_index in ply_frames:
        _write_ply(ply_dir / f"frame_{frame_index:04d}.ply", vertices_global[frame_index].numpy(), faces)

    if args.render_overlay:
        _render_camera_overlay(
            args.output_dir / "4_smplx_camera_overlay.mp4",
            vertices_incam.cuda(),
            faces,
            data["K_fullimg"],
            args.start_frame0,
            width,
            height,
        )

    root_steps = np.linalg.norm(np.diff(root, axis=0), axis=1)
    metadata = {
        "source_video": str(SOURCE_VIDEO),
        "source_cache": str(args.cache_root),
        "source_frame_range_zero_based_end_exclusive": [args.start_frame0, args.end_frame0],
        "frames": int(len(vertices_global)),
        "method": "GVHMR native post-processing + licensed SMPL-X v1.1 neutral mesh",
        "official_postproc": True,
        "camera_motion": "cached official SimpleVO",
        "uses_custom_3d_skeleton": False,
        "smplx_vertices_per_frame": int(vertices_global.shape[1]),
        "smplx_faces": int(len(faces)),
        "root_step_median_m": float(np.median(root_steps)),
        "root_step_max_m": float(np.max(root_steps)),
        "root_speed_mps_median": float(np.median(root_steps) * 30.0),
        "root_speed_mps_max": float(np.max(root_steps) * 30.0),
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_total_mb": int(torch.cuda.get_device_properties("cuda").total_memory / 1024**2),
        "gpu_peak_allocated_mb": int(torch.cuda.max_memory_allocated() / 1024**2),
        "gvhmr_checkpoint": str(GVHMR_ROOT / cfg.ckpt_path),
        "body_model": str(model_file),
        "outputs": {
            "native_parameters": "1_hmr4d_results_native.pt",
            "mesh_sequence": "2_smplx_mesh_global.npz",
            "root_trajectory": "3_root_global_trajectory.csv",
            "camera_overlay": "4_smplx_camera_overlay.mp4" if args.render_overlay else None,
            "free_view_samples": "mesh_frames_ply/",
        },
    }
    (args.output_dir / "experiment_record.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def rerender_existing_overlay(args: argparse.Namespace) -> None:
    """Replace only an overlay rendered with an unsuitable rasterizer setting."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    result_path = args.output_dir / "1_hmr4d_results_native.pt"
    if not result_path.exists():
        raise FileNotFoundError(f"Native result is missing: {result_path}")
    os.chdir(GVHMR_ROOT)
    data, width, height = _slice_cached_data(args.cache_root, args.start_frame0, args.end_frame0, args.f_mm)
    result = torch.load(result_path, map_location="cpu")
    from hmr4d.utils.smplx_utils import make_smplx

    smplx_model = make_smplx("supermotion").eval().cuda()
    params_incam = {key: value.cuda(non_blocking=True) for key, value in result["smpl_params_incam"].items()}
    with torch.no_grad():
        vertices_incam = smplx_model(**params_incam).vertices.detach()
    _render_camera_overlay(
        args.output_dir / "4_smplx_camera_overlay.mp4",
        vertices_incam,
        np.asarray(smplx_model.faces, dtype=np.int32),
        data["K_fullimg"],
        args.start_frame0,
        width,
        height,
    )
    record_path = args.output_dir / "experiment_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["camera_overlay_renderer"] = "PyTorch3D Renderer(bin_size=0)"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Re-rendered native SMPL-X overlay: {args.output_dir / '4_smplx_camera_overlay.mp4'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=SOURCE_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame0", type=int, default=120)
    parser.add_argument("--end-frame0", type=int, default=240)
    parser.add_argument("--f-mm", type=int, default=24)
    parser.add_argument("--render-overlay", action="store_true")
    parser.add_argument("--rerender-overlay-only", action="store_true")
    args = parser.parse_args()
    if args.rerender_overlay_only:
        rerender_existing_overlay(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
