"""Interactive free-view player for a native GVHMR two-player court bundle."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d


PROJECT = Path(r"D:\Projects\BadmintonPose")
DEFAULT_BUNDLE = (
    PROJECT
    / "outputs"
    / "world_hmr_native_mesh_full"
    / "two_player_court_full"
    / "two_player_smplx_court_animation.npz"
)


def _court_lines(width: float, length: float, net_height: float) -> o3d.geometry.LineSet:
    half_w, half_l = width / 2.0, length / 2.0
    vertices = np.array(
        [
            [-half_w, -half_l, 0], [half_w, -half_l, 0], [half_w, half_l, 0], [-half_w, half_l, 0],
            [-half_w, 0, 0], [half_w, 0, 0],
            [-half_w, 0, net_height], [half_w, 0, net_height],
        ],
        dtype=np.float64,
    )
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [4, 6], [5, 7], [6, 7]], dtype=np.int32)
    court = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines),
    )
    court.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.92, 0.92, 0.92]]), (len(lines), 1)))
    return court


def _make_mesh(vertices: np.ndarray, faces: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def view(bundle_path: Path) -> None:
    data = np.load(bundle_path)
    p0, p1, faces = data["p0_vertices_court"], data["p1_vertices_court"], data["faces"]
    if len(p0) != len(p1):
        raise ValueError("Animation bundle has a player-frame mismatch")
    fps = float(data["fps"])
    mesh0 = _make_mesh(p0[0], faces, (0.12, 0.52, 0.96))
    mesh1 = _make_mesh(p1[0], faces, (1.0, 0.42, 0.12))
    court = _court_lines(float(data["court_width_m"]), float(data["court_length_m"]), float(data["net_height_m"]))

    state = {"frame": 0, "playing": False, "last": time.monotonic()}
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window("BadmintonPose — native GVHMR SMPL-X", width=1280, height=800)
    visualizer.add_geometry(court)
    visualizer.add_geometry(mesh0)
    visualizer.add_geometry(mesh1)
    opts = visualizer.get_render_option()
    opts.background_color = np.asarray([0.055, 0.065, 0.085])
    opts.mesh_show_back_face = True
    opts.line_width = 2.0
    control = visualizer.get_view_control()
    if all(key in data for key in ("initial_view_lookat", "initial_view_front", "initial_view_up")):
        # Start from the physical camera side recovered with the court, so the
        # opening view matches the source video rather than Open3D's arbitrary
        # default orbit direction.  The user can still freely orbit afterward.
        control.set_lookat(data["initial_view_lookat"].astype(np.float64))
        control.set_front(data["initial_view_front"].astype(np.float64))
        control.set_up(data["initial_view_up"].astype(np.float64))
        control.set_zoom(0.58)

    def apply_frame(index: int) -> None:
        state["frame"] = index % len(p0)
        mesh0.vertices = o3d.utility.Vector3dVector(p0[state["frame"]].astype(np.float64))
        mesh1.vertices = o3d.utility.Vector3dVector(p1[state["frame"]].astype(np.float64))
        mesh0.compute_vertex_normals()
        mesh1.compute_vertex_normals()
        visualizer.update_geometry(mesh0)
        visualizer.update_geometry(mesh1)
        visualizer.update_renderer()

    def previous(_):
        state["playing"] = False
        apply_frame(state["frame"] - 1)
        return False

    def next_frame(_):
        state["playing"] = False
        apply_frame(state["frame"] + 1)
        return False

    def toggle(_):
        state["playing"] = not state["playing"]
        state["last"] = time.monotonic()
        return False

    def first(_):
        state["playing"] = False
        apply_frame(0)
        return False

    def animate(_):
        if state["playing"] and time.monotonic() - state["last"] >= 1.0 / fps:
            apply_frame(state["frame"] + 1)
            state["last"] = time.monotonic()
        return False

    visualizer.register_key_callback(ord("A"), previous)
    visualizer.register_key_callback(ord("D"), next_frame)
    visualizer.register_key_callback(ord(" "), toggle)
    visualizer.register_key_callback(ord("R"), first)
    visualizer.register_animation_callback(animate)
    print("Controls: drag = orbit, wheel = zoom, A/D = previous/next, Space = play/pause, R = first frame")
    visualizer.run()
    visualizer.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    if not args.bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {args.bundle}")
    view(args.bundle)


if __name__ == "__main__":
    main()
