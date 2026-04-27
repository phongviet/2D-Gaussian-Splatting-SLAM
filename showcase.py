#!/usr/bin/env python3
"""
2dgslam Showcase — Dual-Pane Side-by-Side Gaussian Splatting Viewer

Renders two Gaussian Splatting scenes side-by-side for A/B comparison.
Serves rendered frames to the SIBR remote viewer (SIBR_remoteGaussian_app).
Auto-detects 2DGS vs 3DGS from the PLY file.

Usage:
    python showcase.py [--ip 127.0.0.1] [--port 6009]
    python showcase.py --left results/sceneA --right results/sceneB

Then connect with the SIBR viewer:
    LD_LIBRARY_PATH="..." SIBR_remoteGaussian_app

Camera controls in SIBR move both scenes. Switching view modes
(RGBA / Depth / Normal / Alpha) in SIBR applies to both sides.
"""

import os
import sys
import math
import yaml
import json
import time
import torch
import struct
import socket
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, ttk
from argparse import ArgumentParser
from munch import munchify

# ---------------------------------------------------------------------------
# Path setup — make submodule rasterizers importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_SCRIPT_DIR, "submodules", "diff-surfel-rasterization"))
sys.path.append(os.path.join(_SCRIPT_DIR, "submodules", "diff-gaussian-rasterization"))

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh
from plyfile import PlyData


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers  (inlined from view.py for self-containment)
# ═══════════════════════════════════════════════════════════════════════════

def depths_to_points(view, depthmap):
    """Unproject a depth map to 3D world-space points."""
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, W / 2],
        [0, H / 2, 0, H / 2],
        [0, 0, 0, 1],
    ]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T
    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device="cuda").float(),
        torch.arange(H, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    return depthmap.reshape(-1, 1) * rays_d + rays_o


def depth_to_normal(view, depth):
    """Compute surface normals from a depth map via finite differences."""
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    elif depth.dim() == 4:
        depth = depth.squeeze(0)
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    if points.dim() == 4:
        points = points.squeeze(0)
    output = torch.zeros_like(points)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


# ═══════════════════════════════════════════════════════════════════════════
# MiniCam — lightweight camera for SIBR protocol
# ═══════════════════════════════════════════════════════════════════════════

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar,
                 world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform

        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3, :3]
        self.projection_matrix = view_inv @ self.full_proj_transform
        self.cam_rot_delta = torch.zeros(3, device="cuda")
        self.cam_trans_delta = torch.zeros(3, device="cuda")


# ═══════════════════════════════════════════════════════════════════════════
# NetworkGUI — SIBR remote-viewer TCP protocol
# ═══════════════════════════════════════════════════════════════════════════

class NetworkGUI:
    def __init__(self):
        self.conn = None
        self.addr = None
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.host = "127.0.0.1"
        self.port = 6009

    def init(self, host, port):
        self.host = host
        self.port = port
        self.listener.bind((self.host, self.port))
        self.listener.listen()
        self.listener.settimeout(0)

    def send_json_data(self, conn, data):
        if conn is None:
            return
        try:
            raw = json.dumps(data).encode("utf-8")
            conn.sendall(struct.pack("I", len(raw)))
            conn.sendall(raw)
        except Exception:
            self.conn = None

    def try_connect(self, render_items):
        try:
            self.conn, self.addr = self.listener.accept()
            self.conn.settimeout(None)
            self.send_json_data(self.conn, render_items)
        except Exception:
            pass

    def read(self):
        msg_len = self.conn.recv(4)
        msg_len = int.from_bytes(msg_len, "little")
        msg = self.conn.recv(msg_len)
        return json.loads(msg.decode("utf-8"))

    def send(self, image_bytes, verify, metrics):
        if self.conn is None:
            return
        try:
            if image_bytes is not None:
                self.conn.sendall(image_bytes)
            self.conn.sendall(len(verify).to_bytes(4, "little"))
            self.conn.sendall(bytes(verify, "ascii"))
            self.send_json_data(self.conn, metrics)
        except Exception:
            self.conn = None

    def receive(self):
        try:
            message = self.read()
        except Exception:
            self.conn = None
            return None, None, None, None, None

        width = message["resolution_x"]
        height = message["resolution_y"]
        if width == 0 or height == 0:
            return None, None, None, None, None
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(
                torch.tensor(message["view_matrix"]), (4, 4)
            ).cuda()
            world_view_transform[:, 1] = -world_view_transform[:, 1]
            world_view_transform[:, 2] = -world_view_transform[:, 2]
            full_proj_transform = torch.reshape(
                torch.tensor(message["view_projection_matrix"]), (4, 4)
            ).cuda()
            full_proj_transform[:, 1] = -full_proj_transform[:, 1]
            custom_cam = MiniCam(
                width, height, fovy, fovx, znear, zfar,
                world_view_transform, full_proj_transform,
            )
            render_mode = message["render_mode"]
        except Exception:
            traceback.print_exc()
            return None, None, None, None, None
        return custom_cam, do_training, keep_alive, scaling_modifier, render_mode


# ═══════════════════════════════════════════════════════════════════════════
# Render-function factory  (2DGS / 3DGS)
# ═══════════════════════════════════════════════════════════════════════════

def get_render_func(mode="2dgs"):
    """Return the appropriate rasteriser callable for *mode*."""
    if mode == "2dgs":
        from gaussian_splatting.gaussian_renderer import render
        return render

    # ---- 3DGS fallback ----
    def render_3dgs(viewpoint_camera, pc, pipe, bg_color,
                    scaling_modifier=1.0, override_color=None):
        try:
            from diff_gaussian_rasterization import (
                GaussianRasterizationSettings,
                GaussianRasterizer,
            )
        except ImportError:
            print("Error: diff-gaussian-rasterization (3DGS) not found.")
            return {}

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
            tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            projmatrix_raw=viewpoint_camera.projection_matrix,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda"
        )
        try:
            means2D.retain_grad()
        except Exception:
            pass

        rendered_image, radii, depth, opacity_map, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=pc.get_features,
            colors_precomp=override_color,
            opacities=pc.get_opacity,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
            cov3D_precomp=None,
        )
        depth_map = depth
        normal = depth_to_normal(viewpoint_camera, depth_map).permute(2, 0, 1)
        alpha = (depth_map > 0).float()
        return {
            "render": rendered_image,
            "depth": depth_map,
            "rend_median_depth": depth_map,
            "rend_alpha": alpha,
            "rend_normal": normal,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    return render_3dgs


# ═══════════════════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════════════════

RENDER_ITEMS = ["RGBA", "Depth", "Normal", "Alpha"]


def render_net_image(render_pkg, render_items, render_mode, camera):
    """Select the appropriate channel from a render package."""
    if render_mode >= len(render_items):
        render_mode = 0
    output = render_items[render_mode].lower()

    if output == "alpha":
        net_image = render_pkg.get("rend_alpha", render_pkg.get("opacity", None))
    elif output == "normal":
        net_image = render_pkg.get("rend_normal", None)
        if net_image is not None:
            net_image = (net_image + 1) / 2
    elif output == "depth":
        net_image = render_pkg.get(
            "rend_median_depth", render_pkg.get("depth", None)
        )
    else:
        net_image = render_pkg["render"]

    if net_image is not None and net_image.shape[0] == 1:
        net_image = (net_image - net_image.min()) / (
            net_image.max() - net_image.min() + 1e-5
        )
        net_image = net_image.repeat(3, 1, 1)

    return net_image


# ═══════════════════════════════════════════════════════════════════════════
# Auto-detection & Scene loading
# ═══════════════════════════════════════════════════════════════════════════

def detect_gs_mode(ply_path):
    """Detect 2DGS vs 3DGS from PLY scale dimensions.

    * 2 scale channels  →  2DGS (surfels)
    * 3+ scale channels →  3DGS (ellipsoids)
    """
    plydata = PlyData.read(ply_path)
    scale_names = [
        p.name for p in plydata.elements[0].properties
        if p.name.startswith("scale_")
    ]
    return "3dgs" if len(scale_names) >= 3 else "2dgs"


def find_ply_path(model_dir):
    """Locate point_cloud.ply inside a result directory."""
    candidate = os.path.join(model_dir, "point_cloud", "final", "point_cloud.ply")
    if os.path.exists(candidate):
        return candidate
    pc_dir = os.path.join(model_dir, "point_cloud")
    if os.path.isdir(pc_dir):
        iters = [d for d in os.listdir(pc_dir) if d.startswith("iteration_")]
        if iters:
            latest = sorted(iters, key=lambda x: int(x.split("_")[-1]))[-1]
            path = os.path.join(pc_dir, latest, "point_cloud.ply")
            if os.path.exists(path):
                return path
    return None


class SceneState:
    """Holds loaded state for one side of the viewer."""

    def __init__(self):
        self.gaussians = None
        self.render_func = None
        self.pipe = None
        self.background = None
        self.mode = None          # "2dgs" | "3dgs"
        self.model_dir = None
        self.num_gaussians = 0
        self.loaded = False
        self.loading = False
        self.error = None


def ensure_monitor_scene(model_dir, gaussians=None):
    """Ensure a SIBR-compatible monitor_scene directory exists.

    Creates cameras.json (from trajectory if available) and input.ply
    so that SIBR can identify the dataset type and load the scene.
    """
    monitor_dir = os.path.join(model_dir, "monitor_scene")
    os.makedirs(monitor_dir, exist_ok=True)

    # cfg_args in the root (mandatory for SIBR)
    cfg_args_path = os.path.join(model_dir, "cfg_args")
    if not os.path.exists(cfg_args_path):
        with open(cfg_args_path, "w") as f:
            f.write("--source_path dummy --model_path dummy")

    # cameras.json
    cameras_json_path = os.path.join(monitor_dir, "cameras.json")
    trj_path = os.path.join(model_dir, "plot", "trj_final.json")
    config_path = os.path.join(model_dir, "config.yml")

    if not os.path.exists(cameras_json_path):
        if os.path.exists(trj_path):
            width, height, fx, fy = 640, 480, 525.0, 525.0
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                calib = cfg.get("Dataset", {}).get("Calibration", {})
                width = calib.get("width", width)
                height = calib.get("height", height)
                fx = calib.get("fx", fx)
                fy = calib.get("fy", fy)
            with open(trj_path, "r") as f:
                trj_data = json.load(f)
            cameras = []
            for i, pose_mat in enumerate(trj_data.get("trj_est", [])):
                pose = torch.tensor(pose_mat).float()
                cameras.append({
                    "id": i,
                    "img_name": f"frame_{i:06d}.png",
                    "width": width, "height": height,
                    "fx": fx, "fy": fy,
                    "position": pose[:3, 3].tolist(),
                    "rotation": pose[:3, :3].tolist(),
                })
            with open(cameras_json_path, "w") as f:
                json.dump(cameras, f, indent=2)
        else:
            # Write a minimal empty cameras list so SIBR can parse it
            with open(cameras_json_path, "w") as f:
                json.dump([], f)

    # input.ply
    input_ply_path = os.path.join(monitor_dir, "input.ply")
    if not os.path.exists(input_ply_path):
        if gaussians is not None:
            gaussians.save_ply(input_ply_path)
        else:
            # Write a minimal valid PLY so SIBR doesn't crash
            _write_minimal_ply(input_ply_path)

    return monitor_dir


def _create_dummy_scene(base_dir):
    """Create a minimal valid scene structure for SIBR at startup.

    This prevents SIBR from crashing before any real scene is loaded.
    """
    dummy_dir = os.path.join(base_dir, ".showcase_dummy")
    monitor_dir = os.path.join(dummy_dir, "monitor_scene")
    os.makedirs(monitor_dir, exist_ok=True)

    # cfg_args
    cfg_path = os.path.join(dummy_dir, "cfg_args")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as f:
            f.write("--source_path dummy --model_path dummy")

    # cameras.json
    cam_path = os.path.join(monitor_dir, "cameras.json")
    if not os.path.exists(cam_path):
        with open(cam_path, "w") as f:
            json.dump([{
                "id": 0, "img_name": "dummy.png",
                "width": 640, "height": 480,
                "fx": 525.0, "fy": 525.0,
                "position": [0.0, 0.0, 0.0],
                "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }], f)

    # input.ply
    ply_path = os.path.join(monitor_dir, "input.ply")
    if not os.path.exists(ply_path):
        _write_minimal_ply(ply_path)

    return monitor_dir


def _write_minimal_ply(path):
    """Write a minimal valid PLY with one dummy point."""
    import numpy as np
    from plyfile import PlyElement
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.array([(0, 0, 0, 0, 0, 1, 128, 128, 128)], dtype=dtype)
    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(path)


def load_scene(model_dir):
    """Load a Gaussian Splatting scene and return a populated SceneState."""
    state = SceneState()
    state.model_dir = os.path.abspath(model_dir)

    # Find PLY
    ply_path = find_ply_path(model_dir)
    if ply_path is None:
        raise FileNotFoundError(f"No point_cloud.ply found in {model_dir}")

    # Detect mode
    state.mode = detect_gs_mode(ply_path)
    print(f"  Detected: {state.mode.upper()}")

    # Load config
    config_path = os.path.join(model_dir, "config.yml")
    config = None
    sh_degree = 0
    white_background = False

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        sh_degree = config.get("model_params", {}).get("sh_degree", 0)
        white_background = config.get("model_params", {}).get(
            "white_background", False
        )
    else:
        # Infer sh_degree from PLY f_rest count
        plydata = PlyData.read(ply_path)
        n_rest = len([
            p.name for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ])
        if n_rest > 0:
            sh_degree = int(round(math.sqrt(n_rest / 3 + 1) - 1))
        print(f"  No config.yml — inferred sh_degree={sh_degree}")

    # Load Gaussians
    state.gaussians = GaussianModel(sh_degree, config=config)
    print(f"  Loading PLY: {ply_path}")
    state.gaussians.load_ply(ply_path)
    state.num_gaussians = state.gaussians.get_xyz.shape[0]
    print(f"  Loaded {state.num_gaussians:,} Gaussians")

    # Ensure SIBR-compatible monitor_scene exists
    ensure_monitor_scene(model_dir, state.gaussians)

    # Background
    bg = [1, 1, 1] if white_background else [0, 0, 0]
    state.background = torch.tensor(bg, dtype=torch.float32, device="cuda")

    # Pipeline params
    state.pipe = munchify({
        "compute_cov3D_python": False,
        "convert_SHs_python": False,
        "depth_ratio": 0.0,
    })

    # Renderer
    state.render_func = get_render_func(state.mode)
    state.loaded = True
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Showcase Application
# ═══════════════════════════════════════════════════════════════════════════

class ShowcaseApp:
    """Dual-pane viewer: tkinter control panel + SIBR TCP server."""

    def __init__(self, ip="127.0.0.1", port=6009,
                 left_dir=None, right_dir=None):
        self.ip = ip
        self.port = port
        self.left = SceneState()
        self.right = SceneState()
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.network_gui = NetworkGUI()

        # Create a dummy scene dir so SIBR doesn't crash before scenes load
        self.verify_path = _create_dummy_scene(_SCRIPT_DIR)

        # Build tkinter GUI
        self.root = tk.Tk()
        self.root.title("2dgslam Showcase — Control Panel")
        self.root.geometry("560x500")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_gui()

        # Start SIBR server thread
        self._server_thread = threading.Thread(
            target=self._server_loop, daemon=True
        )
        self._server_thread.start()

        # Periodic status refresh
        self._update_status()

        # Pre-load scenes from CLI if provided
        if left_dir:
            self.root.after(100, lambda: self._start_load(left_dir, "left"))
        if right_dir:
            self.root.after(200, lambda: self._start_load(right_dir, "right"))

        # Block in the tkinter main-loop
        self.root.mainloop()

    # ── GUI construction ─────────────────────────────────────────────

    def _build_gui(self):
        style = ttk.Style()
        style.theme_use("clam")

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(
            main,
            text="2dgslam Showcase Viewer",
            font=("Helvetica", 15, "bold"),
        ).pack(pady=(0, 8))

        # Connection status
        conn_frame = ttk.LabelFrame(main, text="Server", padding=6)
        conn_frame.pack(fill=tk.X, pady=4)
        self.status_label = ttk.Label(
            conn_frame,
            text=f"Waiting for SIBR on {self.ip}:{self.port}…",
        )
        self.status_label.pack(anchor=tk.W)

        # ── Left scene ──
        lf = ttk.LabelFrame(main, text="Left Scene  (A)", padding=6)
        lf.pack(fill=tk.X, pady=4)

        btn_row_l = ttk.Frame(lf)
        btn_row_l.pack(fill=tk.X)
        ttk.Button(
            btn_row_l,
            text="Select Folder…",
            command=lambda: self._select_folder("left"),
        ).pack(side=tk.LEFT)
        self.left_info = ttk.Label(btn_row_l, text="", foreground="gray")
        self.left_info.pack(side=tk.RIGHT, padx=6)

        self.left_path = ttk.Label(lf, text="(not loaded)", foreground="gray")
        self.left_path.pack(anchor=tk.W, pady=(4, 0))

        # ── Right scene ──
        rf = ttk.LabelFrame(main, text="Right Scene  (B)", padding=6)
        rf.pack(fill=tk.X, pady=4)

        btn_row_r = ttk.Frame(rf)
        btn_row_r.pack(fill=tk.X)
        ttk.Button(
            btn_row_r,
            text="Select Folder…",
            command=lambda: self._select_folder("right"),
        ).pack(side=tk.LEFT)
        self.right_info = ttk.Label(btn_row_r, text="", foreground="gray")
        self.right_info.pack(side=tk.RIGHT, padx=6)

        self.right_path = ttk.Label(rf, text="(not loaded)", foreground="gray")
        self.right_path.pack(anchor=tk.W, pady=(4, 0))

        # ── Render info ──
        info_frame = ttk.LabelFrame(main, text="Rendering", padding=6)
        info_frame.pack(fill=tk.X, pady=4)
        self.render_mode_label = ttk.Label(
            info_frame, text="View mode: (waiting for SIBR)"
        )
        self.render_mode_label.pack(anchor=tk.W)

        # ── Instructions ──
        inst_frame = ttk.LabelFrame(main, text="How to Use", padding=6)
        inst_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        ttk.Label(
            inst_frame,
            text=(
                "1. Select result folders for Left and Right scenes.\n"
                "2. Launch the SIBR viewer:\n"
                "     SIBR_remoteGaussian_app\n"
                "3. Camera controls in SIBR move both scenes.\n"
                "4. Switch view modes in SIBR\n"
                "   (RGBA / Depth / Normal / Alpha)\n"
                "   → applies to both sides simultaneously.\n"
                "5. Left half = Scene A, Right half = Scene B."
            ),
            justify=tk.LEFT,
            wraplength=500,
        ).pack(anchor=tk.W)

    # ── Folder selection ─────────────────────────────────────────────

    def _select_folder(self, side):
        init_dir = os.path.join(_SCRIPT_DIR, "results")
        if not os.path.isdir(init_dir):
            init_dir = _SCRIPT_DIR
        folder = filedialog.askdirectory(
            title=f"Select {side.capitalize()} Scene Folder",
            initialdir=init_dir,
        )
        if folder:
            self._start_load(folder, side)

    def _start_load(self, folder, side):
        """Mark UI as loading and kick off background load."""
        path_lbl = self.left_path if side == "left" else self.right_path
        info_lbl = self.left_info if side == "left" else self.right_info
        path_lbl.config(text=os.path.basename(folder), foreground="orange")
        info_lbl.config(text="Loading…", foreground="orange")
        threading.Thread(
            target=self._load_thread, args=(folder, side), daemon=True
        ).start()

    def _load_thread(self, folder, side):
        try:
            print(f"\n{'='*60}")
            print(f"Loading {side.upper()} scene: {folder}")
            print(f"{'='*60}")
            new_state = load_scene(folder)
            with self.lock:
                if side == "left":
                    self.left = new_state
                else:
                    self.right = new_state
                # Update verify_path to the real monitor_scene so SIBR
                # can load proper scene metadata (cameras, point cloud).
                monitor = os.path.join(new_state.model_dir, "monitor_scene")
                if os.path.isdir(monitor):
                    self.verify_path = monitor
            self.root.after(0, self._update_scene_ui, side, new_state, None)
        except Exception as e:
            print(f"Error loading {side} scene: {e}")
            traceback.print_exc()
            self.root.after(0, self._update_scene_ui, side, None, str(e))

    def _update_scene_ui(self, side, state, error):
        path_lbl = self.left_path if side == "left" else self.right_path
        info_lbl = self.left_info if side == "left" else self.right_info

        if error:
            path_lbl.config(text="Error!", foreground="red")
            info_lbl.config(text=error[:50], foreground="red")
        elif state and state.loaded:
            # Show last two path components for context
            parts = state.model_dir.rstrip("/").split("/")
            short = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            path_lbl.config(text=short, foreground="#2e7d32")
            info_lbl.config(
                text=f"{state.mode.upper()}  |  {state.num_gaussians:,} GS",
                foreground="#1565c0",
            )

    # ── Status polling ───────────────────────────────────────────────

    def _update_status(self):
        if not self.running:
            return
        if self.connected:
            self.status_label.config(
                text="✓  SIBR viewer connected", foreground="#2e7d32"
            )
        else:
            self.status_label.config(
                text=f"Waiting for SIBR on {self.ip}:{self.port}…",
                foreground="gray",
            )
        self.root.after(500, self._update_status)

    def _on_close(self):
        self.running = False
        self.root.destroy()

    # ── SIBR server loop (background thread) ─────────────────────────

    def _server_loop(self):
        self.network_gui.init(self.ip, self.port)
        print(f"Showcase server listening on {self.ip}:{self.port}")

        while self.running:
            with torch.no_grad():
                # Wait for a connection
                if self.network_gui.conn is None:
                    self.connected = False
                    self.network_gui.try_connect(RENDER_ITEMS)
                    if self.network_gui.conn is None:
                        time.sleep(0.01)
                        continue
                    self.connected = True
                    print("SIBR viewer connected!")

                # Serve frames while connected
                while self.network_gui.conn is not None and self.running:
                    try:
                        net_image_bytes = None
                        (
                            custom_cam,
                            do_training,
                            keep_alive,
                            scaling_modifier,
                            render_mode,
                        ) = self.network_gui.receive()

                        if self.network_gui.conn is None:
                            self.connected = False
                            break

                        if custom_cam is not None:
                            # Snapshot scene references under lock
                            with self.lock:
                                left = self.left
                                right = self.right

                            composite = self._render_composite(
                                left, right,
                                custom_cam, render_mode, scaling_modifier,
                            )

                            if composite is not None:
                                net_image_bytes = memoryview(
                                    (torch.clamp(composite, 0, 1.0) * 255)
                                    .byte()
                                    .permute(1, 2, 0)
                                    .contiguous()
                                    .cpu()
                                    .numpy()
                                )

                            metrics = {}
                            if left.loaded:
                                metrics["L"] = (
                                    f"{left.mode.upper()} "
                                    f"{left.num_gaussians:,}"
                                )
                            if right.loaded:
                                metrics["R"] = (
                                    f"{right.mode.upper()} "
                                    f"{right.num_gaussians:,}"
                                )

                            # Update view-mode label on tk thread
                            mode_name = (
                                RENDER_ITEMS[render_mode]
                                if render_mode < len(RENDER_ITEMS)
                                else "RGBA"
                            )
                            self.root.after(
                                0,
                                lambda m=mode_name: self.render_mode_label
                                    .config(text=f"View mode: {m}"),
                            )
                        else:
                            metrics = {}

                        self.network_gui.send(
                            net_image_bytes, self.verify_path, metrics
                        )

                        if keep_alive is False:
                            self.network_gui.conn.close()
                            self.network_gui.conn = None
                            self.connected = False

                    except Exception as e:
                        print(f"Connection error: {e}")
                        self.network_gui.conn = None
                        self.connected = False

    # ── Compositing ──────────────────────────────────────────────────

    def _render_composite(self, left, right, cam, render_mode,
                          scaling_modifier):
        """Render both scenes and composite into a split-screen image.

        Layout: left half shows Scene A, right half shows Scene B.
        A 2-pixel white divider marks the boundary.
        If only one scene is loaded, it fills the whole frame.
        """
        W = cam.image_width
        H = cam.image_height
        mid = W // 2

        left_loaded = left.loaded
        right_loaded = right.loaded

        if not left_loaded and not right_loaded:
            return torch.zeros(3, H, W, device="cuda")

        # ── render each scene ──
        left_img = None
        if left_loaded:
            try:
                pkg = left.render_func(
                    cam, left.gaussians, left.pipe,
                    left.background, scaling_modifier,
                )
                if pkg is not None:
                    left_img = render_net_image(
                        pkg, RENDER_ITEMS, render_mode, cam
                    )
            except Exception as e:
                print(f"Left render error: {e}")

        right_img = None
        if right_loaded:
            try:
                pkg = right.render_func(
                    cam, right.gaussians, right.pipe,
                    right.background, scaling_modifier,
                )
                if pkg is not None:
                    right_img = render_net_image(
                        pkg, RENDER_ITEMS, render_mode, cam
                    )
            except Exception as e:
                print(f"Right render error: {e}")

        # ── composite ──
        if left_img is not None and right_img is not None:
            composite = torch.zeros(3, H, W, device="cuda")
            composite[:, :, :mid] = left_img[:, :, :mid]
            composite[:, :, mid:] = right_img[:, :, mid:]
            # 2-pixel bright divider
            div_lo = max(0, mid - 1)
            div_hi = min(W, mid + 1)
            composite[:, :, div_lo:div_hi] = 1.0
            return composite
        elif left_img is not None:
            return left_img
        elif right_img is not None:
            return right_img
        else:
            return torch.zeros(3, H, W, device="cuda")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = ArgumentParser(description="2dgslam Showcase — Dual-Pane Viewer")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument(
        "--left", type=str, default=None,
        help="Pre-load left scene from this result directory",
    )
    parser.add_argument(
        "--right", type=str, default=None,
        help="Pre-load right scene from this result directory",
    )
    args = parser.parse_args()

    ShowcaseApp(
        ip=args.ip,
        port=args.port,
        left_dir=args.left,
        right_dir=args.right,
    )
