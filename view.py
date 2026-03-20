import argparse
import importlib
import json
import shutil
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from utils.renderer_utils import get_renderer_components


def _resolve_run_dir(result_dir: str) -> Path:
    run_dir = Path(result_dir)
    if run_dir.exists():
        return run_dir.resolve()

    candidates = sorted(Path("results").glob(f"*/{result_dir}"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find run directory '{result_dir}'. "
            "Pass a full path or a timestamp under results/*/."
        )
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Found multiple runs matching '{result_dir}': "
            + ", ".join(str(p) for p in candidates)
            + ". Please pass an unambiguous path."
        )
    return candidates[0].resolve()


def _load_yaml(path: Path):
    try:
        import yaml
    except ImportError:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_trajectory_points(trj_path: Path):
    if not trj_path.exists():
        return None
    with open(trj_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = data.get("trj_est", [])
    if not poses:
        return None
    pts = []
    for pose in poses:
        pose_np = np.asarray(pose, dtype=np.float32)
        if pose_np.shape != (4, 4):
            continue
        pts.append(pose_np[:3, 3])
    if not pts:
        return None
    return np.stack(pts, axis=0)


def _load_trajectory_poses(trj_path: Path):
    if not trj_path.exists():
        return []
    with open(trj_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = []
    for pose in data.get("trj_est", []):
        pose_np = np.asarray(pose, dtype=np.float32)
        if pose_np.shape == (4, 4):
            poses.append(pose_np)
    return poses


def _build_monitor_cameras_json(poses, cfg, fallback_count: int = 32):
    calib = cfg.get("Dataset", {}).get("Calibration", {}) if isinstance(cfg, dict) else {}
    width = int(calib.get("width", 640))
    height = int(calib.get("height", 480))
    fx = float(calib.get("fx", 525.0))
    fy = float(calib.get("fy", 525.0))

    cameras = []
    if poses:
        for idx, pose in enumerate(poses):
            rot = pose[:3, :3].tolist()
            pos = pose[:3, 3].tolist()
            cameras.append(
                {
                    "id": idx,
                    "img_name": f"frame_{idx:06d}.png",
                    "width": width,
                    "height": height,
                    "fx": fx,
                    "fy": fy,
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "rotation": [
                        [float(rot[0][0]), float(rot[0][1]), float(rot[0][2])],
                        [float(rot[1][0]), float(rot[1][1]), float(rot[1][2])],
                        [float(rot[2][0]), float(rot[2][1]), float(rot[2][2])],
                    ],
                }
            )
    else:
        for idx in range(fallback_count):
            cameras.append(
                {
                    "id": idx,
                    "img_name": f"frame_{idx:06d}.png",
                    "width": width,
                    "height": height,
                    "fx": fx,
                    "fy": fy,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                }
            )
    return cameras


def _prepare_monitor_scene_dir(run_dir: Path, ply_path: Path, cfg, trj_path: Path) -> Path:
    scene_dir = run_dir / "monitor_scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    input_ply = scene_dir / "input.ply"
    if not input_ply.exists() or input_ply.stat().st_size != ply_path.stat().st_size:
        shutil.copy2(ply_path, input_ply)

    poses = _load_trajectory_poses(trj_path)
    cameras = _build_monitor_cameras_json(poses, cfg)
    cameras_json = scene_dir / "cameras.json"
    with open(cameras_json, "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)

    return scene_dir


def _trajectory_topdown_image(points: Optional[np.ndarray], size: int = 800) -> np.ndarray:
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        cv2 = None

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if points is None or len(points) < 2:
        if cv2 is not None:
            cv2.putText(
                canvas,
                "No trajectory",
                (20, size // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
        return canvas

    xy = points[:, [0, 1]]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    margin = 0.08
    usable = (1.0 - 2.0 * margin) * size
    scale = usable / np.max(span)
    centered = (xy - min_xy) * scale

    px = (centered[:, 0] + margin * size).astype(np.int32)
    py = (size - 1 - (centered[:, 1] + margin * size)).astype(np.int32)
    if cv2 is not None:
        poly = np.stack([px, py], axis=1).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (px[0], py[0]), 6, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px[-1], py[-1]), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Trajectory (top-down XY)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _pick_render_mode(render_mode, render_items):
    if isinstance(render_mode, int) and 0 <= render_mode < len(render_items):
        return render_items[render_mode]
    if isinstance(render_mode, str) and render_mode in render_items:
        return render_mode
    return "RGB"


def _render_item_image(render_pkg, mode: str, trajectory_img: np.ndarray):
    import torch

    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        cv2 = None

    if mode == "Depth":
        depth_key = "depth" if "depth" in render_pkg else "mean_depth"
        depth = render_pkg[depth_key][0].detach().float().cpu().numpy()
        max_depth = max(float(depth.max()), 1e-6)
        depth_u8 = np.clip(depth / max_depth * 255.0, 0, 255).astype(np.uint8)
        if cv2 is None:
            return np.repeat(depth_u8[:, :, None], 3, axis=2)
        depth_img = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)

    if mode == "Normal":
        normal = render_pkg.get("rend_normal")
        if normal is None:
            normal = render_pkg.get("surf_normal")
        if normal is None:
            return np.zeros_like(trajectory_img)
        normal_img = normal.detach().float().permute(1, 2, 0).cpu().numpy()
        normal_img = np.clip((normal_img + 1.0) * 0.5, 0.0, 1.0)
        return (normal_img * 255.0).astype(np.uint8)

    if mode == "Opacity":
        opacity = render_pkg["opacity"][0].detach().float().cpu().numpy()
        max_opacity = max(float(opacity.max()), 1e-6)
        opacity_u8 = np.clip(opacity / max_opacity * 255.0, 0, 255).astype(np.uint8)
        if cv2 is None:
            return np.repeat(opacity_u8[:, :, None], 3, axis=2)
        opacity_img = cv2.applyColorMap(opacity_u8, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(opacity_img, cv2.COLOR_BGR2RGB)

    if mode == "Trajectory":
        return trajectory_img

    rgb = (
        (torch.clamp(render_pkg["render"], min=0.0, max=1.0) * 255.0)
        .byte()
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )
    return rgb


def _fallback_image(width: int, height: int, mode: str, trajectory_img: np.ndarray) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if mode == "Trajectory" and trajectory_img is not None:
        try:
            cv2 = importlib.import_module("cv2")
            image = cv2.resize(trajectory_img, (width, height), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            y = np.linspace(0, trajectory_img.shape[0] - 1, height).astype(np.int32)
            x = np.linspace(0, trajectory_img.shape[1] - 1, width).astype(np.int32)
            image = trajectory_img[y][:, x]
    return np.ascontiguousarray(image)


def _launch_viewer(viewer_cmd: str):
    if not viewer_cmd:
        return None
    return subprocess.Popen(shlex.split(viewer_cmd))


def _init_renderer(ply_path: Path, white_background: bool, renderer_mode: str):
    import torch

    render, gaussian_model_cls = get_renderer_components(renderer_mode)
    gaussians = gaussian_model_cls(sh_degree=3)
    gaussians.load_ply(str(ply_path))
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False)
    bg_color = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return render, gaussians, pipe, background


class MiniCamCompat:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        world_view_transform,
        full_proj_transform,
    ):
        import torch

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.cam_rot_delta = torch.zeros(3, device="cuda")
        self.cam_trans_delta = torch.zeros(3, device="cuda")
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.projection_matrix = torch.bmm(
            self.world_view_transform.unsqueeze(0).inverse(),
            self.full_proj_transform.unsqueeze(0),
        ).squeeze(0)


class MonitorProtocolServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind((host, port))
        self.listener.listen()
        self.listener.settimeout(0)
        self.conn = None

    @staticmethod
    def _send_json_data(conn, data):
        payload = json.dumps(data).encode("utf-8")
        conn.sendall(len(payload).to_bytes(4, "little"))
        conn.sendall(payload)

    def try_connect(self, render_items):
        if self.conn is not None:
            return
        try:
            self.conn, _addr = self.listener.accept()
            self.conn.settimeout(None)
            self._send_json_data(self.conn, render_items)
        except Exception:
            return

    def receive(self):
        import torch

        if self.conn is None:
            return None, None, None, None, None

        def _recv_exact(n):
            buf = b""
            conn = self.conn
            if conn is None:
                raise ConnectionError("Socket not connected")
            while len(buf) < n:
                chunk = conn.recv(n - len(buf))
                if not chunk:
                    raise ConnectionError("Socket closed")
                buf += chunk
            return buf

        raw_len = _recv_exact(4)
        msg_len = int.from_bytes(raw_len, "little")
        message = json.loads(_recv_exact(msg_len).decode("utf-8"))

        width = message["resolution_x"]
        height = message["resolution_y"]
        if width == 0 or height == 0:
            return None, None, None, None, None

        fovy = message["fov_y"]
        fovx = message["fov_x"]
        scaling_modifier = message["scaling_modifier"]
        view = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
        view[:, 1] = -view[:, 1]
        view[:, 2] = -view[:, 2]
        view_proj = torch.reshape(
            torch.tensor(message["view_projection_matrix"]), (4, 4)
        ).cuda()
        view_proj[:, 1] = -view_proj[:, 1]
        view_proj[:, 2] = -view_proj[:, 2]
        render_mode = message.get("render_mode", "RGB")

        cam = MiniCamCompat(width, height, fovy, fovx, view, view_proj)
        do_training = bool(message.get("train", False))
        keep_alive = bool(message.get("keep_alive", True))
        return cam, do_training, keep_alive, scaling_modifier, render_mode

    def send(self, message_bytes, verify, metrics):
        if self.conn is None:
            return
        if message_bytes is not None:
            self.conn.sendall(message_bytes)
        self.conn.sendall(len(verify).to_bytes(4, "little"))
        self.conn.sendall(verify.encode("ascii"))
        self._send_json_data(self.conn, metrics)

    def close_connection(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def close(self):
        self.close_connection()
        try:
            self.listener.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="View MonoGS output with Gaussian-Splatting-Monitor protocol"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help=(
            "Run directory, e.g. results/tum_rgbd_dataset_freiburg1_desk/"
            "2026-03-18-18-36-41"
        ),
    )
    parser.add_argument(
        "--monitor-root",
        type=str,
        default=str((Path(__file__).resolve().parent.parent / "Gaussian-Splatting-Monitor")),
        help="Path to Gaussian-Splatting-Monitor repository",
    )
    parser.add_argument(
        "--ply",
        type=str,
        default="",
        help="Optional explicit point cloud path (defaults to point_cloud/final/point_cloud.ply)",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="",
        help="Optional explicit trajectory JSON path (defaults to plot/trj_final.json)",
    )
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument(
        "--render-items",
        type=str,
        default="RGB,Depth,Normal,Opacity,Trajectory",
        help="Comma-separated render item list exposed to Monitor viewer",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=["2dgs", "3dgs"],
        default="2dgs",
        help="Renderer mode used for visualization",
    )
    parser.add_argument(
        "--viewer-cmd",
        type=str,
        default="",
        help="Optional command to auto-start SIBR viewer",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="Exit if no successful viewing activity within this many seconds (<=0 disables)",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.result_dir)
    ply_path = Path(args.ply) if args.ply else run_dir / "point_cloud" / "final" / "point_cloud.ply"
    trj_path = Path(args.trajectory) if args.trajectory else run_dir / "plot" / "trj_final.json"

    if not ply_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")

    config_path = run_dir / "config.yml"
    cfg = {}
    white_background = False
    if config_path.exists():
        cfg = _load_yaml(config_path)
        white_background = bool(cfg.get("model_params", {}).get("white_background", False))

    trajectory_points = _load_trajectory_points(trj_path)
    trajectory_img = _trajectory_topdown_image(trajectory_points)
    monitor_scene_dir = _prepare_monitor_scene_dir(run_dir, ply_path, cfg, trj_path)

    monitor_root = Path(args.monitor_root).resolve()
    if not monitor_root.exists():
        raise FileNotFoundError(f"Gaussian-Splatting-Monitor not found: {monitor_root}")
    monitor_server = MonitorProtocolServer(args.ip, args.port)

    render_items = [x.strip() for x in args.render_items.split(",") if x.strip()]
    if not render_items:
        render_items = ["RGB", "Depth", "Normal", "Opacity", "Trajectory"]

    render_func = None
    gaussians = None
    pipe = None
    background = None
    renderer_error = ""

    viewer_proc = _launch_viewer(args.viewer_cmd)
    print(f"Run directory: {run_dir}")
    print(f"Point cloud: {ply_path}")
    if trj_path.exists():
        print(f"Trajectory: {trj_path}")
    else:
        print("Trajectory: not found")
    print(f"Monitor scene dir: {monitor_scene_dir}")
    print(f"Listening for Monitor viewer on {args.ip}:{args.port}")

    start = time.time()
    had_connection = False
    had_render = False

    try:
        while True:
            if args.timeout_sec > 0 and (time.time() - start) > args.timeout_sec:
                status = "after connection" if had_connection else "without any viewer connection"
                print(f"Timed out after {args.timeout_sec:.1f}s {status}.")
                return

            if monitor_server.conn is None:
                monitor_server.try_connect(render_items)
                if monitor_server.conn is not None:
                    had_connection = True

            while monitor_server.conn is not None:
                if args.timeout_sec > 0 and (time.time() - start) > args.timeout_sec:
                    print(f"Timed out after {args.timeout_sec:.1f}s while serving viewer.")
                    return
                try:
                    (
                        custom_cam,
                        _do_training,
                        _keep_alive,
                        scaling_modifier,
                        render_mode,
                    ) = monitor_server.receive()

                    net_image_bytes = None
                    if custom_cam is not None:
                        if render_func is None and not renderer_error:
                            try:
                                render_func, gaussians, pipe, background = _init_renderer(
                                    ply_path, white_background, args.renderer
                                )
                            except Exception as e:
                                renderer_error = str(e)

                        if render_func is not None:
                            import torch

                            assert gaussians is not None
                            assert pipe is not None
                            assert background is not None
                            custom_cam.cam_rot_delta = torch.zeros(3, device="cuda")
                            custom_cam.cam_trans_delta = torch.zeros(3, device="cuda")
                            render_pkg = render_func(
                                custom_cam,
                                gaussians,
                                pipe,
                                background,
                                scaling_modifier=float(
                                    1.0 if scaling_modifier is None else scaling_modifier
                                ),
                            )
                            mode = _pick_render_mode(render_mode, render_items)
                            image = _render_item_image(render_pkg, mode, trajectory_img)
                            if image.ndim == 2:
                                image = np.repeat(image[:, :, None], 3, axis=2)
                            net_image_bytes = memoryview(np.ascontiguousarray(image))
                            had_render = True
                        else:
                            mode = _pick_render_mode(render_mode, render_items)
                            fallback = _fallback_image(
                                int(custom_cam.image_width),
                                int(custom_cam.image_height),
                                mode,
                                trajectory_img,
                            )
                            net_image_bytes = memoryview(fallback)

                    metrics_dict = {
                        "#": int(0 if gaussians is None else gaussians.get_opacity.shape[0]),
                        "trajectory_points": int(0 if trajectory_points is None else trajectory_points.shape[0]),
                        "rendered": int(had_render),
                        "renderer_ready": int(render_func is not None),
                        "renderer_error": float(1.0 if renderer_error else 0.0),
                    }
                    monitor_server.send(net_image_bytes, str(monitor_scene_dir), metrics_dict)
                except Exception:
                    monitor_server.close_connection()
                    break

            if viewer_proc is not None and viewer_proc.poll() is not None:
                if had_connection:
                    print("Viewer process exited.")
                    return

            time.sleep(0.01)
    finally:
        monitor_server.close()
        if viewer_proc is not None and viewer_proc.poll() is None:
            viewer_proc.terminate()


if __name__ == "__main__":
    main()
