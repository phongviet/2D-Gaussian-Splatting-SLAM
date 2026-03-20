import os
import sys
from argparse import ArgumentParser
import json
import socket
import struct
import traceback
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import yaml


def _add_monitor_to_path():
    monogs_root = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(monogs_root)
    monitor_root = os.path.join(workspace_root, "Gaussian-Splatting-Monitor")
    if monitor_root not in sys.path:
        sys.path.insert(0, monitor_root)
    return monitor_root


MONITOR_ROOT = _add_monitor_to_path()


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.projection_matrix = torch.bmm(
            self.world_view_transform.unsqueeze(0).inverse(),
            self.full_proj_transform.unsqueeze(0),
        ).squeeze(0)
        self.cam_rot_delta = torch.zeros(3, device="cuda", dtype=torch.float32)
        self.cam_trans_delta = torch.zeros(3, device="cuda", dtype=torch.float32)


class NetworkGUI:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 6009
        self.conn = None
        self.addr = None
        self.listener = None

    @staticmethod
    def _create_listener():
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return listener

    def init(self, wish_host, wish_port):
        self.host = wish_host
        self.port = wish_port
        if self.listener is not None:
            try:
                self.listener.close()
            except Exception:
                pass
        self.listener = self._create_listener()
        self.listener.bind((self.host, self.port))
        self.listener.listen()
        self.listener.settimeout(0)

    @staticmethod
    def send_json_data(conn, data):
        serialized_data = json.dumps(data)
        bytes_data = serialized_data.encode("utf-8")
        conn.sendall(struct.pack("I", len(bytes_data)))
        conn.sendall(bytes_data)

    def try_connect(self, render_items):
        if self.listener is None:
            return
        try:
            self.conn, self.addr = self.listener.accept()
            print(f"\nConnected by {self.addr}")
            self.conn.settimeout(None)
            self.send_json_data(self.conn, render_items)
        except Exception:
            pass

    def read(self):
        if self.conn is None:
            raise RuntimeError("No active monitor connection")
        message_length = self.conn.recv(4)
        if not message_length:
            raise ConnectionError("Monitor disconnected")
        message_length = int.from_bytes(message_length, "little")
        message = self.conn.recv(message_length)
        if not message:
            raise ConnectionError("Monitor disconnected")
        return json.loads(message.decode("utf-8"))

    def send(self, message_bytes, verify, metrics):
        if self.conn is None:
            raise RuntimeError("No active monitor connection")
        if message_bytes != None:
            self.conn.sendall(message_bytes)
        self.conn.sendall(len(verify).to_bytes(4, "little"))
        self.conn.sendall(bytes(verify, "ascii"))
        self.send_json_data(self.conn, metrics)

    def disconnect(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = None
        self.addr = None

    def receive(self):
        message = self.read()
        width = message["resolution_x"]
        height = message["resolution_y"]

        if width != 0 and height != 0:
            custom_cam = None
            do_training = False
            keep_alive = False
            scaling_modifier = 1.0
            render_mode = 0
            try:
                do_training = bool(message["train"])
                fovy = message["fov_y"]
                fovx = message["fov_x"]
                znear = message["z_near"]
                zfar = message["z_far"]
                keep_alive = bool(message["keep_alive"])
                scaling_modifier = message["scaling_modifier"]
                world_view_transform = torch.reshape(
                    torch.tensor(message["view_matrix"], dtype=torch.float32), (4, 4)
                ).cuda()
                world_view_transform[:, 1] = -world_view_transform[:, 1]
                world_view_transform[:, 2] = -world_view_transform[:, 2]
                full_proj_transform = torch.reshape(
                    torch.tensor(
                        message["view_projection_matrix"], dtype=torch.float32
                    ),
                    (4, 4),
                ).cuda()
                full_proj_transform[:, 1] = -full_proj_transform[:, 1]
                custom_cam = MiniCam(
                    width,
                    height,
                    fovy,
                    fovx,
                    znear,
                    zfar,
                    world_view_transform,
                    full_proj_transform,
                )
                render_mode = message["render_mode"]
            except Exception:
                print("")
                traceback.print_exc()
            return custom_cam, do_training, keep_alive, scaling_modifier, render_mode
        else:
            return None, None, None, None, None


network_gui = NetworkGUI()


def gradient_map(image):
    sobel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
        / 4
    )
    sobel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda()
        / 4
    )

    grad_x = torch.cat(
        [F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])]
    )
    grad_y = torch.cat(
        [F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])]
    )
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude


def unproject_depth_map(depth_map, camera):
    depth_map = depth_map.squeeze()
    height, width = depth_map.shape
    x = torch.linspace(0, width - 1, width).cuda()
    y = torch.linspace(0, height - 1, height).cuda()
    Y, X = torch.meshgrid(y, x, indexing="ij")

    depth_flat = depth_map.reshape(-1)
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)

    X_norm = (X_flat / (width - 1)) * 2 - 1
    Y_norm = (Y_flat / (height - 1)) * 2 - 1

    points_camera = torch.stack([X_norm, Y_norm, depth_flat], dim=-1)

    K_matrix = camera.projection_matrix
    f1 = K_matrix[2, 2]
    f2 = K_matrix[3, 2]

    sdepth = (f1 * points_camera[..., 2:3] + f2) / (points_camera[..., 2:3] + 1e-8)

    points_camera = torch.cat((points_camera[..., 0:2], sdepth), dim=-1)
    points_camera = points_camera.view((height, width, 3))
    points_camera = torch.cat(
        [points_camera, torch.ones_like(points_camera[:, :, :1])], dim=-1
    )
    points_world = torch.matmul(points_camera, camera.full_proj_transform.inverse())

    points_world = points_world[:, :, :3] / points_world[:, :, 3:]
    points_world = points_world.view((height, width, 3))

    return points_world


def depth_to_normal(depth_map, camera):
    depth_map = depth_map.squeeze()
    height, width = depth_map.shape
    points_world = torch.zeros((height + 1, width + 1, 3)).to(depth_map.device)
    points_world[:height, :width, :] = unproject_depth_map(depth_map, camera)

    p1 = points_world[:-1, :-1, :]
    p2 = points_world[1:, :-1, :]
    p3 = points_world[:-1, 1:, :]

    v1 = p2 - p1
    v2 = p3 - p1

    normals = torch.cross(v1, v2, dim=-1)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

    return normals


def colormap(map, cmap="turbo"):
    if hasattr(plt, "colormaps"):
        cmap_values = plt.colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
    else:
        cmap_values = plt.cm.get_cmap(cmap)(np.linspace(0, 1, 256))[:, :3]
    colors = torch.tensor(cmap_values, device=map.device, dtype=torch.float32)
    map = (map - map.min()) / (map.max() - map.min())
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2, 0, 1)
    return map


def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == "alpha":
        net_image = render_pkg["alpha"]
        net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min())
    elif output == "depth":
        net_image = render_pkg["mean_depth"]
        net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min())
    elif output == "normal":
        net_image = depth_to_normal(render_pkg["mean_depth"], camera).permute(2, 0, 1)
        net_image = (net_image + 1) / 2
    elif output == "edge":
        net_image = gradient_map(render_pkg["render"])
    elif output == "curvature":
        net_image = gradient_map(
            depth_to_normal(render_pkg["mean_depth"], camera).permute(2, 0, 1)
        )
    else:
        net_image = render_pkg["render"]

    if net_image.shape[0] == 1:
        net_image = colormap(net_image)
    return net_image


def _resolve_config(result_path):
    candidates = [
        os.path.join(result_path, "config.yml"),
        os.path.join(result_path, "config.yaml"),
        os.path.join(os.path.dirname(result_path), "config.yml"),
        os.path.join(os.path.dirname(result_path), "config.yaml"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            with open(c, "r", encoding="utf-8") as f:
                return yaml.safe_load(f), c
    return {}, None


def _resolve_ply_path(result_path, iteration):
    if os.path.isfile(result_path) and result_path.endswith(".ply"):
        return result_path

    if iteration == "final":
        candidates = [
            os.path.join(result_path, "point_cloud", "final", "point_cloud.ply"),
            os.path.join(result_path, "final", "point_cloud.ply"),
            os.path.join(result_path, "point_cloud.ply"),
        ]
    else:
        iter_name = f"iteration_{iteration}"
        candidates = [
            os.path.join(result_path, "point_cloud", iter_name, "point_cloud.ply"),
            os.path.join(result_path, iter_name, "point_cloud.ply"),
        ]

    for c in candidates:
        if os.path.isfile(c):
            return c

    raise FileNotFoundError(
        f"Could not find point cloud PLY in '{result_path}' for iteration '{iteration}'."
    )


def _resolve_monitor_verify_path(result_path):
    monitor_scene = os.path.join(result_path, "monitor_scene")
    if os.path.isdir(monitor_scene):
        return monitor_scene
    return result_path


def _load_components(renderer_mode):
    if renderer_mode == "3dgs":
        from gaussian_splatting.gaussian_renderer.render_3d import render
        from gaussian_splatting.scene.gaussian_model_3d import GaussianModel
    else:
        from gaussian_splatting.gaussian_renderer.render_2d import render
        from gaussian_splatting.scene.gaussian_model import GaussianModel

    return render, GaussianModel


def _wrap_render_for_monitor(render_fn):
    def _render_with_monitor_compat(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0
    ):
        render_pkg = render_fn(
            viewpoint_camera, pc, pipe, bg_color, scaling_modifier
        )
        if "mean_depth" not in render_pkg and "depth" in render_pkg:
            render_pkg["mean_depth"] = render_pkg["depth"]
        if "alpha" not in render_pkg and "opacity" in render_pkg:
            render_pkg["alpha"] = render_pkg["opacity"]
        return render_pkg

    return _render_with_monitor_compat


def view(dataset, pipe, gaussians, render_fn):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    while True:
        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render_fn(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0]
                        # Add more metrics as needed
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                except Exception as e:
                    network_gui.disconnect()
                    break


if __name__ == "__main__":
    parser = ArgumentParser(description="MonoGS viewer for Gaussian-Splatting-Monitor")
    parser.add_argument("result_path", type=str, help="MonoGS result directory or point_cloud.ply path")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--iteration", type=str, default="final")
    parser.add_argument("--renderer", type=str, choices=["auto", "2dgs", "3dgs"], default="auto")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--sh_degree", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])

    result_path = os.path.abspath(args.result_path)
    config, config_path = _resolve_config(result_path)
    renderer_mode = args.renderer
    if renderer_mode == "auto":
        renderer_mode = str(config.get("Training", {}).get("renderer", "2dgs")).lower()
        if renderer_mode not in {"2dgs", "3dgs"}:
            renderer_mode = "2dgs"

    render_fn, GaussianModel = _load_components(renderer_mode)
    render_fn = _wrap_render_for_monitor(render_fn)

    sh_degree = args.sh_degree
    if sh_degree is None:
        sh_degree = int(config.get("model_params", {}).get("sh_degree", 0))

    gaussians = GaussianModel(sh_degree)
    ply_path = _resolve_ply_path(result_path, args.iteration)
    gaussians.load_ply(ply_path)

    render_items = ["RGB", "Alpha", "Depth", "Normal", "Curvature", "Edge"]
    white_background = args.white_background
    if not white_background:
        white_background = bool(config.get("model_params", {}).get("white_background", False))
    dataset = SimpleNamespace(
        white_background=white_background,
        render_items=render_items,
        source_path=_resolve_monitor_verify_path(result_path),
    )
    pipe = SimpleNamespace(
        convert_SHs_python=bool(config.get("pipeline_params", {}).get("convert_SHs_python", False)),
        compute_cov3D_python=bool(config.get("pipeline_params", {}).get("compute_cov3D_python", False)),
    )

    print("View:", result_path)
    print("Config:", config_path if config_path else "<none>")
    print("Renderer:", renderer_mode)
    print("PLY:", ply_path)
    network_gui.init(args.ip, args.port)

    view(dataset, pipe, gaussians, render_fn)

    print("\nViewing complete.")
