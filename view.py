import os
import sys
import torch
import math
import yaml
import json
import traceback
import socket
import struct
from argparse import ArgumentParser
from munch import munchify

# Add submodules to path for easy importing if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "submodules/diff-surfel-rasterization")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "submodules/diff-gaussian-rasterization")))

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh

# --- Inlined Geometry Utilities (Self-Contained) ---
def depths_to_points(view, depthmap):
    """
    Unproject a depth map to 3D points in world space.
    Args:
        view: camera with world_view_transform and full_proj_transform
        depthmap: [1, H, W] depth
    Returns:
        points: [H*W, 3] world-space 3D points
    """
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, W / 2],
        [0, H / 2, 0, H / 2],
        [0, 0, 0, 1]
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
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    """
    Compute a surface normal map from a depth map using finite differences.
    Args:
        view: camera
        depth: [1, H, W] or [H, W] depth map
    Returns:
        normal_map: [H, W, 3] world-space unit normals
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    elif depth.dim() == 4:
        depth = depth.squeeze(0)
        
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    if points.dim() == 4: # Handle [1, H, W, 3]
        points = points.squeeze(0)
        
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

# --- MiniCam Class for SIBR Viewer ---
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
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
        
        # Derive projection_matrix for 2dgslam renderer: VP = V @ P => P = V^-1 @ VP
        self.projection_matrix = view_inv @ self.full_proj_transform
        
        self.cam_rot_delta = torch.zeros(3, device="cuda")
        self.cam_trans_delta = torch.zeros(3, device="cuda")

# --- Network GUI (SIBR Protocol) ---
class NetworkGUI:
    def __init__(self):
        self.conn = None
        self.addr = None
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.host = "127.0.0.1"
        self.port = 6009

    def init(self, wish_host, wish_port):
        self.host = wish_host
        self.port = wish_port
        self.listener.bind((self.host, self.port))
        self.listener.listen()
        self.listener.settimeout(0)

    def send_json_data(self, conn, data):
        if conn is None:
            return
        try:
            serialized_data = json.dumps(data)
            bytes_data = serialized_data.encode('utf-8')
            conn.sendall(struct.pack('I', len(bytes_data)))
            conn.sendall(bytes_data)
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
        messageLength = self.conn.recv(4)
        messageLength = int.from_bytes(messageLength, 'little')
        message = self.conn.recv(messageLength)
        return json.loads(message.decode("utf-8"))

    def send(self, message_bytes, verify, metrics):
        if self.conn is None:
            return
        try:
            if message_bytes is not None:
                self.conn.sendall(message_bytes)
            self.conn.sendall(len(verify).to_bytes(4, 'little'))
            self.conn.sendall(bytes(verify, 'ascii'))
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

        if width != 0 and height != 0:
            try:
                do_training = bool(message["train"])
                fovy = message["fov_y"]
                fovx = message["fov_x"]
                znear = message["z_near"]
                zfar = message["z_far"]
                keep_alive = bool(message["keep_alive"])
                scaling_modifier = message["scaling_modifier"]
                world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
                world_view_transform[:,1] = -world_view_transform[:,1]
                world_view_transform[:,2] = -world_view_transform[:,2]
                full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
                full_proj_transform[:,1] = -full_proj_transform[:,1]
                custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
                render_mode = message["render_mode"]
            except Exception:
                traceback.print_exc()
                return None, None, None, None, None
            return custom_cam, do_training, keep_alive, scaling_modifier, render_mode
        else:
            return None, None, None, None, None

network_gui = NetworkGUI()

# --- Image Processing ---
def render_net_image(render_pkg, render_items, render_mode, camera):
    if render_mode >= len(render_items):
        render_mode = 0
    output = render_items[render_mode].lower()
    
    if output == 'alpha':
        net_image = render_pkg.get("rend_alpha", render_pkg.get("opacity", None))
    elif output == 'normal':
        net_image = render_pkg.get("rend_normal", None)
        if net_image is not None:
            net_image = (net_image + 1) / 2
    elif output == 'depth':
        # Use median depth if available, otherwise expected depth
        net_image = render_pkg.get("rend_median_depth", render_pkg.get("depth", None))
    else:
        net_image = render_pkg["render"]

    if net_image is not None and net_image.shape[0] == 1:
        # Simple colormapping for 1-channel outputs
        net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min() + 1e-5)
        net_image = net_image.repeat(3, 1, 1)

    return net_image

# --- Rendering Logic ---
def get_render_func(mode="2dgs"):
    if mode == "2dgs":
        from gaussian_splatting.gaussian_renderer import render
        return render
    else:
        # 3DGS mode
        def render_3dgs(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
            # 3DGS-specific rendering flow
            try:
                from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
            except ImportError:
                print("Error: diff-gaussian-rasterization (3DGS) not found.")
                # Fallback to empty if needed, but usually we expect it to be there
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
                debug=False
            )
            
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            means3D = pc.get_xyz
            means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
            try:
                means2D.retain_grad()
            except: pass
            
            opacity = pc.get_opacity
            scales = pc.get_scaling
            rotations = pc.get_rotation
            shs = pc.get_features
            colors_precomp = override_color
            
            rendered_image, radii, depth, opacity_map, n_touched = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=None
            )
            
            # Compute normals from depth for 3DGS (using inlined logic)
            depth_map = depth # Standard 3DGS depth is [1, H, W]
            normal = depth_to_normal(viewpoint_camera, depth_map).permute(2, 0, 1)
            alpha = (depth_map > 0).float()
            
            return {
                "render": rendered_image,
                "depth": depth_map,
                "rend_median_depth": depth_map,
                "rend_alpha": alpha,
                "rend_normal": normal,
                "visibility_filter": radii > 0,
                "radii": radii
            }
        
        return render_3dgs

def ensure_monitor_scene(model_dir, gaussians=None):
    """
    Standardizes the result directory so SIBR can recognize and load it.
    Mimics the 'monitor_scene' structure found in '2DGS-Baseline'.
    """
    monitor_dir = os.path.join(model_dir, "monitor_scene")
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)
        print(f"Created monitor_scene directory: {monitor_dir}")

    # 1. cfg_args in the root (mandatory for SIBR)
    cfg_args_path = os.path.join(model_dir, "cfg_args")
    if not os.path.exists(cfg_args_path):
        with open(cfg_args_path, "w") as f:
            f.write("--source_path dummy --model_path dummy")
        print(f"Created dummy cfg_args")

    # 2. cameras.json inside monitor_scene (mandatory for SIBR)
    cameras_json_path = os.path.join(monitor_dir, "cameras.json")
    trj_path = os.path.join(model_dir, "plot/trj_final.json")
    config_path = os.path.join(model_dir, "config.yml")
    
    if not os.path.exists(cameras_json_path) and os.path.exists(trj_path):
        print(f"Generating cameras.json from trajectory...")
        
        # Get calibration from config
        width, height, fx, fy = 640, 480, 525.0, 525.0
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                calib = config.get("Dataset", {}).get("Calibration", {})
                width = calib.get("width", width)
                height = calib.get("height", height)
                fx = calib.get("fx", fx)
                fy = calib.get("fy", fy)

        with open(trj_path, "r") as f:
            trj_data = json.load(f)
        
        cameras = []
        for i, pose_mat in enumerate(trj_data.get("trj_est", [])):
            # In 2dgslam Baseline, 'rotation' is the first 3x3 of C2W matrix
            # and 'position' is the 4th column of C2W matrix.
            pose = torch.tensor(pose_mat).float()
            rot = pose[:3, :3].tolist()
            pos = pose[:3, 3].tolist()
            
            cameras.append({
                "id": i,
                "img_name": f"frame_{i:06d}.png",
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "position": pos,
                "rotation": rot
            })
        
        with open(cameras_json_path, "w") as f:
            json.dump(cameras, f, indent=2)
        print(f"Generated cameras.json with {len(cameras)} frames")

    # 3. input.ply inside monitor_scene
    input_ply_path = os.path.join(monitor_dir, "input.ply")
    if not os.path.exists(input_ply_path) and gaussians is not None:
        print(f"Exporting current Gaussians to input.ply for SIBR...")
        gaussians.save_ply(input_ply_path)

def view(model_dir, mode, ip, port):
    # Load config to get sh_degree
    config_path = os.path.join(model_dir, "config.yml")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    sh_degree = config.get("model_params", {}).get("sh_degree", 0)
    white_background = config.get("model_params", {}).get("white_background", False)
    
    # Initialize model
    gaussians = GaussianModel(sh_degree, config=config)
    ply_path = os.path.join(model_dir, "point_cloud/final/point_cloud.ply")
    if not os.path.exists(ply_path):
        # Try finding the latest iteration if 'final' doesn't exist
        point_cloud_dir = os.path.join(model_dir, "point_cloud")
        iters = [d for d in os.listdir(point_cloud_dir) if d.startswith("iteration_")]
        if iters:
            latest_iter = sorted(iters, key=lambda x: int(x.split("_")[-1]))[-1]
            ply_path = os.path.join(point_cloud_dir, latest_iter, "point_cloud.ply")
        else:
            print(f"Error: Could not find point cloud in {point_cloud_dir}")
            return
            
    print(f"Loading model from {ply_path}...")
    gaussians.load_ply(ply_path)
    
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Pipeline params dummy
    pipe = munchify({
        "compute_cov3D_python": False,
        "convert_SHs_python": False,
        "depth_ratio": 0.0 # for 2dgs
    })
    
    render_func = get_render_func(mode)
    render_items = ['RGBA', 'Depth', 'Normal'] # standard items
    
    network_gui.init(ip, port)
    print(f"Listening on {ip}:{port}...", flush=True)

    monitor_dir = os.path.join(model_dir, "monitor_scene")
    
    # Ensure SIBR-compatible metadata exists
    ensure_monitor_scene(model_dir, gaussians)
    
    path_to_send = monitor_dir if os.path.exists(monitor_dir) else model_dir

    while True:
        with torch.no_grad():
            if network_gui.conn is None:
                network_gui.try_connect(render_items)
            
            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifier, render_mode = network_gui.receive()
                    
                    if network_gui.conn is None:
                        break

                    if custom_cam is not None:
                        render_pkg = render_func(custom_cam, gaussians, pipe, background, scaling_modifier)
                        net_image = render_net_image(render_pkg, render_items, render_mode, custom_cam)
                        
                        # Diagnostic print
                        visible_count = (render_pkg["radii"] > 0).sum().item()
                        print(f"Render: {render_mode}, Visible: {visible_count}, Cam: {custom_cam.camera_center.tolist()}", flush=True)
                        
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        
                        metrics_dict = {
                            "#": gaussians.get_xyz.shape[0],
                            "Visible": visible_count
                        }
                    else:
                        metrics_dict = {
                            "#": gaussians.get_xyz.shape[0]
                        }

                    network_gui.send(net_image_bytes, path_to_send, metrics_dict)
                    
                    if keep_alive is False:
                        network_gui.conn.close()
                        network_gui.conn = None
                        
                except Exception as e:
                    print(f"Viewer connection lost: {e}")
                    # traceback.print_exc()
                    network_gui.conn = None

if __name__ == "__main__":
    parser = ArgumentParser(description="2dgslam Results Viewer")
    parser.add_argument("--mode", type=str, choices=["2dgs", "3dgs"], default="2dgs")
    parser.add_argument("--dir", type=str, required=True, help="Path to the result directory (containing config.yml and point_cloud/)")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    
    args = parser.parse_args()
    
    view(os.path.abspath(args.dir), args.mode, args.ip, args.port)
