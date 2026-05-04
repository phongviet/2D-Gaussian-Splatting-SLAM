import os
import sys
import torch
import numpy as np
import cv2
from munch import munchify
import yaml
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--res", type=str, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from utils.dataset import load_dataset
    from utils.camera_utils import Camera
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    from scripts.align_utils import umeyama_alignment, apply_sim3_to_gaussians

    def depth_to_normal(view, depth):
        # From view.py
        def depths_to_points(view, depthmap):
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

        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        elif depth.dim() == 4:
            depth = depth.squeeze(0)
            
        points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
        if points.dim() == 4:
            points = points.squeeze(0)
            
        output = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        output[1:-1, 1:-1, :] = normal_map
        return output

    config_path = os.path.join(args.res, "config.yml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = "cuda"
    
    class ArgsMock: pass
    dargs = ArgsMock()
    dargs.config = config_path
    dataset = load_dataset(dargs, config_path, config)
    
    gaussians = GaussianModel(config["model_params"]["sh_degree"])
    gaussians.load_ply(os.path.join(args.res, "point_cloud/final/point_cloud.ply"))
    
    trj_path = os.path.join(args.res, "plot/trj_final.json")
    with open(trj_path, "r") as f:
        trj_data = json.load(f)
    
    est_poses = np.array(trj_data["trj_est"])
    gt_poses = np.array(trj_data["trj_gt"])
    est_centers = est_poses[:, :3, 3]
    gt_centers = gt_poses[:, :3, 3]
    s, R, t = umeyama_alignment(est_centers, gt_centers, with_scale=True)
    apply_sim3_to_gaussians(gaussians, s, R, t)
    
    pipe = munchify(config["pipeline_params"])
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0,
        fx=dataset.fx, fy=dataset.fy,
        cx=dataset.cx, cy=dataset.cy,
        W=dataset.width, H=dataset.height
    ).transpose(0, 1)
    
    image, gt_depth, gt_pose = dataset[args.frame]
    
    view = Camera(
        args.frame, image, gt_depth, gt_pose, projection_matrix,
        dataset.fx, dataset.fy, dataset.cx, dataset.cy,
        dataset.fovx, dataset.fovy, dataset.height, dataset.width,
        device=device
    )
    view.update_RT(gt_pose[:3, :3], gt_pose[:3, 3])
    
    with torch.no_grad():
        render_pkg = render(view, gaussians, pipe, background)
        rendered_depth = render_pkg["depth"]
        rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        rendered_normal = depth_to_normal(view, rendered_depth)
        
    gt_d = gt_depth if isinstance(gt_depth, np.ndarray) else gt_depth.cpu().numpy()
    np.savez(args.out,
        rgb=rendered_image.cpu().numpy().transpose(1, 2, 0),
        depth=rendered_depth[0].cpu().numpy(),
        normal=rendered_normal.cpu().numpy() * 0.5 + 0.5,
        gt_rgb=image.cpu().numpy().transpose(1, 2, 0),
        gt_depth=gt_d
    )

if __name__ == "__main__":
    main()
