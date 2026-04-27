import os
import torch
import numpy as np
from PIL import Image
from munch import munchify
import json
import random

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.camera_utils import Camera

def generate_comparison(gaussians, dataset, pipeline_params, projection_matrix, frame_idx, output_dir, output_name):
    # Initialize Camera from dataset
    camera = Camera.init_from_dataset(dataset, frame_idx, projection_matrix)
    
    # Set current pose (R, T) to Estimated Pose if available, else we can't really render it well
    # But for this task, I'll assume we only pick frames that have estimated poses
    # These are passed into the function by ensuring frame_idx is valid
    
    # The actual pose setting will be done outside to avoid reloading trj_final every time
    # So I'll pass the w2c pose as well
    pass

def run_batch_generation(scene, results_dir, config_path, num_samples=50):
    print(f"--- Processing {scene} ---")
    config = load_config(config_path)
    model_params = munchify(config["model_params"])
    pipeline_params = munchify(config["pipeline_params"])
    
    # Initialize Gaussian Model
    gaussians = GaussianModel(model_params.sh_degree, config=config)
    ply_path = os.path.join(results_dir, "point_cloud/final/point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Skipping {scene}, PLY not found at {ply_path}")
        return
    gaussians.load_ply(ply_path)
    
    # Load Dataset
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    
    # Create projection matrix
    projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0, 
        fx=dataset.fx, fy=dataset.fy, 
        cx=dataset.cx, cy=dataset.cy, 
        W=dataset.width, H=dataset.height
    ).transpose(0, 1)
    
    # Load Estimated Poses
    trj_path = os.path.join(results_dir, "plot/trj_final.json")
    if not os.path.exists(trj_path):
        print(f"Skipping {scene}, trj_final.json not found")
        return
    with open(trj_path, "r") as f:
        trj_data = json.load(f)
    
    kf_ids = trj_data["trj_id"]
    trj_est = trj_data["trj_est"]
    
    # Sample 50 indices
    if len(kf_ids) > num_samples:
        indices = random.sample(range(len(kf_ids)), num_samples)
    else:
        indices = range(len(kf_ids))
    
    output_scene_dir = os.path.join("media", scene)
    os.makedirs(output_scene_dir, exist_ok=True)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    for idx in indices:
        fid = kf_ids[idx]
        pose_est_c2w = np.array(trj_est[idx])
        pose_est_w2c = np.linalg.inv(pose_est_c2w)
        
        camera = Camera.init_from_dataset(dataset, fid, projection_matrix)
        camera.R = torch.from_numpy(pose_est_w2c[:3, :3]).float().cuda()
        camera.T = torch.from_numpy(pose_est_w2c[:3, 3]).float().cuda()
        
        with torch.no_grad():
            rendering = render(camera, gaussians, pipeline_params, background)["render"]
            rendered_image = torch.clamp(rendering, 0.0, 1.0)

        gt_image_torch = camera.original_image
        gt_np = (gt_image_torch.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        rendered_np = (rendered_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        
        combined = np.hstack((rendered_np, gt_np))
        Image.fromarray(combined).save(os.path.join(output_scene_dir, f"frame_{fid:06}.png"))
        
    print(f"Done {scene}: {len(indices)} images saved to {output_scene_dir}")

if __name__ == "__main__":
    replica_scenes = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
    tum_scenes = [
        ("fr1_desk", "results/tum_rgbd_dataset_freiburg1_desk", "configs/mono/tum/fr1_desk.yaml"),
        ("fr2_xyz", "results/tum_rgbd_dataset_freiburg2_xyz", "configs/mono/tum/fr2_xyz.yaml"),
        ("fr3_office", "results/tum_rgbd_dataset_freiburg3_long_office_household", "configs/mono/tum/fr3_office.yaml")
    ]
    
    # Process Replica
    for scene in replica_scenes:
        base_results = f"results/replica_{scene}"
        if not os.path.exists(base_results):
            continue
        latest_run = sorted(os.listdir(base_results))[-1]
        results_dir = os.path.join(base_results, latest_run)
        config_path = f"configs/mono/replica/{scene}.yaml"
        run_batch_generation(scene, results_dir, config_path, num_samples=50)

    # Process TUM
    for scene_name, base_results, config_path in tum_scenes:
        if not os.path.exists(base_results):
            print(f"No results for {scene_name}")
            continue
        latest_run = sorted(os.listdir(base_results))[-1]
        results_dir = os.path.join(base_results, latest_run)
        run_batch_generation(scene_name, results_dir, config_path, num_samples=50)
