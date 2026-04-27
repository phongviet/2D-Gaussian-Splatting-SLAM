import os
import torch
import numpy as np
from PIL import Image
from munch import munchify

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.camera_utils import Camera

def generate_comparison(results_path, config_path, frame_idx, output_name):
    # Load config
    config = load_config(config_path)
    model_params = munchify(config["model_params"])
    pipeline_params = munchify(config["pipeline_params"])
    
    # Initialize Gaussian Model
    gaussians = GaussianModel(model_params.sh_degree, config=config)
    ply_path = os.path.join(results_path, "point_cloud/final/point_cloud.ply")
    print(f"Loading PLY from {ply_path}")
    gaussians.load_ply(ply_path)
    
    # Load Dataset
    print(f"Loading dataset from {model_params.source_path}")
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    
    # Create projection matrix
    projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0, 
        fx=dataset.fx, fy=dataset.fy, 
        cx=dataset.cx, cy=dataset.cy, 
        W=dataset.width, H=dataset.height
    ).transpose(0, 1)
    
    # Initialize Camera from dataset
    camera = Camera.init_from_dataset(dataset, frame_idx, projection_matrix)
    
    # Load Estimated Poses
    import json
    trj_path = os.path.join(results_path, "plot/trj_final.json")
    with open(trj_path, "r") as f:
        trj_data = json.load(f)
    
    # Find the index in trj_data corresponding to frame_idx
    try:
        idx_in_trj = trj_data["trj_id"].index(frame_idx)
    except ValueError:
        print(f"Frame {frame_idx} is not a keyframe or not in trj_final.json")
        return
        
    pose_est_c2w = np.array(trj_data["trj_est"][idx_in_trj])
    pose_est_w2c = np.linalg.inv(pose_est_c2w)
    
    # SYNC POSES: Set current pose (R, T) to Estimated Pose
    camera.R = torch.from_numpy(pose_est_w2c[:3, :3]).float().cuda()
    camera.T = torch.from_numpy(pose_est_w2c[:3, 3]).float().cuda()
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # Render
    with torch.no_grad():
        rendering = render(camera, gaussians, pipeline_params, background)["render"]
        rendered_image = torch.clamp(rendering, 0.0, 1.0)

    # Convert to NumPy for saving
    gt_image_torch = camera.original_image
    gt_np = (gt_image_torch.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    rendered_np = (rendered_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    
    # Create side-by-side
    combined = np.hstack((rendered_np, gt_np))
    
    # Save
    os.makedirs("media", exist_ok=True)
    save_path = os.path.join("media", output_name)
    Image.fromarray(combined).save(save_path)
    print(f"Saved comparison to {save_path}")

if __name__ == "__main__":
    scene = "office3"
    results = "results/replica_office3/2026-04-25-13-01-00"
    config = "configs/mono/replica/office3.yaml"
    
    import json
    trj_path = os.path.join(results, "plot/trj_final.json")
    with open(trj_path, "r") as f:
        trj_data = json.load(f)
    kf_ids = trj_data["trj_id"]
    
    # Pick some keyframes spaced out
    selected_fids = [kf_ids[len(kf_ids)//4], kf_ids[len(kf_ids)//2], kf_ids[-10]]
    
    for fid in selected_fids:
        out = f"comparison_{scene}_frame{fid}.png"
        generate_comparison(results, config, fid, out)
