import os
import torch
import numpy as np
import cv2
import open3d as o3d
from tqdm import tqdm
import argparse
from munch import munchify
import yaml
import json
from scripts.align_utils import umeyama_alignment, apply_sim3_to_gaussians

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.camera_utils import Camera
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2

def post_process_mesh(mesh, cluster_to_keep=50):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    mesh_0 = copy.deepcopy(mesh)
    triangle_clusters, cluster_n_triangles, _ = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) < cluster_to_keep:
        cluster_to_keep = len(cluster_n_triangles)
        
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    return mesh_0

def to_cam_open3d(view, intrinsic):
    extrinsic = np.asarray((view.world_view_transform.T).detach().cpu().numpy())
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = extrinsic
    camera.intrinsic = intrinsic
    return camera

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--ply", type=str, required=True, help="Path to the .ply point cloud file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="TSDF voxel size (m)")
    parser.add_argument("--depth_trunc", type=float, default=5.0, help="Max depth for TSDF (m)")
    parser.add_argument("--skip", type=int, default=5, help="Skip every N frames")
    parser.add_argument("--align_trj", type=str, default=None, help="Path to trj_final.json for Sim(3) alignment")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    device = "cuda"
    
    # Load dataset
    dataset = load_dataset(args, args.config, config)
    
    # Load Gaussian model
    gaussians = GaussianModel(config["model_params"]["sh_degree"])
    gaussians.load_ply(args.ply)
    
    # Optional Sim(3) alignment
    if args.align_trj:
        print(f"Loading trajectory from {args.align_trj} for alignment...")
        with open(args.align_trj, "r") as f:
            trj_data = json.load(f)
        
        est_poses = np.array(trj_data["trj_est"]) # [N, 4, 4]
        gt_poses = np.array(trj_data["trj_gt"])   # [N, 4, 4]
        
        est_centers = est_poses[:, :3, 3]
        gt_centers = gt_poses[:, :3, 3]
        
        s, R, t = umeyama_alignment(est_centers, gt_centers, with_scale=True)
        apply_sim3_to_gaussians(gaussians, s, R, t)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Setup rendering parameters
    pipe = munchify(config["pipeline_params"])
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    # TSDF setup
    voxel_size = args.voxel_size
    sdf_trunc = 5.0 * voxel_size
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # Intrinsics for Open3D
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=dataset.width,
        height=dataset.height,
        fx=dataset.fx,
        fy=dataset.fy,
        cx=dataset.cx,
        cy=dataset.cy
    )
    
    # Projection matrix for Camera object
    projection_matrix = getProjectionMatrix2(
        znear=0.01, zfar=100.0,
        fx=dataset.fx, fy=dataset.fy,
        cx=dataset.cx, cy=dataset.cy,
        W=dataset.width, H=dataset.height
    ).transpose(0, 1)
    
    l1_errors = []
    
    print("Processing frames...")
    for i in tqdm(range(0, dataset.num_imgs, args.skip)):
        # Get data from dataset
        image, gt_depth, gt_pose = dataset[i]
        
        # Create Camera object for rendering
        view = Camera(
            i, image, gt_depth, gt_pose, projection_matrix,
            dataset.fx, dataset.fy, dataset.cx, dataset.cy,
            dataset.fovx, dataset.fovy, dataset.height, dataset.width,
            device=device
        )
        
        # Use GT pose for evaluation and TSDF
        view.update_RT(gt_pose[:3, :3], gt_pose[:3, 3])
        
        # Render
        render_pkg = render(view, gaussians, pipe, background)
        rendered_depth = render_pkg["depth"] # [1, H, W]
        rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        
        # Depth L1 Error (cm)
        if gt_depth is not None:
            mask = gt_depth > 0
            if mask.any():
                # rendered_depth is [1, H, W], gt_depth is [H, W] or similar?
                # dataset returns gt_depth as numpy array probably.
                if isinstance(gt_depth, np.ndarray):
                    gt_depth_pt = torch.from_numpy(gt_depth).to(device)
                else:
                    gt_depth_pt = gt_depth.to(device)
                
                diff = torch.abs(rendered_depth[0] - gt_depth_pt) * 100.0 # to cm
                l1_errors.append(diff[mask].mean().item())
        
        # Integrate into TSDF
        rgb_np = np.ascontiguousarray((rendered_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
        depth_np = np.ascontiguousarray(rendered_depth[0].detach().cpu().numpy().astype(np.float32))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_np),
            o3d.geometry.Image(depth_np),
            depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0
        )
        
        cam_o3d = to_cam_open3d(view, o3d_intrinsic)
        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)
        
    if l1_errors:
        mean_l1 = np.mean(l1_errors)
        print(f"Mean Depth L1 Error: {mean_l1:.4f} cm")
        with open(os.path.join(args.output, "depth_l1_error.txt"), "w") as f:
            f.write(f"{mean_l1}\n")
            
    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")
    
    print("Post-processing mesh...")
    mesh_post = post_process_mesh(mesh)
    print(f"Post-processed mesh has {len(mesh_post.vertices)} vertices.")
    
    mesh_path = os.path.join(args.output, "extracted_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh_post)
    print(f"Mesh saved to {mesh_path}")

if __name__ == "__main__":
    main()
