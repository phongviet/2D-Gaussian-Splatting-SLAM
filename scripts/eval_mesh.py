import os
import torch
import numpy as np
import open3d as o3d
import trimesh
import argparse
from tqdm import tqdm

def calculate_f1(pred_mesh, gt_mesh, threshold=0.01, num_samples=1000000):
    """
    Calculate F1 score between two meshes.
    threshold: distance threshold in meters (default 0.01 = 1cm)
    """
    print(f"Sampling {num_samples} points from meshes...")
    pcd_pred = pred_mesh.sample_points_uniformly(number_of_points=num_samples)
    pcd_gt = gt_mesh.sample_points_uniformly(number_of_points=num_samples)
    
    # Distance from pred to gt (for Precision)
    print("Computing distances from predicted to GT...")
    dists_p2g = pcd_pred.compute_point_cloud_distance(pcd_gt)
    dists_p2g = np.asarray(dists_p2g)
    precision = np.mean(dists_p2g < threshold)
    
    # Distance from gt to pred (for Recall)
    print("Computing distances from GT to predicted...")
    dists_g2p = pcd_gt.compute_point_cloud_distance(pcd_pred)
    dists_g2p = np.asarray(dists_g2p)
    recall = np.mean(dists_g2p < threshold)
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
        
    return precision, recall, f1

def load_mesh(path):
    print(f"Loading mesh from {path}...")
    try:
        mesh = trimesh.load(path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        return o3d_mesh
    except Exception as e:
        print(f"Error loading mesh with trimesh: {e}")
        # Fallback to Open3D
        return o3d.io.read_triangle_mesh(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to the predicted mesh (.ply)")
    parser.add_argument("--gt", type=str, required=True, help="Path to the ground truth mesh (.ply)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Distance threshold (m), default 0.01 (1cm)")
    parser.add_argument("--samples", type=int, default=1000000, help="Number of points to sample")
    parser.add_argument("--output", type=str, default=None, help="Output file to save results")
    args = parser.parse_args()
    
    pred_mesh = load_mesh(args.pred)
    gt_mesh = load_mesh(args.gt)
    
    precision, recall, f1 = calculate_f1(pred_mesh, gt_mesh, threshold=args.threshold, num_samples=args.samples)
    
    print("\nResults:")
    print(f"Precision @ {args.threshold*100:.1f}cm: {precision:.4f}")
    print(f"Recall @ {args.threshold*100:.1f}cm: {recall:.4f}")
    print(f"F1-score @ {args.threshold*100:.1f}cm: {f1:.4f}")
    
    if args.output:
        import json
        results = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(args.threshold),
            "samples": int(args.samples)
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
