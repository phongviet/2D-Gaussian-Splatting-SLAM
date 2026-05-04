import os
import json

GS2D_ROOT = "/home/2DGS_SLAM/MonoGS"
GS3D_ROOT = "/home/2DGS_SLAM/Original_MonoGS_Code"

GS2D_RESULTS = {
    "tum_fr1_desk": "results/tum_rgbd_dataset_freiburg1_desk/2026-04-27-17-57-19",
    "tum_fr2_xyz": "results/tum_rgbd_dataset_freiburg2_xyz/2026-04-28-01-47-13",
    "tum_fr3_office": "results/tum_rgbd_dataset_freiburg3_long_office_household/2026-04-28-04-18-55",
    "replica_office0": "results/replica_office0/2026-04-29-22-35-42",
    "replica_office1": "results/replica_office1/2026-04-30-02-25-50",
    "replica_office2": "results/replica_office2/2026-04-30-05-35-07",
    "replica_office3": "results/replica_office3/2026-04-30-10-46-13",
    "replica_office4": "results/replica_office4/2026-04-30-15-09-04",
    "replica_room0": "results/replica_room0/2026-04-30-19-40-16",
    "replica_room1": "results/replica_room1/2026-05-01-02-06-35",
    "replica_room2": "results/replica_room2/2026-05-01-06-02-40",
}

GS3D_RESULTS = {
    "tum_fr1_desk": "results/tum_rgbd_dataset_freiburg1_desk/2026-05-03-23-05-59",
    "tum_fr2_xyz": "results/tum_rgbd_dataset_freiburg2_xyz/2026-05-03-23-26-17",
    "tum_fr3_office": "results/datasets_tum/2026-05-04-00-37-08",
    "replica_office0": "results/replica_office0/2026-05-04-01-38-26",
    "replica_office1": "results/replica_office1/2026-05-04-02-55-11",
    "replica_office2": "results/replica_office2/2026-05-04-03-58-40",
    "replica_office3": "results/replica_office3/2026-05-04-05-32-47",
    "replica_office4": "results/replica_office4/2026-05-04-06-57-33",
    "replica_room0": "results/replica_room0/2026-05-04-08-19-07",
    "replica_room1": "results/replica_room1/2026-05-04-10-02-21",
    "replica_room2": "results/replica_room2/2026-05-04-11-30-36",
}

def get_metrics(repo_root, rel_path, is_replica=True):
    abs_path = os.path.join(repo_root, rel_path)
    metrics = {}
    
    # 1. Tracking and general metrics
    summ_path = os.path.join(abs_path, "eval_summary.json")
    if os.path.exists(summ_path):
        with open(summ_path, 'r') as f:
            d = json.load(f)
            metrics['ate'] = d.get('rmse_ate_m', -1)
            metrics['fps'] = d.get('total_fps', -1)
            metrics['psnr'] = d.get('mean_psnr', -1)
            metrics['ssim'] = d.get('mean_ssim', -1)
            metrics['lpips'] = d.get('mean_lpips', -1)
            metrics['gaussians'] = d.get('gaussian_count', -1)
    
    # 2. Depth L1
    l1_path = os.path.join(abs_path, "mesh_eval_final/depth_l1_error.txt")
    if os.path.exists(l1_path):
        with open(l1_path, 'r') as f:
            content = f.read().strip()
            if content:
                try:
                    metrics['depth_l1'] = float(content)
                except:
                    metrics['depth_l1'] = -1
            
    # 3. F1 score
    if is_replica:
        f1_path = os.path.join(abs_path, "mesh_eval_final/f1_score.json")
        if os.path.exists(f1_path):
            with open(f1_path, 'r') as f:
                d = json.load(f)
                metrics['f1'] = d.get('f1', -1)
                metrics['precision'] = d.get('precision', -1)
                metrics['recall'] = d.get('recall', -1)
    return metrics

def main():
    all_data = {}
    for seq in GS2D_RESULTS.keys():
        is_replica = "replica" in seq
        all_data[seq] = {
            "2DGS": get_metrics(GS2D_ROOT, GS2D_RESULTS[seq], is_replica),
            "3DGS": get_metrics(GS3D_ROOT, GS3D_RESULTS[seq], is_replica)
        }
    
    print(json.dumps(all_data, indent=2))

if __name__ == "__main__":
    main()
