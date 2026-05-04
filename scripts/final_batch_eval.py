import os
import subprocess
import json

# Paths
GS2D_ROOT = "/home/2DGS_SLAM/MonoGS"
GS3D_ROOT = "/home/2DGS_SLAM/Original_MonoGS_Code"
GS2D_PYTHON = "/home/phong/miniconda3/envs/2dgslam/bin/python3"
GS3D_PYTHON = "/home/phong/miniconda3/envs/MonoGS/bin/python3" # User asked to use MonoGS env for 3DGS

# 2DGS Result Folders (Matching README metrics)
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

# 3DGS Result Folders (Latest)
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

GT_MESHES = {
    "replica_office0": "datasets/replica/office0_mesh.ply",
    "replica_office1": "datasets/replica/office1_mesh.ply",
    "replica_office2": "datasets/replica/office2_mesh.ply",
    "replica_office3": "datasets/replica/office3_mesh.ply",
    "replica_office4": "datasets/replica/office4_mesh.ply",
    "replica_room0": "datasets/replica/room0_mesh.ply",
    "replica_room1": "datasets/replica/room1_mesh.ply",
    "replica_room2": "datasets/replica/room2_mesh.ply",
}

def run_eval(repo_root, python_bin, rel_res_path, name_prefix, is_3dgs=False):
    res_abs = os.path.join(repo_root, rel_res_path)
    config = os.path.join(res_abs, "config.yml")
    ply = os.path.join(res_abs, "point_cloud/final/point_cloud.ply")
    trj = os.path.join(res_abs, "plot/trj_final.json")
    output = os.path.join(res_abs, "mesh_eval_final")
    os.makedirs(output, exist_ok=True)
    
    print(f"Evaluating {name_prefix} in {rel_res_path}...")
    
    # 1. Render and Extract Mesh
    cmd_render = [
        python_bin, "scripts/render_and_extract_mesh.py",
        "--config", config,
        "--ply", ply,
        "--output", output,
        "--skip", "1",
        "--align_trj", trj
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
    
    subprocess.run(cmd_render, cwd=repo_root, env=env)
    
    # 2. Mesh Evaluation (F1-score)
    seq_name = None
    for k in GT_MESHES.keys():
        if k in name_prefix:
            seq_name = k
            break
            
    if seq_name and seq_name in GT_MESHES:
        gt_mesh = os.path.join(GS2D_ROOT, GT_MESHES[seq_name])
        pred_mesh = os.path.join(output, "extracted_mesh.ply")
        f1_out = os.path.join(output, "f1_score.json")
        
        cmd_f1 = [
            python_bin, "scripts/eval_mesh.py",
            "--pred", pred_mesh,
            "--gt", gt_mesh,
            "--output", f1_out,
            "--threshold", "0.01" # 1cm
        ]
        subprocess.run(cmd_f1, cwd=GS2D_ROOT, env=env)

def main():
    # Evaluate 2DGS
    print("=== Evaluating 2DGS Results ===")
    for seq, path in GS2D_RESULTS.items():
        run_eval(GS2D_ROOT, GS2D_PYTHON, path, f"2DGS_{seq}")

    # Evaluate 3DGS
    print("=== Evaluating 3DGS Results ===")
    for seq, path in GS3D_RESULTS.items():
        run_eval(GS3D_ROOT, GS3D_PYTHON, path, f"3DGS_{seq}")

if __name__ == "__main__":
    main()
