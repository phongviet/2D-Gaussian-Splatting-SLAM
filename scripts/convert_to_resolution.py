#!/usr/bin/env python3
import cv2
import os
import csv
import numpy as np
from tqdm import tqdm
import yaml
import sys

def save_config(target_config_path, dataset_name, intrinsics, width, height, is_mono=True):
    config = {
        "inherit_from": f"configs/{'mono' if is_mono else 'rgbd'}/tum/base_config.yaml",
        "Dataset": {
            "dataset_path": f"datasets/tum/{dataset_name}",
            "Calibration": {
                "fx": float(intrinsics[0]),
                "fy": float(intrinsics[1]),
                "cx": float(intrinsics[2]),
                "cy": float(intrinsics[3]),
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
                "width": int(width),
                "height": int(height),
                "distorted": False,
                "depth_scale": 1000.0
            }
        }
    }
    os.makedirs(os.path.dirname(target_config_path), exist_ok=True)
    with open(target_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Generated config: {target_config_path}")

def convert_dataset(name, target_w, target_h):
    base_path = f"/home/2DGS_SLAM/2dgslam/datasets/wild/{name}"
    target_name = f"{name}_{target_w}x{target_h}"
    target_path = f"/home/2DGS_SLAM/2dgslam/datasets/tum/{target_name}"
    
    rgb_video = os.path.join(base_path, "rgb.mp4")
    odom_file = os.path.join(base_path, "odometry.csv")
    depth_dir = os.path.join(base_path, "depth")
    camera_matrix_file = os.path.join(base_path, "camera_matrix.csv")
    
    if not os.path.exists(rgb_video) or not os.path.exists(odom_file):
        print(f"Error: dataset {name} not found or incomplete in datasets/wild.")
        return False
        
    output_rgb_dir = os.path.join(target_path, "rgb")
    output_depth_dir = os.path.join(target_path, "depth")
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)
    
    # Calculate scale
    with open(camera_matrix_file, 'r') as f:
        reader = csv.reader(f)
        matrix = list(reader)
        fx = float(matrix[0][0])
        fy = float(matrix[1][1])
        cx = float(matrix[0][2])
        cy = float(matrix[1][2])
        
    cap_temp = cv2.VideoCapture(rgb_video)
    orig_w = cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_temp.release()
    
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    scaled_intrinsics = (fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y)
    print(f"Converting {name} to {target_w}x{target_h}...")
    print(f"Scaled Intrinsics: fx={scaled_intrinsics[0]:.4f}, fy={scaled_intrinsics[1]:.4f}, cx={scaled_intrinsics[2]:.4f}, cy={scaled_intrinsics[3]:.4f}")
    
    save_config(f"/home/2DGS_SLAM/2dgslam/configs/mono/tum/{target_name}.yaml", target_name, scaled_intrinsics, target_w, target_h, is_mono=True)
    
    # Load odometry
    rows = []
    with open(odom_file, 'r') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            rows.append(row)
            
    cap = cv2.VideoCapture(rgb_video)
    rgb_txt = []
    depth_txt = []
    gt_txt = []
    
    gt_txt.append("# timestamp tx ty tz qx qy qz qw")
    
    target_size = (target_w, target_h)
    
    for row in tqdm(rows, desc="Frames"):
        row = {k: v.strip() for k, v in row.items()}
        ts_str = row['timestamp']
        frame_idx = int(row['frame'])
        
        rgb_filename = f"{frame_idx:06d}.png"
        rgb_out_path = os.path.join(output_rgb_dir, rgb_filename)
        
        if not os.path.exists(rgb_out_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, target_size)
                cv2.imwrite(rgb_out_path, frame_resized)
            else:
                continue
                
        if os.path.exists(rgb_out_path):
            rgb_txt.append(f"{ts_str} rgb/{rgb_filename}")
            
            depth_filename = f"{frame_idx:06d}.png"
            depth_out_path = os.path.join(output_depth_dir, depth_filename)
            depth_txt.append(f"{ts_str} depth/{depth_filename}")
            
            if not os.path.exists(depth_out_path):
                depth_src = os.path.join(depth_dir, f"{frame_idx:06d}.png")
                if os.path.exists(depth_src):
                    depth_img = cv2.imread(depth_src, cv2.IMREAD_UNCHANGED)
                    if depth_img is not None:
                        depth_upscaled = cv2.resize(depth_img, target_size, interpolation=cv2.INTER_NEAREST)
                        cv2.imwrite(depth_out_path, depth_upscaled)
                        
            gt_line = f"{ts_str} {row['x']} {row['y']} {row['z']} {row['qx']} {row['qy']} {row['qz']} {row['qw']}"
            gt_txt.append(gt_line)
            
    cap.release()
    
    with open(os.path.join(target_path, "rgb.txt"), "w") as f:
        f.write("\n".join(rgb_txt))
    with open(os.path.join(target_path, "depth.txt"), "w") as f:
        f.write("\n".join(depth_txt))
    with open(os.path.join(target_path, "groundtruth.txt"), "w") as f:
        f.write("\n".join(gt_txt))
        
    print(f"Conversion of {target_name} complete!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: convert_to_resolution.py <name> <width> <height>")
        sys.exit(1)
    convert_dataset(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
