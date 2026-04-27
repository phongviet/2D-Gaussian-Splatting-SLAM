import cv2
import os
import csv
import numpy as np
from tqdm import tqdm
import shutil
import yaml

def save_config(target_config_path, dataset_name, intrinsics, is_mono=False):
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
                "width": 640,
                "height": 480,
                "distorted": False,
                "depth_scale": 1000.0
            }
        }
    }
    os.makedirs(os.path.dirname(target_config_path), exist_ok=True)
    # Custom dumper to match the existing style (no brackets, simple structure)
    with open(target_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Generated config: {target_config_path}")

def convert():
    dataset_name = "laptoppiano"
    base_path = f"/home/2DGS_SLAM/2dgslam/datasets/wild/{dataset_name}"
    target_path = f"/home/2DGS_SLAM/2dgslam/datasets/tum/{dataset_name}"
    
    rgb_video = os.path.join(base_path, "rgb.mp4")
    odom_file = os.path.join(base_path, "odometry.csv")
    depth_dir = os.path.join(base_path, "depth")
    camera_matrix_file = os.path.join(base_path, "camera_matrix.csv")
    
    output_rgb_dir = os.path.join(target_path, "rgb")
    output_depth_dir = os.path.join(target_path, "depth")
    
    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_depth_dir, exist_ok=True)
    
    # Load camera matrix and calculate scale
    intrinsics = []
    if os.path.exists(camera_matrix_file):
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
        
        target_w, target_h = 640, 480
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        scaled_intrinsics = (fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y)
        print(f"Original resolution: {orig_w}x{orig_h}, Target: {target_w}x{target_h}")
        print(f"Scaled Intrinsics: fx={scaled_intrinsics[0]:.4f}, fy={scaled_intrinsics[1]:.4f}, cx={scaled_intrinsics[2]:.4f}, cy={scaled_intrinsics[3]:.4f}")
        
        # Save configs
        save_config(f"/home/2DGS_SLAM/2dgslam/configs/rgbd/tum/{dataset_name}.yaml", dataset_name, scaled_intrinsics, is_mono=False)
        save_config(f"/home/2DGS_SLAM/2dgslam/configs/mono/tum/{dataset_name}.yaml", dataset_name, scaled_intrinsics, is_mono=True)
    else:
        print("Warning: camera_matrix.csv not found, skipping config generation.")

    # Load odometry
    rows = []
    with open(odom_file, 'r') as f:
        reader = csv.DictReader(f)
        # Clean column names
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            rows.append(row)
    
    print(f"Total frames in odometry: {len(rows)}")
    
    # Extract RGB frames
    cap = cv2.VideoCapture(rgb_video)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_video_frames}")
    
    rgb_txt = []
    depth_txt = []
    gt_txt = []
    
    target_size = (640, 480)
    
    gt_txt.append("# timestamp tx ty tz qx qy qz qw")
    
    for row in tqdm(rows, desc="Converting frames"):
        # Strip all values in the row to avoid leading/trailing spaces
        row = {k: v.strip() for k, v in row.items()}
        
        ts_str = row['timestamp']
        ts = float(ts_str)
        frame_idx = int(row['frame'])
        
        rgb_filename = f"{frame_idx:06d}.png"
        rgb_out_path = os.path.join(output_rgb_dir, rgb_filename)
        
        # Save RGB only if missing
        successfully_read = True
        if not os.path.exists(rgb_out_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, target_size)
                cv2.imwrite(rgb_out_path, frame_resized)
            else:
                print(f"\nWarning: Could not read frame {frame_idx}")
                successfully_read = False
        
        # Only add to dataset if RGB exists
        if os.path.exists(rgb_out_path):
            rgb_txt.append(f"{ts_str} rgb/{rgb_filename}")
            
            # Process Depth
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
                else:
                    print(f"Warning: Depth file {depth_src} missing")
                
            # Ground truth: timestamp tx ty tz qx qy qz qw
            gt_line = f"{ts_str} {row['x']} {row['y']} {row['z']} {row['qx']} {row['qy']} {row['qz']} {row['qw']}"
            gt_txt.append(gt_line)
        else:
            print(f"Skipping frame {frame_idx} due to missing RGB data.")
        
    cap.release()
    
    # Write txt files
    with open(os.path.join(target_path, "rgb.txt"), "w") as f:
        f.write("\n".join(rgb_txt))
        
    with open(os.path.join(target_path, "depth.txt"), "w") as f:
        f.write("\n".join(depth_txt))
        
    with open(os.path.join(target_path, "groundtruth.txt"), "w") as f:
        f.write("\n".join(gt_txt))
        
    print("Conversion complete!")

if __name__ == "__main__":
    convert()
