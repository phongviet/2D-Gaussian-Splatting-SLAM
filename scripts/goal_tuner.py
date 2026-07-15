#!/usr/bin/env python3
import os
import subprocess
import yaml
import json
import time
import re
import sys

PYTHON_PATH = "/home/phong/miniconda3/envs/2dgslam/bin/python"

def generate_config(base_cfg_path, out_cfg_path, dataset_path, resolution_w, resolution_h, training_params):
    # Load base config
    with open(base_cfg_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Update calibration and dataset path
    config["Dataset"]["dataset_path"] = f"datasets/tum/{dataset_path}"
    if "Calibration" not in config["Dataset"]:
        config["Dataset"]["Calibration"] = {}
    config["Dataset"]["Calibration"]["width"] = int(resolution_w)
    config["Dataset"]["Calibration"]["height"] = int(resolution_h)

    
    # Calculate scaled intrinsics
    # Original: fx=1595.1766, fy=1595.1766, cx=953.6618, cy=716.76733 for 1920x1440
    orig_w, orig_h = 1920, 1440
    fx, fy, cx, cy = 1595.1766, 1595.1766, 953.6618, 716.76733
    scale_x = resolution_w / orig_w
    scale_y = resolution_h / orig_h
    
    config["Dataset"]["Calibration"]["fx"] = float(fx * scale_x)
    config["Dataset"]["Calibration"]["fy"] = float(fy * scale_y)
    config["Dataset"]["Calibration"]["cx"] = float(cx * scale_x)
    config["Dataset"]["Calibration"]["cy"] = float(cy * scale_y)
    config["Dataset"]["Calibration"]["distorted"] = False
    config["Dataset"]["Calibration"]["depth_scale"] = 1000.0

    config["Dataset"]["Calibration"]["k1"] = 0.0
    config["Dataset"]["Calibration"]["k2"] = 0.0
    config["Dataset"]["Calibration"]["p1"] = 0.0
    config["Dataset"]["Calibration"]["p2"] = 0.0
    config["Dataset"]["Calibration"]["k3"] = 0.0

    
    # Disable video recording and W&B for speed
    config["Results"]["save_results"] = True
    config["Results"]["use_gui"] = False
    config["Results"]["eval_rendering"] = True
    config["Results"]["use_wandb"] = False
    
    # Update training parameters
    if "Training" not in config:
        config["Training"] = {}
        
    for k, v in training_params.items():
        if isinstance(v, dict) and k in config["Training"]:
            config["Training"][k].update(v)
        else:
            config["Training"][k] = v
            
    # Write config
    os.makedirs(os.path.dirname(out_cfg_path), exist_ok=True)
    with open(out_cfg_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    return config

def run_slam_with_monitoring(config_path, max_allowable_ate=0.08):
    cmd = [PYTHON_PATH, "slam.py", "--config", config_path, "--eval"]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    
    import signal
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        preexec_fn=os.setsid
    )
    
    last_ate = 0.0
    last_frame = 0
    drifted = False
    
    ate_pattern = re.compile(r"Eval: RMSE ATE\s+\(\d+\s+gaussians,\s+[\d\.]+\s+fps\)\s+([\d\.]+)")
    ate_final_pattern = re.compile(r"Eval: RMSE ATE\s+\(\d+\s+gaussians\)\s+([\d\.]+)")
    frame_pattern = re.compile(r"2dgslam: Evaluating ATE at frame:\s+(\d+)")
    
    print(f"\nStarted SLAM run with config: {config_path}")
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
            
        line_str = line.strip()
        if line_str:
            print(f"  [SLAM] {line_str}")
            
        # Parse frame index
        frame_match = frame_pattern.search(line_str)
        if frame_match:
            last_frame = int(frame_match.group(1))
            
        # Parse ATE
        ate_match = ate_pattern.search(line_str) or ate_final_pattern.search(line_str)
        if ate_match:
            last_ate = float(ate_match.group(1))
            print(f"  >>> Parsed ATE: {last_ate:.4f} m at frame {last_frame}")
            
            # Check for drift limit (skip checking very early frames like frame 60)
            if last_frame > 80 and last_ate > max_allowable_ate:
                print(f"  !!! DRIFT DETECTED: ATE={last_ate:.4f} m (threshold={max_allowable_ate} m) at frame {last_frame} !!!")
                print("  !!! Killing SLAM process group early to save time !!!")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception as e:
                    print(f"Error killing process group: {e}")
                drifted = True
                break
                
    # Wait for group cleanup
    try:
        process.wait(timeout=2)
    except Exception:
        pass
    rc = process.poll()

    print(f"Process finished with exit code: {rc}, Drifted: {drifted}, Final ATE: {last_ate:.4f} m")
    return rc, drifted, last_ate

def main():
    base_cfg = "configs/mono/tum/base_config.yaml"
    out_cfg_dir = "configs/mono/tum/tuning"
    
    # List of configurations to try
    experiments = [
        # --- Group A: Resolution 960x720 ---
        {
            "name": "Res960_BaselineTuned",
            "resolution": (960, 720),
            "dataset": "B1-9a_960x720",
            "params": {
                "pcd_downsample": 32,
                "pcd_downsample_init": 16,
                "tracking_itr_num": 200,
                "mapping_itr_num": 250,
                "window_size": 10,
                "kf_interval": 4,
                "kf_translation": 0.06,
                "kf_min_translation": 0.03,
            }
        },
        {
            "name": "Res960_DensePoints",
            "resolution": (960, 720),
            "dataset": "B1-9a_960x720",
            "params": {
                "pcd_downsample": 16,
                "pcd_downsample_init": 8,
                "tracking_itr_num": 200,
                "mapping_itr_num": 250,
                "window_size": 10,
                "kf_interval": 4,
                "kf_translation": 0.06,
                "kf_min_translation": 0.03,
            }
        },
        {
            "name": "Res960_TighterKF_MaxBA",
            "resolution": (960, 720),
            "dataset": "B1-9a_960x720",
            "params": {
                "pcd_downsample": 24,
                "pcd_downsample_init": 12,
                "tracking_itr_num": 200,
                "mapping_itr_num": 300,
                "window_size": 12,
                "kf_interval": 3,
                "kf_translation": 0.05,
                "kf_min_translation": 0.02,
            }
        },
        {
            "name": "Res960_SlowerLR",
            "resolution": (960, 720),
            "dataset": "B1-9a_960x720",
            "params": {
                "pcd_downsample": 32,
                "pcd_downsample_init": 16,
                "tracking_itr_num": 200,
                "mapping_itr_num": 250,
                "window_size": 10,
                "kf_interval": 4,
                "kf_translation": 0.06,
                "kf_min_translation": 0.03,
                "lr": {
                    "cam_rot_delta": 0.002,
                    "cam_trans_delta": 0.0005,
                }
            }
        },
        # --- Group B: Resolution 1280x960 ---
        {
            "name": "Res1280_BaselineTuned",
            "resolution": (1280, 960),
            "dataset": "B1-9a_1280x960",
            "params": {
                "pcd_downsample": 32,
                "pcd_downsample_init": 16,
                "tracking_itr_num": 200,
                "mapping_itr_num": 250,
                "window_size": 10,
                "kf_interval": 4,
                "kf_translation": 0.06,
                "kf_min_translation": 0.03,
            }
        },
        {
            "name": "Res1280_DensePoints",
            "resolution": (1280, 960),
            "dataset": "B1-9a_1280x960",
            "params": {
                "pcd_downsample": 16,
                "pcd_downsample_init": 8,
                "tracking_itr_num": 200,
                "mapping_itr_num": 250,
                "window_size": 10,
                "kf_interval": 4,
                "kf_translation": 0.06,
                "kf_min_translation": 0.03,
            }
        },
        {
            "name": "Res1280_TighterKF_MaxBA",
            "resolution": (1280, 960),
            "dataset": "B1-9a_1280x960",
            "params": {
                "pcd_downsample": 24,
                "pcd_downsample_init": 12,
                "tracking_itr_num": 200,
                "mapping_itr_num": 300,
                "window_size": 12,
                "kf_interval": 3,
                "kf_translation": 0.05,
                "kf_min_translation": 0.02,
            }
        },
        {
            "name": "Res1280_SlowerLR",
            "resolution": (1280, 960),
            "dataset": "B1-9a_1280x960",
            "params": {
                "pcd_downsample": 32,
                "pcd_downsample_init": 16,
                "tracking_itr_num": 200,
                "mapping_itr_num": 250,
                "window_size": 10,
                "kf_interval": 4,
                "kf_translation": 0.06,
                "kf_min_translation": 0.03,
                "lr": {
                    "cam_rot_delta": 0.002,
                    "cam_trans_delta": 0.0005,
                }
            }
        },
        {
            "name": "Res1280_TighterKF_MaxBA_DensePoints",
            "resolution": (1280, 960),
            "dataset": "B1-9a_1280x960",
            "params": {
                "pcd_downsample": 16,
                "pcd_downsample_init": 8,
                "tracking_itr_num": 200,
                "mapping_itr_num": 300,
                "window_size": 12,
                "kf_interval": 3,
                "kf_translation": 0.05,
                "kf_min_translation": 0.02,
            }
        },
        {
            "name": "Res1280_TighterKF_MaxBA_SlowerLR",
            "resolution": (1280, 960),
            "dataset": "B1-9a_1280x960",
            "params": {
                "pcd_downsample": 24,
                "pcd_downsample_init": 12,
                "tracking_itr_num": 200,
                "mapping_itr_num": 300,
                "window_size": 12,
                "kf_interval": 3,
                "kf_translation": 0.05,
                "kf_min_translation": 0.02,
                "lr": {
                    "cam_rot_delta": 0.002,
                    "cam_trans_delta": 0.0005,
                }
            }
        },
        {
            "name": "Res1280_TighterKF_MaxBA_DensePoints_SlowerLR",
            "resolution": (1280, 960),
            "dataset": "B1-9a_1280x960",
            "params": {
                "pcd_downsample": 16,
                "pcd_downsample_init": 8,
                "tracking_itr_num": 200,
                "mapping_itr_num": 300,
                "window_size": 12,
                "kf_interval": 3,
                "kf_translation": 0.05,
                "kf_min_translation": 0.02,
                "lr": {
                    "cam_rot_delta": 0.002,
                    "cam_trans_delta": 0.0005,
                }
            }
        },
    ]
    
    summary_path = "results/goal_tuner_summary.json"
    history = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                history = json.load(f)
        except Exception:
            pass
            
    print(f"Goal Tuner started. Total runs scheduled: {len(experiments)}")
    
    for idx, exp in enumerate(experiments):
        # Skip if experiment already ran
        if any(h["name"] == exp["name"] for h in history):
            print(f"Skipping {exp['name']} as it is already in summary history.")
            continue

        print(f"\n==================================================")
        print(f"RUNNING EXPERIMENT {idx+1}/{len(experiments)}: {exp['name']}")
        print(f"==================================================")
        
        cfg_path = os.path.join(out_cfg_dir, f"{exp['name']}.yaml")
        generate_config(base_cfg, cfg_path, exp["dataset"], exp["resolution"][0], exp["resolution"][1], exp["params"])
        
        # Monitor run, if ATE goes above 0.08 we kill it
        # However, for finding the BEST config, we can adjust the ATE threshold dynamically.
        # If we already have a run that finished with say 0.33, maybe set threshold to 0.15 to prune bad ones quickly.
        rc, drifted, final_ate = run_slam_with_monitoring(cfg_path, max_allowable_ate=0.15)
        
        exp_result = {
            "name": exp["name"],
            "resolution": f"{exp['resolution'][0]}x{exp['resolution'][1]}",
            "params": exp["params"],
            "exit_code": rc,
            "drifted": drifted,
            "final_ate": final_ate,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        history.append(exp_result)
        
        with open(summary_path, 'w') as f:
            json.dump(history, f, indent=4)
            
        if final_ate <= 0.035 and not drifted:
            print(f"\n🎉 GOAL ACHIEVED! Configuration {exp['name']} reached ATE {final_ate:.4f} m!")
            break
            
if __name__ == "__main__":
    main()
