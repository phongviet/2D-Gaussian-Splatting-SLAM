#!/usr/bin/env python3
import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def compress_video_to_30s(input_path, output_path, target_fps=30, target_duration=30):
    if not os.path.exists(input_path):
        print(f"Error: Input video does not exist: {input_path}")
        return
        
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_total_frames = target_fps * target_duration
    
    print(f"\nProcessing {os.path.basename(input_path)}:")
    print(f"  Input: {total_frames} frames, {width}x{height}")
    print(f"  Output: {target_total_frames} frames at {target_fps} FPS (exactly {target_duration}s)")
    
    # Calculate indices to sample evenly
    indices = np.linspace(0, total_frames - 1, target_total_frames, dtype=int)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    if not writer.isOpened():
        print(f"Error: Could not open output video writer: {output_path}")
        cap.release()
        return
        
    # Read frames and write selected ones
    for i in tqdm(range(total_frames), desc="Sampling frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        count = np.sum(indices == i)
        for _ in range(count):
            writer.write(frame)
            
    cap.release()
    writer.release()
    print(f"Saved: {output_path}")

def main():
    base_dir = "/home/2DGS_SLAM/2dgslam/results"
    
    # We want to find the latest runs of each wild sequence:
    sequences = [
        "Bedroom_desk", "Studyroom_desk", "Studyroom", 
        "B1-9a", "B1-9b", 
        "B1-9a_1280x960", "B1-9a_960x720",
        "B1-9b_1280x960", "B1-9b_960x720"
    ]

    for seq in sequences:
        # Find directories matching tum_{seq}
        seq_dir_pattern = f"tum_{seq}"
        seq_path = Path(base_dir) / seq_dir_pattern
        if not seq_path.exists():
            print(f"Warning: Sequence path {seq_path} does not exist.")
            continue
            
        # Find the subdirectories (timestamp folders)
        runs = sorted([d for d in seq_path.iterdir() if d.is_dir()])
        if not runs:
            print(f"Warning: No runs found for sequence {seq}.")
            continue
            
        latest_run = runs[-1]
        video_dir = latest_run / "optimization_videos"
        if not video_dir.exists():
            print(f"Warning: video dir {video_dir} does not exist.")
            continue
            
        print(f"\n========================================")
        print(f"Targeting Latest Run for {seq}: {latest_run}")
        print(f"========================================")
        
        videos = ["rgb.mp4", "depth.mp4", "normal.mp4"]
        for v in videos:
            in_path = video_dir / v
            out_path = video_dir / f"{in_path.stem}_30s.mp4"
            compress_video_to_30s(str(in_path), str(out_path))

if __name__ == "__main__":
    main()
