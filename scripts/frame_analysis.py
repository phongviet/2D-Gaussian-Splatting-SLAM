#!/usr/bin/env python3
"""Experiment 3: Investigate frames 150-200 in B1-9a for motion blur, texturelessness, or fast rotation."""
import cv2
import numpy as np
import os
import json
from pathlib import Path

dataset_dir = "/home/2DGS_SLAM/2dgslam/datasets/tum/B1-9a"
rgb_dir = os.path.join(dataset_dir, "rgb")
output_dir = "/home/2DGS_SLAM/2dgslam/results/tum_B1-9a/2026-07-14-09-01-11/frame_analysis"
os.makedirs(output_dir, exist_ok=True)

# Analyze frames 140-210 (the drift zone and surrounding stable frames)
frame_range = list(range(140, 211))

results = []

for frame_idx in frame_range:
    img_path = os.path.join(rgb_dir, f"{frame_idx:06d}.png")
    if not os.path.exists(img_path):
        continue
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Laplacian variance (blur detection) - lower = more blurry
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Gradient magnitude (texture richness)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2).mean()
    
    # 3. ORB feature count (trackable features)
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints = orb.detect(gray, None)
    n_features = len(keypoints)
    
    # 4. Brightness
    brightness = gray.mean()
    
    # 5. Optical flow (frame-to-frame motion) - compare to previous frame
    motion_magnitude = 0.0
    if frame_idx > frame_range[0]:
        prev_path = os.path.join(rgb_dir, f"{frame_idx-1:06d}.png")
        if os.path.exists(prev_path):
            prev_gray = cv2.cvtColor(cv2.imread(prev_path), cv2.COLOR_BGR2GRAY)
            # Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_magnitude = mag.mean()
    
    results.append({
        "frame": frame_idx,
        "laplacian_var": round(float(laplacian_var), 2),
        "gradient_mag": round(float(gradient_mag), 2),
        "orb_features": n_features,
        "brightness": round(float(brightness), 2),
        "motion_magnitude": round(float(motion_magnitude), 4),
    })

# Save raw results
with open(os.path.join(output_dir, "frame_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)

# Print summary table
print(f"{'Frame':>6} | {'Laplacian':>10} | {'Gradient':>9} | {'ORB Feat':>9} | {'Bright':>7} | {'Motion':>8} | Notes")
print("-" * 80)

# Calculate baselines from stable frames (140-155)
stable = [r for r in results if r["frame"] <= 155]
baseline_lap = np.mean([r["laplacian_var"] for r in stable])
baseline_grad = np.mean([r["gradient_mag"] for r in stable])
baseline_feat = np.mean([r["orb_features"] for r in stable])
baseline_motion = np.mean([r["motion_magnitude"] for r in stable])

for r in results:
    notes = []
    if r["laplacian_var"] < baseline_lap * 0.5:
        notes.append("BLURRY")
    if r["gradient_mag"] < baseline_grad * 0.6:
        notes.append("LOW-TEXTURE")
    if r["orb_features"] < baseline_feat * 0.5:
        notes.append("FEW-FEATURES")
    if r["motion_magnitude"] > baseline_motion * 2.0:
        notes.append("FAST-MOTION")
    if r["brightness"] < 50:
        notes.append("DARK")
    
    marker = " ***" if r["frame"] in [170, 175, 184] else ""
    print(f"{r['frame']:>6} | {r['laplacian_var']:>10.2f} | {r['gradient_mag']:>9.2f} | {r['orb_features']:>9d} | {r['brightness']:>7.2f} | {r['motion_magnitude']:>8.4f} | {', '.join(notes)}{marker}")

print(f"\nBaseline (frames 140-155): Laplacian={baseline_lap:.2f}, Gradient={baseline_grad:.2f}, ORB={baseline_feat:.0f}, Motion={baseline_motion:.4f}")

# Create visualization: montage of key frames
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

key_frames = [145, 150, 155, 160, 165, 168, 170, 172, 175, 180, 184, 190, 195, 200, 205, 210]
fig, axes = plt.subplots(4, 4, figsize=(20, 15))
fig.suptitle("B1-9a Frame Investigation: Frames 145–210\n(Red border = tracking failure frame)", fontsize=16, fontweight="bold")

for ax_idx, frame_idx in enumerate(key_frames):
    ax = axes[ax_idx // 4, ax_idx % 4]
    img_path = os.path.join(rgb_dir, f"{frame_idx:06d}.png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
    
    # Find metrics for this frame
    r = next((x for x in results if x["frame"] == frame_idx), None)
    if r:
        info = f"Lap={r['laplacian_var']:.0f} Feat={r['orb_features']} Mot={r['motion_magnitude']:.3f}"
    else:
        info = ""
    
    is_failure = frame_idx in [170, 175, 184]
    color = "red" if is_failure else ("orange" if 157 <= frame_idx <= 190 else "green")
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(4 if is_failure else 2)
    
    ax.set_title(f"Frame {frame_idx}", fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel(info, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
montage_path = os.path.join(output_dir, "frame_montage.png")
plt.savefig(montage_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved frame montage to: {montage_path}")

# Plot metrics over frame range
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle("B1-9a Frame Quality Metrics (Frames 140–210)", fontsize=16, fontweight="bold", y=0.98)

frames = [r["frame"] for r in results]
failure_frames = [170, 175, 184]

for ax in axes:
    ax.axvspan(157, 190, alpha=0.15, color="red", label="Drift zone")
    for ff in failure_frames:
        ax.axvline(x=ff, color="red", linestyle="--", alpha=0.5)

axes[0].plot(frames, [r["laplacian_var"] for r in results], "b-o", markersize=3)
axes[0].axhline(y=baseline_lap * 0.5, color="red", linestyle=":", alpha=0.7, label="50% baseline")
axes[0].set_ylabel("Laplacian Var\n(blur)")
axes[0].set_title("Sharpness (higher = sharper)", fontsize=11)
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(frames, [r["gradient_mag"] for r in results], "g-o", markersize=3)
axes[1].axhline(y=baseline_grad * 0.6, color="red", linestyle=":", alpha=0.7, label="60% baseline")
axes[1].set_ylabel("Gradient Mag\n(texture)")
axes[1].set_title("Texture Richness (higher = more texture)", fontsize=11)
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

axes[2].plot(frames, [r["orb_features"] for r in results], "m-o", markersize=3)
axes[2].axhline(y=baseline_feat * 0.5, color="red", linestyle=":", alpha=0.7, label="50% baseline")
axes[2].set_ylabel("ORB Features")
axes[2].set_title("Trackable Feature Count", fontsize=11)
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

axes[3].plot(frames, [r["motion_magnitude"] for r in results], "r-o", markersize=3)
axes[3].axhline(y=baseline_motion * 2.0, color="red", linestyle=":", alpha=0.7, label="2x baseline")
axes[3].set_ylabel("Optical Flow\n(motion)")
axes[3].set_title("Inter-Frame Motion (higher = faster movement)", fontsize=11)
axes[3].legend(loc="upper right")
axes[3].grid(True, alpha=0.3)

axes[4].plot(frames, [r["brightness"] for r in results], "orange", marker="o", markersize=3)
axes[4].set_ylabel("Brightness")
axes[4].set_xlabel("Frame Index")
axes[4].set_title("Mean Brightness", fontsize=11)
axes[4].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
metrics_path = os.path.join(output_dir, "frame_quality_metrics.png")
plt.savefig(metrics_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved frame quality metrics to: {metrics_path}")
