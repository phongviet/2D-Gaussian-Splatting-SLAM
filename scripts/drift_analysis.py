#!/usr/bin/env python3
"""Analyze B1-9a drift: plot ATE, tracking iters, mapping loss, and gaussian count over frames."""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

results_dir = "/home/2DGS_SLAM/2dgslam/results/tum_B1-9a/2026-07-14-09-01-11"
out_dir = results_dir

# Load data
with open(os.path.join(results_dir, "metrics_time_series.json")) as f:
    ts = json.load(f)
with open(os.path.join(results_dir, "tracking_iterations.json")) as f:
    ti = json.load(f)

# ATE checkpoints from log
ate_frames = [60, 117, 156, 189, 221, 263, 304, 360, 408, 432]
ate_values = [0.0082, 0.0129, 0.0195, 0.0594, 0.1709, 0.2567, 0.3120, 0.3289, 0.3310, 0.3386]

frames_ts = list(range(len(ts["gaussian_counts"])))
tracking_frames = ti["frame_indices"]
tracking_iters = ti["iters"]

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
fig.suptitle("B1-9a Drift Analysis (Tuned Run)", fontsize=16, fontweight="bold", y=0.98)

# Define drift zones
zones = [
    (0, 156, "#2ecc71", "Stable (ATE < 2cm)", 0.15),
    (157, 220, "#f39c12", "Drift Onset (ATE 6→17cm)", 0.15),
    (221, 304, "#e74c3c", "Severe Drift (ATE 17→31cm)", 0.15),
    (305, 432, "#95a5a6", "Plateau (ATE ~33cm)", 0.15),
]

for ax in axes:
    for start, end, color, label, alpha in zones:
        ax.axvspan(start, end, alpha=alpha, color=color)

# 1. ATE over time
ax1 = axes[0]
ax1.plot(ate_frames, ate_values, "o-", color="#e74c3c", linewidth=2, markersize=8, zorder=5)
ax1.set_ylabel("RMSE ATE (m)", fontsize=12)
ax1.set_title("Absolute Trajectory Error", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, label="5cm threshold")
for f, v in zip(ate_frames, ate_values):
    ax1.annotate(f"{v:.3f}m", (f, v), textcoords="offset points", xytext=(0, 10),
                 fontsize=8, ha="center", fontweight="bold")

# 2. Tracking iterations
ax2 = axes[1]
ax2.bar(tracking_frames, tracking_iters, width=1.0, color="#3498db", alpha=0.7, zorder=3)
ax2.axhline(y=200, color="red", linestyle="--", alpha=0.7, label="Max iters (200)")
ax2.set_ylabel("Tracking Iterations", fontsize=12)
ax2.set_title("Frontend Pose Tracking Iterations per Frame", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper left")

# 3. Mapping loss
ax3 = axes[2]
ax3.plot(frames_ts, ts["mapping_losses"], color="#9b59b6", linewidth=1.0, alpha=0.8, zorder=3)
ax3.set_ylabel("Mapping Loss", fontsize=12)
ax3.set_title("Backend Mapping Loss", fontsize=13, fontweight="bold")
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="Low-loss regime")
ax3.legend(loc="upper left")

# 4. Gaussian count
ax4 = axes[3]
ax4.plot(frames_ts, ts["gaussian_counts"], color="#e67e22", linewidth=1.5, zorder=3)
ax4.set_ylabel("Gaussian Count", fontsize=12)
ax4.set_xlabel("Frame Index", fontsize=12)
ax4.set_title("Scene Gaussian Count", fontsize=13, fontweight="bold")
ax4.grid(True, alpha=0.3)

# Add zone legend
legend_patches = [mpatches.Patch(color=c, alpha=0.4, label=l) for _, _, c, l, _ in zones]
fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 0.01), frameon=True)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
out_path = os.path.join(out_dir, "drift_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
