import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def colorize_depth(depth, min_depth=None, max_depth=None):
    valid = depth > 0
    if min_depth is None:
        min_depth = depth[valid].min() if valid.any() else 0.0
    if max_depth is None:
        max_depth = depth[valid].max() if valid.any() else 1.0
        
    depth_norm = (depth - min_depth) / (max_depth - min_depth + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    
    # Black out invalid
    depth_color[~valid] = 0
    return depth_color

def main():
    d2 = np.load('dump_2dgs.npz')
    d3 = np.load('dump_3dgs.npz')
    
    # Extract
    gt_rgb = d2['gt_rgb']
    gt_depth = d2['gt_depth']
    
    rgb_2dgs = d2['rgb']
    depth_2dgs = d2['depth']
    normal_2dgs = d2['normal']
    
    rgb_3dgs = d3['rgb']
    depth_3dgs = d3['depth']
    normal_3dgs = d3['normal']
    
    # Colorize depth. Use a common scale based on GT depth.
    # TUM office depth usually around 0.5m to 5.0m
    valid = gt_depth > 0
    if valid.any():
        min_d = np.percentile(gt_depth[valid], 1)
        max_d = np.percentile(gt_depth[valid], 99)
    else:
        min_d, max_d = 0.5, 5.0
        
    gt_depth_c = colorize_depth(gt_depth, min_d, max_d)
    depth_2dgs_c = colorize_depth(depth_2dgs, min_d, max_d)
    depth_3dgs_c = colorize_depth(depth_3dgs, min_d, max_d)
    
    # Setup Figure: 3 rows (GT, 3DGS, 2DGS), 3 columns (RGB, Depth, Normal)
    # Wait, the user asked for:
    # GT RGB, Depth
    # 3DGS RGB, Depth, Normal
    # 2DGS RGB, Depth, Normal
    # Better to have 3 rows: GT, 3DGS, 2DGS. Columns: RGB, Depth, Normal. 
    # For GT, leave Normal empty.
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    
    # Row 0: GT
    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title("GT RGB", fontsize=16)
    axes[0, 1].imshow(gt_depth_c)
    axes[0, 1].set_title("GT Depth", fontsize=16)
    axes[0, 2].axis('off') # No GT Normal
    
    # Row 1: 3DGS
    axes[1, 0].imshow(rgb_3dgs)
    axes[1, 0].set_title("MonoGS (Monocular) RGB", fontsize=16)
    axes[1, 1].imshow(depth_3dgs_c)
    axes[1, 1].set_title("MonoGS (Monocular) Depth", fontsize=16)
    axes[1, 2].imshow(normal_3dgs)
    axes[1, 2].set_title("MonoGS (Monocular) Normal", fontsize=16)
    
    # Row 2: 2DGS
    axes[2, 0].imshow(rgb_2dgs)
    axes[2, 0].set_title("2DGSLAM RGB", fontsize=16, fontweight='bold')
    axes[2, 1].imshow(depth_2dgs_c)
    axes[2, 1].set_title("2DGSLAM Depth", fontsize=16, fontweight='bold')
    axes[2, 2].imshow(normal_2dgs)
    axes[2, 2].set_title("2DGSLAM Normal", fontsize=16, fontweight='bold')
    
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
    os.makedirs('media', exist_ok=True)
    plt.savefig('media/comparison_frame_2134.png', bbox_inches='tight', dpi=150)
    print("Saved comparison image to media/comparison_frame_2134.png")

if __name__ == "__main__":
    main()
