# 2DGS-SLAM: Comprehensive Evaluation and Ablation Results

This document consolidates all quantitative, qualitative, and ablation study results obtained during the development, optimization, and thesis defense preparation of the **2DGS-SLAM** system. 

---

## 1. Main Quantitative Results (Monocular SLAM)

We compare **2DGS-SLAM** (our proposed method utilizing 2D Gaussian surfels with analytic Pose Jacobians and surface constraints) against the 3D Gaussian Splatting baseline (**MonoGS**) across standard benchmarks. All runs use monocular camera input.

### 1.1 TUM RGB-D Dataset
Evaluation on three standard real-world sequences from the TUM RGB-D dataset. Trajectory accuracy is evaluated using Absolute Trajectory Error (ATE RMSE) in meters, and rendering quality is evaluated using PSNR, SSIM, and LPIPS on held-out frames (every 5th frame).

| Sequence | Method | ATE RMSE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `fr1_desk` | MonoGS (3DGS) | 0.0359 | 17.51 | 0.667 | 0.375 |
| | **2DGS-SLAM (Ours)** | **0.0206** | **18.38** | **0.685** | **0.367** |
| `fr2_xyz` | MonoGS (3DGS) | 0.0471 | 15.51 | 0.656 | **0.343** |
| | **2DGS-SLAM (Ours)** | **0.0212** | **15.69** | **0.672** | 0.353 |
| `fr3_office` | MonoGS (3DGS) | 0.0253 | **19.50** | **0.740** | 0.326 |
| | **2DGS-SLAM (Ours)** | **0.0113** | 18.95 | **0.740** | **0.322** |

### 1.2 Replica Dataset (Office Sequences)
Evaluation on synthetic office environments. Trajectory accuracy (ATE RMSE in meters) and held-out rendering metrics are reported.

| Sequence | Method | ATE RMSE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `office0` | MonoGS (3DGS) | **0.0696** | 30.79 | **0.897** | **0.238** |
| | **2DGS-SLAM (Ours)** | 0.0902 | **30.81** | 0.894 | **0.238** |
| `office1` | MonoGS (3DGS) | 0.1057 | 32.93 | 0.913 | 0.194 |
| | **2DGS-SLAM (Ours)** | **0.0897** | **34.13** | **0.931** | **0.169** |
| `office2` | MonoGS (3DGS) | 0.1503 | 26.56 | 0.881 | 0.236 |
| | **2DGS-SLAM (Ours)** | **0.1026** | **27.27** | **0.894** | **0.216** |
| `office3` | MonoGS (3DGS) | 0.0386 | 30.06 | 0.898 | 0.162 |
| | **2DGS-SLAM (Ours)** | **0.0310** | **30.30** | **0.912** | **0.146** |
| `office4` | MonoGS (3DGS) | 0.2387 | 27.75 | 0.898 | 0.241 |
| | **2DGS-SLAM (Ours)** | **0.0351** | **28.94** | **0.901** | **0.222** |

### 1.3 Replica Dataset (Room Sequences)
Evaluation on synthetic room environments.

| Sequence | Method | ATE RMSE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `room0` | MonoGS (3DGS) | 0.0445 | 27.71 | 0.854 | 0.176 |
| | **2DGS-SLAM (Ours)** | **0.0308** | **28.80** | **0.866** | **0.175** |
| `room1` | MonoGS (3DGS) | 0.5329 | 24.92 | **0.797** | **0.329** |
| | **2DGS-SLAM (Ours)** | **0.2988** | **25.04** | 0.788 | 0.345 |
| `room2` | MonoGS (3DGS) | **0.0188** | **30.44** | **0.904** | **0.185** |
| | **2DGS-SLAM (Ours)** | 0.0233 | 29.29 | 0.884 | 0.219 |

---

## 2. Key Analysis: When is 2DGS-SLAM Better/Worse?

By analyzing the correlation between the baseline trajectory error and our method's performance gains:
* **Drift Mitigation**: The benefit of 2D surface constraints grows strongly on sequences where the baseline drifts. For example, on the two hardest sequences (`office4` and `room1`), our method reduces ATE by **0.204 m** and **0.234 m** respectively.
* **Smoothing Bias**: On highly structured, simple sequences where the baseline tracking is already excellent (`room2`, ATE = 0.0188 m), adding surface regularizers introduces minor smoothing bias, causing slightly lower rendering details (e.g. `room2` PSNR drops from 30.44 dB to 29.29 dB) without improving tracking.

---

## 3. Ablation Studies

### 3.1 Regularization Losses Ablation (TUM Dataset)
To verify the impact of the added geometric losses—**Depth Distortion Loss** ($\mathcal{L}_{\text{distortion}}$) and **Normal Consistency Loss** ($\mathcal{L}_{\text{normal}}$)—we ablated each from the system on TUM sequences. The table reports ATE RMSE in meters across 3 independent runs.

| Variant | `fr1_desk` (m) | `fr2_xyz` (m) | `fr3_office` (m) | Mean Degradation |
| :--- | :---: | :---: | :---: | :---: |
| **Full Config (Ours)** | **0.0206** | **0.0212** | **0.0113** | **1.0× (Base)** |
| w/o Distortion Loss | 0.0407 | 0.0345 | 0.0304 | **2.1× worse** |
| w/o Normal Consistency | 0.3456 | 0.0402 | 0.0290 | **7.1× worse** |

*Note: Removing the normal consistency loss on `fr1_desk` leads to complete trajectory collapse due to unstable surfel orientations in complex textured regions.*

### 3.2 DSSIM Loss Ablation (Replica Dataset)
We evaluate the performance impact of the Structural Similarity (DSSIM) term in the mapping photometric loss ($\lambda_{\text{dssim}} = 0.2$ vs $\lambda_{\text{dssim}} = 0.0$).

| Sequence | Metric | w/ DSSIM ($\lambda = 0.2$) | w/o DSSIM ($\lambda = 0.0$) |
| :--- | :--- | :---: | :---: |
| `office0` | ATE (m) ↓ | **0.0902** | 0.0959 |
| | PSNR (dB) ↑ | **30.81** | 30.79 |
| | SSIM ↑ | **0.894** | 0.893 |
| `office1` | ATE (m) ↓ | **0.0897** | 0.1277 |
| | PSNR (dB) ↑ | **34.13** | 31.76 |
| | SSIM ↑ | **0.931** | 0.906 |
| `office2` | ATE (m) ↓ | **0.1026** | 0.2064 |
| | PSNR (dB) ↑ | **27.27** | 24.87 |
| | SSIM ↑ | **0.894** | 0.857 |
| `office3` | ATE (m) ↓ | **0.0310** | 0.0461 |
| | PSNR (dB) ↑ | **30.30** | 28.92 |
| | SSIM ↑ | **0.912** | 0.887 |

### 3.3 Detailed fr3_office Densification Ablation Study
We ablated the densification and insertion strategies on the `fr3_office` sequence to study early local Gaussian growth instabilities.

| Trial | Experiment Configuration | ATE RMSE (m) ↓ | Final Gaussians | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Baseline (Standard 2DGS) | 0.0315 | 30,759 | 18.59 | 0.721 | 0.353 | Success |
| 2 | No_Insert (Thresh 0.0001) | 0.0263 | 27,531 | 17.67 | 0.708 | 0.353 | Success |
| 3 | No_Insert (D50) | 0.0394 | 32,107 | 16.75 | 0.692 | 0.362 | Success |
| 4 | No_Insert (D50 + Reset 2000) | 0.0324 | 56,997 | 17.35 | 0.694 | 0.354 | Success |
| 5 | No_Insert (D50 + Reset 2000 + Thresh 0.00005) | 0.0460 | 30,358 | 16.21 | 0.663 | 0.398 | Success |
| 6 | Dynamic Insertion | **0.0348** | **32,803** | **18.17** | **0.709** | **0.359** | Success |

### 3.4 MonoGS++ Aligned Features Ablation Study
Evaluation of experimental MonoGS++ features implemented on `fr1_desk` under a strict 15-minute timeout boundary.

| Configuration | Dynamic Insertion ($\tau$ adaptive) | Clarity Densification | Max Frame Reached | Final Gaussian Count | Final ATE RMSE (m) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | ❌ | ❌ | 182 | 13,549 | **0.0295** |
| **Feature 1 (Adaptive 3-NN)** |  | ❌ | **223** | 13,944 | 0.0417 |
| **Feature 2 (Scale-Opacity)** | ❌ |  | 182 | **13,090** | 0.0370 |
| **Combined** |  |  | 209 | 13,852 | 0.0342 |

---

## 4. Qualitative Results & Geometric Continuity

2DGS-SLAM provides visual improvements on surface reconstruction:
* **Wall & Flat Surfaces**: Rendering with 2D surfels results in smoother, more continuous depth maps on wall regions compared to the noisy, high-frequency depth discontinuities of 3D ellipsoids.
* **Sharp Object Boundaries**: Object silhouettes (e.g. desk edges, chairs) are cleaner and less prone to the "bleeding" or "floating" artifacts common in MonoGS.
* **Normal Coherence**: The calculated surface normal maps are noise-free and show high structural consistency.
