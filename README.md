# 2dgslam: 2D Gaussian Splatting for Dense SLAM

This repository implements a dense SLAM system based on **2D Gaussian Splatting (2DGS)**, extending the **MonoGS** framework. By representing the scene with oriented circular disks (surfels) instead of standard 3D ellipsoids, **2dgslam** achieves more accurate surface modeling, improved tracking stability via **analytic Pose Jacobians**, and high-quality geometry reconstruction using **Normal** and **Distortion** losses.

## Key Features
- **2D Gaussian Primitives**: Oriented surfels/disks for better surface representation and fewer "cloudy" artifacts.
- **Analytic Pose Jacobians**: Custom CUDA implementation for stable and fast camera tracking.
- **Advanced Loss Functions**: Includes Normal Consistency and Distortion losses to regularize geometry.
- **Performance**: High tracking accuracy with stable ATE RMSE across challenging sequences.

## Getting Started
### Installation

This system requires a CUDA-enabled GPU. These instructions are verified for **Ubuntu 22.04/24.04** with **CUDA 12.x**.

The recommended environment below is the one that was successfully used to reproduce Kaggle behavior locally.

#### 1. Setup Environment
Use **Python 3.12** for best compatibility with modern PyTorch and CUDA 12. You can create the environment using the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate 2dgslam
```

Alternatively, if you prefer using `pip` directly:
```bash
conda create -n 2dgslam python=3.12.12 -y
conda activate 2dgslam
pip install -r requirements.txt
```

#### 4. Build Submodules
The system relies on custom CUDA kernels for the 2D surfel rasterizer and KNN.



```bash
# Clone with --recursive
# (If you already cloned, run: git submodule update --init --recursive)

# 1. Apply CUDA 12 fix to simple-knn
sed -i '1i #include <cfloat>' submodules/simple-knn/simple_knn.cu

# 2. Build diff-surfel-rasterization (Custom 2dgslam version)
cd submodules/diff-surfel-rasterization
pip install . --no-build-isolation

# 3. Build simple-knn
cd ../simple-knn
pip install . --no-build-isolation

cd ../..
```

### Quick Run (TUM RGB-D)
Download a TUM sequence:
```bash
bash scripts/download_tum.sh
```
Run the system with a specific configuration:
```bash
python slam.py --config configs/rgbd/tum/fr1_desk.yaml
```

## Evaluation
To run the system in headless mode and log metrics (ATE and rendering quality):
```bash
python slam.py --config configs/mono/replica/office0.yaml --eval
```

### Quantitative Results (Monocular)

| Dataset | Sequence | RMSE ATE (m) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **TUM** | `fr1_desk` | 0.0381 | 17.66 | 0.672 | 0.365 |
| | `fr2_xyz` | 0.0443 | 15.67 | 0.677 | 0.315 |
| | `fr3_office` | 0.0507 | 18.66 | 0.737 | 0.315 |
| **Replica** | `office0` | 0.0909 | 30.25 | 0.902 | 0.225 |
| | `office1` | 0.1665 | 32.53 | 0.920 | 0.184 |
| | `office2` | 0.1026 | 27.27 | 0.894 | 0.216 |
| | `office3` | 0.0310 | 30.30 | 0.912 | 0.146 |
| | `office4` | 0.0478 | 30.04 | 0.915 | 0.178 |
| | `room0` | 0.0612 | 27.31 | 0.857 | 0.167 |
| | `room1` | 0.4098 | 24.03 | 0.797 | 0.307 |
| | `room2` | 0.0512 | 28.43 | 0.886 | 0.200 |

## Acknowledgments
This work is built upon:
- **MonoGS**: [Gaussian Splatting SLAM (CVPR 2024)](https://github.com/muskie82/MonoGS)
- **2DGS**: [2D Gaussian Splatting for Geometrically Accurate Radiance Fields (SIGGRAPH 2024)](https://github.com/hbb1/2d-gaussian-splatting)

## License
2DGS SLAM is released under the **LICENSE.md**.
