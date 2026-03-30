# 2DGS SLAM: 2D Gaussian Splatting for Dense SLAM

This repository implements a dense SLAM system based on **2D Gaussian Splatting (2DGS)**, extending the **MonoGS** framework. By representing the scene with oriented circular disks (surfel) instead of standard 3D ellipsoids, our system achieves more accurate surface modeling, improved tracking stability via **analytic Pose Jacobians**, and high-quality geometry reconstruction using **Normal & Distortion losses**.

## Key Features
- **2D Gaussian Primitives**: Oriented surfels/disks for better surface representation and fewer "cloudy" artifacts.
- **Analytic Pose Jacobians**: Custom CUDA implementation for stable and fast camera tracking.
- **Advanced Loss Functions**: Includes Normal Consistency and Distortion losses to regularize geometry.
- **Versatile Modes**: Supports Monocular and RGB-D SLAM on standard datasets like TUM RGB-D.
- **Performance**: High tracking accuracy with stable ATE RMSE across challenging sequences.

## Getting Started
### Installation
```bash
git clone https://github.com/phongviet/2D-Gaussian-Splatting-SLAM.git --recursive
cd MonoGS
conda env create -f environment.yml
conda activate MonoGS
cd submodules/diff-surfel-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
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
python slam.py --config configs/rgbd/tum/fr1_desk.yaml --eval
```
This will automatically:
1. Disable the GUI.
2. Run Pose and Rendering evaluation.
3. Save results in the `results` directory.

## Acknowledgments
This work is built upon:
- **MonoGS**: [Gaussian Splatting SLAM (CVPR 2024)](https://github.com/muskie82/MonoGS)
- **2DGS**: [2D Gaussian Splatting for Geometrically Accurate Radiance Fields (SIGGRAPH 2024)](https://github.com/hbb1/2d-gaussian-splatting)

## License
2DGS SLAM is released under the **LICENSE.md**.
