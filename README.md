# 2DGS SLAM: 2D Gaussian Splatting for Dense SLAM

This repository implements a dense SLAM system based on **2D Gaussian Splatting (2DGS)**, extending the **MonoGS** framework. By representing the scene with oriented circular disks (surfels) instead of standard 3D ellipsoids, this system achieves more accurate surface modeling, improved tracking stability via **analytic Pose Jacobians**, and high-quality geometry reconstruction using **Normal** and **Distortion** losses.

## Key Features
- **2D Gaussian Primitives**: Oriented surfels/disks for better surface representation and fewer "cloudy" artifacts.
- **Analytic Pose Jacobians**: Custom CUDA implementation for stable and fast camera tracking.
- **Advanced Loss Functions**: Includes Normal Consistency and Distortion losses to regularize geometry.
- **Versatile Modes**: Supports Monocular and RGB-D SLAM on standard datasets like TUM RGB-D.
- **Performance**: High tracking accuracy with stable ATE RMSE across challenging sequences.

## Getting Started
### Installation

This system requires a CUDA-enabled GPU. These instructions are verified for **Ubuntu 22.04/24.04** with **CUDA 12.x**.

The recommended environment below is the one that was successfully used to reproduce Kaggle behavior locally.

#### 1. Create Environment
Use **Python 3.12** for best compatibility with modern PyTorch and CUDA 12.
```bash
conda create -n 2dgslam python=3.12.12 -y
conda activate 2dgslam
```

#### 2. Install PyTorch (CUDA 12.x)
Use the following command to install the specific PyTorch build compatible with CUDA 12.x:
```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

#### 3. Install Dependencies
Install required Python packages:
```bash
pip install "numpy==1.26.4" \
    plyfile==0.8.1 \
    munch==4.0.0 \
    trimesh==4.11.3 \
    evo==1.11.0 \
    open3d==0.19.0 \
    torchmetrics==1.5.2 \
    imgviz==1.7.5 \
    PyOpenGL==3.1.10 \
    glfw==2.10.0 \
    PyGLM==2.8.3 \
    wandb==0.24.2 \
    lpips==0.1.4 \
    rich==14.3.3 \
    ruff==0.15.5 \
    ninja==1.13.0 \
    opencv-python==4.13.0.92
```
> [!IMPORTANT]
> If `opencv-python` upgrades NumPy to 2.x, downgrade it back to `numpy==1.26.4` to avoid extension/runtime issues.

#### 4. Build Submodules
The system relies on custom CUDA kernels for the 2D surfel rasterizer and KNN.

**Note**: You must apply a small header fix to `simple-knn` if you are using CUDA 12.

```bash
# Clone with --recursive to ensure third_party/glm is initialized
# (If you already cloned, run: git submodule update --init --recursive)

# 1. Apply CUDA 12 fix to simple-knn
sed -i '1i #include <cfloat>' submodules/simple-knn/simple_knn.cu

# 2. Build diff-surfel-rasterization (Custom MonoGS version)
cd submodules/diff-surfel-rasterization
pip install . --no-build-isolation

# 3. Build simple-knn
cd ../simple-knn
pip install . --no-build-isolation

cd ../..
```

#### 5. Verify Extension Imports
Run this once after installation:
```bash
python - <<'PY'
import torch
import simple_knn._C
import diff_surfel_rasterization
print('Extension check: OK')
PY
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

## Troubleshooting

### `ModuleNotFoundError: No module named 'simple_knn._C'`
Rebuild/install the submodule in the active environment:
```bash
pip install -e submodules/simple-knn
```

### `ImportError: libc10.so: cannot open shared object file`
This usually happens when importing the extension without loading PyTorch first in a standalone probe script.

Use this check instead:
```bash
python - <<'PY'
import torch
import simple_knn._C
print('simple_knn._C loaded')
PY
```

Inside normal `slam.py` execution, PyTorch is imported early, so this specific probe-order issue does not usually occur.

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
