import cv2
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def resize_image(path, target_size, is_depth=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    resized = cv2.resize(img, target_size, interpolation=interp)
    cv2.imwrite(path, resized)

def main():
    rgb_dir = "/home/2DGS_SLAM/2dgslam/datasets/tum/B1-802/rgb"
    depth_dir = "/home/2DGS_SLAM/2dgslam/datasets/tum/B1-802/depth"
    target_size = (640, 480)

    rgb_files = glob.glob(os.path.join(rgb_dir, "*.png"))
    depth_files = glob.glob(os.path.join(depth_dir, "*.png"))

    print(f"Resizing {len(rgb_files)} RGB files to {target_size}...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(lambda p: resize_image(p, target_size, False), rgb_files), total=len(rgb_files)))

    print(f"Resizing {len(depth_files)} depth files to {target_size}...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(lambda p: resize_image(p, target_size, True), depth_files), total=len(depth_files)))

    print("Resizing complete.")

if __name__ == "__main__":
    main()
