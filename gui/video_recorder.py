import os

import cv2
import imgviz
import numpy as np
import torch


def _to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def rgb_frame_to_uint8(image):
    image = _to_numpy(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(image.astype(np.uint8))


def depth_range_from_gt(gt_depth):
    gt_depth = _to_numpy(gt_depth)
    gt_depth = np.squeeze(gt_depth)
    gt_depth = np.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid_depth = gt_depth[gt_depth > 0]
    if valid_depth.size == 0:
        return None, None
    min_value = float(np.percentile(valid_depth, 1))
    max_value = float(np.percentile(valid_depth, 99))
    if max_value <= min_value:
        max_value = min_value + 1.0
    return min_value, max_value


def depth_frame_to_uint8(depth, min_value=None, max_value=None):
    depth = _to_numpy(depth)
    depth = np.squeeze(depth)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if min_value is None or max_value is None:
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        min_value = float(valid_depth.min()) if valid_depth.size > 0 else 0.0
        max_value = float(valid_depth.max()) if valid_depth.size > 0 else 1.0
    if max_value <= min_value:
        max_value = min_value + 1.0
    depth_color = imgviz.depth2rgb(
        depth, min_value=min_value, max_value=max_value, colormap="jet"
    ).astype(np.uint8)
    depth_color[depth <= 0] = 0
    return np.ascontiguousarray(depth_color)


def normal_frame_to_uint8(normal):
    normal = _to_numpy(normal)
    if normal.ndim == 3 and normal.shape[0] == 3:
        normal = np.transpose(normal, (1, 2, 0))
    normal = np.nan_to_num(normal, nan=0.0, posinf=1.0, neginf=-1.0)
    normal = np.clip(normal, -1.0, 1.0)
    return np.ascontiguousarray(((normal * 0.5 + 0.5) * 255.0).astype(np.uint8))


class OptimizationVideoRecorder:
    def __init__(self, output_dir, fps=15):
        if output_dir is None:
            raise ValueError("Optimization video output directory is not configured")
        self.output_dir = output_dir
        self.fps = max(1, int(fps))
        self.writers = {}
        os.makedirs(output_dir, exist_ok=True)

    def write(self, stream_name, frame_rgb):
        frame_rgb = rgb_frame_to_uint8(frame_rgb)
        height, width = frame_rgb.shape[:2]
        if stream_name not in self.writers:
            path = os.path.join(self.output_dir, f"{stream_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, self.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {path}")
            self.writers[stream_name] = writer
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.writers[stream_name].write(frame_bgr)

    def close(self):
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
