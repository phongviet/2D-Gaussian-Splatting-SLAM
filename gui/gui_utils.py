import queue

import cv2
import numpy as np
import open3d as o3d
import torch

from gaussian_splatting.utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
)
from utils.camera_utils import Camera

cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up


def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum


class GaussianPacket:
    def __init__(
        self,
        gaussians=None,
        keyframe=None,
        current_frame=None,
        gtcolor=None,
        gtdepth=None,
        gtnormal=None,
        keyframes=None,
        finish=False,
        kf_window=None,
    ):
        self.has_gaussians = False
        if gaussians is not None:
            self.has_gaussians = True
            # Store all Gaussian tensors on CPU so the packet can safely cross
            # process boundaries via mp.Queue without CUDA IPC.
            self.get_xyz = gaussians.get_xyz.detach().clone().cpu()
            self.active_sh_degree = gaussians.active_sh_degree
            self.get_opacity = gaussians.get_opacity.detach().clone().cpu()
            self.get_scaling = gaussians.get_scaling.detach().clone().cpu()
            self.get_rotation = gaussians.get_rotation.detach().clone().cpu()
            self.max_sh_degree = gaussians.max_sh_degree
            self.get_features = gaussians.get_features.detach().clone().cpu()

            self._rotation = gaussians._rotation.detach().clone().cpu()
            self.rotation_activation = torch.nn.functional.normalize
            self.unique_kfIDs = gaussians.unique_kfIDs.clone().cpu()
            self.n_obs = gaussians.n_obs.clone().cpu()

        # Serialize Camera objects to plain CPU dicts so they cross the process
        # boundary safely.  to_cuda() will reconstruct them on the other side.
        self.keyframe = keyframe.to_dict() if keyframe is not None else None
        self.current_frame = (
            current_frame.to_dict() if current_frame is not None else None
        )
        self.keyframes = (
            [kf.to_dict() for kf in keyframes] if keyframes is not None else None
        )

        self.gtcolor = self.resize_img(gtcolor, 320)
        self.gtdepth = self.resize_img(gtdepth, 320)
        self.gtnormal = self.resize_img(gtnormal, 320)

        # Move gtcolor / gtdepth tensors to CPU if they are tensors
        if self.gtcolor is not None and isinstance(self.gtcolor, torch.Tensor):
            self.gtcolor = self.gtcolor.cpu()
        if self.gtdepth is not None and isinstance(self.gtdepth, torch.Tensor):
            self.gtdepth = self.gtdepth.cpu()
        if self.gtnormal is not None and isinstance(self.gtnormal, torch.Tensor):
            self.gtnormal = self.gtnormal.cpu()

        self.finish = finish
        self.kf_window = kf_window

    def to_cuda(self):
        """Move all tensors to CUDA and reconstruct Camera objects from dicts.

        Must be called in the receiving (GUI) process after getting this packet
        from the queue.
        """
        if self.has_gaussians:
            self.get_xyz = self.get_xyz.cuda()
            self.get_opacity = self.get_opacity.cuda()
            self.get_scaling = self.get_scaling.cuda()
            self.get_rotation = self.get_rotation.cuda()
            self.get_features = self.get_features.cuda()
            self._rotation = self._rotation.cuda()
            self.unique_kfIDs = self.unique_kfIDs.cuda()
            self.n_obs = self.n_obs.cuda()

        if self.current_frame is not None:
            self.current_frame = Camera.from_dict(self.current_frame)
        if self.keyframe is not None:
            self.keyframe = Camera.from_dict(self.keyframe)
        if self.keyframes is not None:
            self.keyframes = [Camera.from_dict(kf) for kf in self.keyframes]

        return self

    def resize_img(self, img, width):
        if img is None:
            return None

        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (width, height))
        height = int(width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
        return img.squeeze(0)

    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        if scaling.shape[-1] == 2:
            scaling = torch.cat([scaling, torch.ones_like(scaling[:, :1])], dim=-1)
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm


def get_latest_queue(q):
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.qsize() < 1:
                break
    return message


class Packet_vis2main:
    flag_pause = None


class ParamsGUI:
    def __init__(
        self,
        pipe=None,
        background=None,
        gaussians=None,
        q_main2vis=None,
        q_vis2main=None,
        renderer_mode="2dgs",
    ):
        self.pipe = pipe
        self.background = background
        self.gaussians = gaussians
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.renderer_mode = renderer_mode
