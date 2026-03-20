import torch
from torch import nn

from gaussian_splatting.scene.gaussian_model import GaussianModel as GaussianModel2D
from gaussian_splatting.utils.general_utils import build_rotation


class GaussianModel(GaussianModel2D):
    def build_covariance_from_scaling_rotation(
        self, center, scaling, scaling_modifier, rotation
    ):
        _ = center
        from gaussian_splatting.utils.general_utils import (
            build_scaling_rotation,
            strip_symmetric,
        )

        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        fused_point_cloud, features, scales_2d, rots, opacities = (
            super().create_pcd_from_image_and_depth(cam, rgb, depth, init)
        )
        if scales_2d.shape[-1] == 2:
            scales_3d = torch.cat([scales_2d, scales_2d[:, :1]], dim=-1)
        else:
            scales_3d = scales_2d
        return fused_point_cloud, features, scales_3d, rots, opacities

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    @staticmethod
    def from_dict(d):
        model = GaussianModel(d["max_sh_degree"], config=d["config"])
        model.active_sh_degree = d["active_sh_degree"]
        model.spatial_lr_scale = d["spatial_lr_scale"]
        model._xyz = nn.Parameter(
            d["_xyz"].to(dtype=torch.float32, device="cuda").requires_grad_(True)
        )
        model._features_dc = nn.Parameter(
            d["_features_dc"]
            .to(dtype=torch.float32, device="cuda")
            .requires_grad_(True)
        )
        model._features_rest = nn.Parameter(
            d["_features_rest"]
            .to(dtype=torch.float32, device="cuda")
            .requires_grad_(True)
        )
        model._scaling = nn.Parameter(
            d["_scaling"].to(dtype=torch.float32, device="cuda").requires_grad_(True)
        )
        model._rotation = nn.Parameter(
            d["_rotation"].to(dtype=torch.float32, device="cuda").requires_grad_(True)
        )
        model._opacity = nn.Parameter(
            d["_opacity"].to(dtype=torch.float32, device="cuda").requires_grad_(True)
        )
        n_points = model._xyz.shape[0]
        if "max_radii2D" in d:
            model.max_radii2D = d["max_radii2D"].to(dtype=torch.float32, device="cuda")
        else:
            model.max_radii2D = torch.zeros((n_points), device="cuda")
        if "xyz_gradient_accum" in d:
            model.xyz_gradient_accum = d["xyz_gradient_accum"].to(
                dtype=torch.float32, device="cuda"
            )
        else:
            model.xyz_gradient_accum = torch.zeros((n_points, 1), device="cuda")
        model.unique_kfIDs = d["unique_kfIDs"]
        model.n_obs = d["n_obs"]
        return model
