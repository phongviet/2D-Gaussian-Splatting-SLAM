import math

import torch
from diff_surfel_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh

_rays_d_cam_cache = {}


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height

    cache_key = (W, H, view.FoVx, view.FoVy)
    if cache_key not in _rays_d_cam_cache:
        ndc2pix = torch.tensor(
            [[W / 2, 0, 0, W / 2], [0, H / 2, 0, H / 2], [0, 0, 0, 1]],
            dtype=torch.float32,
            device="cuda",
        ).T
        projection_matrix = c2w.T @ view.full_proj_transform
        intrins = (projection_matrix @ ndc2pix)[:3, :3].T

        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device="cuda", dtype=torch.float32),
            torch.arange(H, device="cuda", dtype=torch.float32),
            indexing="xy",
        )
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
            -1, 3
        )
        rays_d_cam = points @ intrins.inverse().T
        _rays_d_cam_cache[cache_key] = rays_d_cam

    rays_d_cam = _rays_d_cam_cache[cache_key]
    rays_d = rays_d_cam @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view, depth):
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if mask is not None:
        rendered_image, radii, allmap = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask] if shs is not None else None,
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask] if scales is not None else None,
            rotations=rotations[mask] if rotations is not None else None,
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        rendered_image, radii, allmap = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    render_alpha = allmap[1:2]
    render_depth_expected = allmap[0:1]
    render_alpha_safe = torch.where(
        render_alpha > 0, render_alpha, torch.ones_like(render_alpha)
    )
    render_depth_expected = render_depth_expected / render_alpha_safe
    render_depth_expected = torch.where(
        render_alpha > 0, render_depth_expected, torch.zeros_like(render_depth_expected)
    )

    render_normal = allmap[2:5]
    render_normal = (
        render_normal.permute(1, 2, 0)
        @ (viewpoint_camera.world_view_transform[:3, :3].T)
    ).permute(2, 0, 1)

    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    render_dist = allmap[6:7]

    depth_ratio = getattr(pipe, "depth_ratio", 1.0)
    surf_depth = render_depth_expected * (1 - depth_ratio) + depth_ratio * render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2, 0, 1)

    with torch.no_grad():
        n_touched = torch.zeros_like(radii, dtype=torch.int32)
        valid_mask = radii > 0

        pts_3d = means3D[valid_mask]
        pts_4d = torch.cat([pts_3d, torch.ones_like(pts_3d[:, :1])], dim=-1)

        pts_view = pts_4d @ viewpoint_camera.world_view_transform
        pts_depth = pts_view[..., 2]

        pts_clip = pts_4d @ viewpoint_camera.full_proj_transform
        pts_ndc = pts_clip[..., :3] / (pts_clip[..., 3:4] + 1e-6)

        H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
        u = ((pts_ndc[..., 0] + 1) * W - 1) * 0.5
        v = ((pts_ndc[..., 1] + 1) * H - 1) * 0.5

        u = torch.round(u).long()
        v = torch.round(v).long()

        in_screen = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (pts_depth > 0)

        u_valid = u[in_screen]
        v_valid = v[in_screen]
        depth_valid = pts_depth[in_screen]

        rendered_depth_sampled = render_depth_expected[0, v_valid, u_valid]

        is_visible = depth_valid <= (rendered_depth_sampled * 1.05)

        global_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        visible_in_screen_indices = torch.nonzero(in_screen, as_tuple=True)[0][is_visible]
        visible_global_indices = global_indices[visible_in_screen_indices]

        n_touched[visible_global_indices] = 1

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": render_depth_expected,
        "opacity": render_alpha,
        "n_touched": n_touched,
        "rend_normal": render_normal,
        "surf_normal": surf_normal,
        "rend_dist": render_dist,
        "rend_alpha": render_alpha,
    }
