import torch
import torch.nn.functional as F


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False, render_pkg=None):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint, render_pkg=render_pkg)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint, render_pkg=render_pkg)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, render_pkg=None):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    loss = l1.mean()

    return loss


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False, render_pkg=None
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint, render_pkg=render_pkg)
    depth_mask = depth_pixel_mask * opacity_mask
    
    # Use Median (Unbiased) depth for tracking if available
    depth_to_use = render_pkg["rend_median_depth"] if render_pkg is not None and "rend_median_depth" in render_pkg else depth
    l1_depth = torch.abs(depth_to_use * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False, render_pkg=None, iteration=0, total_iterations=1, render_scale=1.0):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint, render_pkg=render_pkg, iteration=iteration, total_iterations=total_iterations, render_scale=render_scale)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint, render_pkg=render_pkg, iteration=iteration, total_iterations=total_iterations, render_scale=render_scale)


def get_loss_mapping_rgb(config, image, depth, viewpoint, render_pkg=None, iteration=0, total_iterations=1, render_scale=1.0):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    if render_scale > 1.0:
        gt_image = F.interpolate(gt_image.unsqueeze(0), scale_factor=1.0/render_scale,
                                 mode="bilinear", recompute_scale_factor=True, antialias=True)[0]
        image = F.interpolate(image.unsqueeze(0), scale_factor=1.0/render_scale,
                              mode="bilinear", recompute_scale_factor=True, antialias=True)[0]
    mask_shape = (1, gt_image.shape[1], gt_image.shape[2])
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    loss = l1_rgb.mean()

    if render_pkg is not None:
        if "lambda_normal" in config["Training"] and config["Training"]["lambda_normal"] > 0:
            if iteration > total_iterations / 4:
                rend_normal = render_pkg["rend_normal"]
                surf_depth = render_pkg["depth"]
                surf_normal = depth_to_normal(viewpoint, surf_depth)
                
                # Transform View Space normal to World Space
                R_wc = viewpoint.world_view_transform[:3, :3].T
                rend_normal = (rend_normal.permute(1, 2, 0) @ R_wc).permute(2, 0, 1)
                
                surf_normal = surf_normal.permute(2, 0, 1)
                loss_normal = (1 - torch.sum(rend_normal * surf_normal, dim=0)).mean()
                loss += config["Training"]["lambda_normal"] * loss_normal

        if "lambda_distortion" in config["Training"] and config["Training"]["lambda_distortion"] > 0:
            if iteration > total_iterations / 10:
                rend_dist = render_pkg["rend_dist"]
                loss += config["Training"]["lambda_distortion"] * rend_dist.mean()

    return loss


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False, render_pkg=None, iteration=0, total_iterations=1, render_scale=1.0):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()
    if render_scale > 1.0:
        gt_image = F.interpolate(gt_image.unsqueeze(0), scale_factor=1.0/render_scale,
                                 mode="bilinear", recompute_scale_factor=True, antialias=True)[0]
        image = F.interpolate(image.unsqueeze(0), scale_factor=1.0/render_scale,
                             mode="bilinear", recompute_scale_factor=True, antialias=True)[0]

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    if render_scale > 1.0:
        gt_depth = F.interpolate(gt_depth.unsqueeze(0).float(), scale_factor=1.0/render_scale,
                                mode="nearest")[0]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    
    # Use Median (Unbiased) depth for mapping
    depth_to_use = render_pkg["rend_median_depth"] if render_pkg is not None and "rend_median_depth" in render_pkg else depth
    l1_depth = torch.abs(depth_to_use * depth_pixel_mask - gt_depth * depth_pixel_mask)

    loss = alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

    if render_pkg is not None:
        if "lambda_normal" in config["Training"] and config["Training"]["lambda_normal"] > 0:
            rend_normal = render_pkg["rend_normal"]
            
            # Use Ground Truth depth for normal reference to provide a stable geometric anchor
            surf_normal = depth_to_normal(viewpoint, gt_depth)
            
            # Transform View Space normal to World Space
            R_wc = viewpoint.world_view_transform[:3, :3].T
            rend_normal = (rend_normal.permute(1, 2, 0) @ R_wc).permute(2, 0, 1)
            
            surf_normal = surf_normal.permute(2, 0, 1)
            loss_normal = (1 - torch.sum(rend_normal * surf_normal, dim=0)).mean()
            loss += config["Training"]["lambda_normal"] * loss_normal

        if "lambda_distortion" in config["Training"] and config["Training"]["lambda_distortion"] > 0:
            rend_dist = render_pkg["rend_dist"]
            loss += config["Training"]["lambda_distortion"] * rend_dist.mean()

    return loss


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    # 2dgslam Camera might have different projection matrix structure.
    # In 2dgslam Camera: self.projection_matrix = getProjectionMatrix2(...)
    # Let's use the same logic as 2DGS but adapt it to 2dgslam Camera.
    
    # In 2DGS, ndc2pix is used.
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    
    # In 2dgslam viewpoint (Camera), full_proj_transform is world2pix-ish?
    # No, it's projection @ world2view.
    projection_matrix = c2w.T @ viewpoint_full_proj_transform(view)
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    # rays_d = points @ K^-1 @ R^T
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def viewpoint_full_proj_transform(view):
    # 2dgslam Camera object has full_proj_transform attribute (or result of update_RT)
    return view.full_proj_transform


def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
