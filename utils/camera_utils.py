import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None

    def to_dict(self):
        """Serialize Camera to a CPU-only dict for safe cross-process queue transfer."""
        return {
            "uid": self.uid,
            "device": self.device,
            "R": self.R.detach().cpu(),
            "T": self.T.detach().cpu(),
            "R_gt": self.R_gt.detach().cpu(),
            "T_gt": self.T_gt.detach().cpu(),
            "original_image": self.original_image.detach().cpu()
            if self.original_image is not None
            else None,
            "depth": self.depth,
            "grad_mask": self.grad_mask.detach().cpu()
            if self.grad_mask is not None
            else None,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "FoVx": self.FoVx,
            "FoVy": self.FoVy,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "cam_rot_delta": self.cam_rot_delta.data.detach().cpu()
            if self.cam_rot_delta is not None
            else None,
            "cam_trans_delta": self.cam_trans_delta.data.detach().cpu()
            if self.cam_trans_delta is not None
            else None,
            "exposure_a": self.exposure_a.data.detach().cpu()
            if self.exposure_a is not None
            else None,
            "exposure_b": self.exposure_b.data.detach().cpu()
            if self.exposure_b is not None
            else None,
            "projection_matrix": self.projection_matrix.detach().cpu(),
        }

    @staticmethod
    def from_dict(d):
        """Reconstruct a Camera from a CPU-only dict produced by to_dict()."""
        device = d["device"]
        # Build a minimal gt_T from the stored R_gt/T_gt
        gt_T = torch.eye(4)
        gt_T[:3, :3] = d["R_gt"]
        gt_T[:3, 3] = d["T_gt"]
        cam = Camera(
            uid=d["uid"],
            color=d["original_image"].to(device)
            if d["original_image"] is not None
            else None,
            depth=d["depth"],
            gt_T=gt_T.to(device),
            projection_matrix=d["projection_matrix"].to(device),
            fx=d["fx"],
            fy=d["fy"],
            cx=d["cx"],
            cy=d["cy"],
            fovx=d["FoVx"],
            fovy=d["FoVy"],
            image_height=d["image_height"],
            image_width=d["image_width"],
            device=device,
        )
        # Restore pose (may differ from gt after tracking)
        cam.R = d["R"].to(device)
        cam.T = d["T"].to(device)
        # Restore grad_mask
        cam.grad_mask = (
            d["grad_mask"].to(device) if d["grad_mask"] is not None else None
        )
        # Restore optimizable parameters
        if d["cam_rot_delta"] is not None:
            cam.cam_rot_delta = nn.Parameter(
                d["cam_rot_delta"].to(device), requires_grad=True
            )
        if d["cam_trans_delta"] is not None:
            cam.cam_trans_delta = nn.Parameter(
                d["cam_trans_delta"].to(device), requires_grad=True
            )
        if d["exposure_a"] is not None:
            cam.exposure_a = nn.Parameter(
                d["exposure_a"].to(device), requires_grad=True
            )
        if d["exposure_b"] is not None:
            cam.exposure_b = nn.Parameter(
                d["exposure_b"].to(device), requires_grad=True
            )
        return cam
