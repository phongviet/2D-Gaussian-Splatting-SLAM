import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def fixed_traj_colormap(ax, traj, array, plot_mode, min_map, max_map, title=""):
    """
    Fixed version of evo.tools.plot.traj_colormap that explicitly passes 'ax' to colorbar.
    This avoids the ValueError in Matplotlib 3.8+.
    """
    import matplotlib as mpl
    from evo.tools.plot import colored_line_collection, PlotMode
    
    pos = traj.positions_xyz
    norm = mpl.colors.Normalize(vmin=min_map, vmax=max_map, clip=True)
    mapper = cm.ScalarMappable(
        norm=norm,
        cmap=SETTINGS.plot_trajectory_cmap)
    mapper.set_array(array)
    colors = [mapper.to_rgba(a) for a in array]
    line_collection = colored_line_collection(pos, colors, plot_mode)
    ax.add_collection(line_collection)
    ax.autoscale_view(True, True, True)
    if plot_mode == PlotMode.xyz:
        ax.set_zlim(
            np.amin(traj.positions_xyz[:, 2]),
            np.amax(traj.positions_xyz[:, 2]))
    
    fig = ax.get_figure()
    # Explicitly pass ax=ax to colorbar
    cbar = fig.colorbar(
        mapper, ax=ax, ticks=[min_map, (max_map - (max_map - min_map) / 2), max_map])
    cbar.ax.set_yticklabels([
        "{0:0.3f}".format(min_map),
        "{0:0.3f}".format(max_map - (max_map - min_map) / 2),
        "{0:0.3f}".format(max_map)
    ])
    if title:
        ax.legend(frameon=True)
        plt.title(title)


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False, gaussian_count=None, total_fps=None):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    if gaussian_count is not None and total_fps is not None:
        Log(f"RMSE ATE  ({gaussian_count} gaussians, {total_fps:.3f} fps)", ape_stat, tag="Eval")
    elif gaussian_count is not None:
        Log(f"RMSE ATE [m] ({gaussian_count} gaussians)", ape_stat, tag="Eval")
    else:
        Log("RMSE ATE [m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    fixed_traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False, gaussian_count=None, total_fps=None):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
        gaussian_count=gaussian_count,
        total_fps=total_fps,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array, depth_l1_array = [], [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, gt_depth, _ = dataset[idx]


        render_result = render(frame, gaussians, pipe, background)
        rendering = render_result["render"]
        depth = render_result["depth"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        gt_d = None
        if gt_depth is not None:
            if hasattr(gt_depth, "cpu"):
                gt_d = gt_depth.detach().cpu().numpy()
            else:
                gt_d = np.array(gt_depth)
            
            if gt_d.ndim == 3:
                gt_d = gt_d[0]
            elif gt_d.ndim > 3:
                gt_d = gt_d.squeeze()
        if gt_d is not None:
            pred_d = depth[0, :, :].detach().cpu().numpy()
            valid_mask = gt_d > 0.01

            if valid_mask.sum() > 0:
                depth_l1 = np.abs(pred_d[valid_mask] - gt_d[valid_mask]).mean()
                depth_l1_array.append(depth_l1)

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_depth_l1"] = float(np.mean(depth_l1_array)) if depth_l1_array else None

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, depth_l1_cm: {output["mean_depth_l1"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_eval_summary(
    save_dir,
    rmse_ate_m,
    total_time_s,
    total_fps,
    gaussian_count,
    rendering_result,
    depth_l1_cm=None,
):
    if save_dir is None:
        return

    output = {
        "rmse_ate_m": float(rmse_ate_m),
        "gaussian_count": int(gaussian_count),
        "total_time_s": float(total_time_s),
        "total_fps": float(total_fps),
        "mean_psnr": float(rendering_result["mean_psnr"]),
        "mean_ssim": float(rendering_result["mean_ssim"]),
        "mean_lpips": float(rendering_result["mean_lpips"]),
    }
    if depth_l1_cm is not None:
        output["mean_depth_l1_cm"] = float(depth_l1_cm)

    eval_summary_path = os.path.join(save_dir, "eval_summary.json")
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    Log(f"Saved eval summary to {eval_summary_path}", tag="Eval")


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


def save_metrics_graphs(save_dir, gaussian_counts, fps_history, wall_times=None):
    """
    Save Gaussian count and FPS time-series as JSON and PNG graph.
    """
    if save_dir is None or not gaussian_counts or not fps_history:
        return

    frame_indices = list(range(len(gaussian_counts)))
    wall_times_s = wall_times if wall_times is not None else []

    # Save raw time-series JSON
    time_series_data = {
        "frame_indices": frame_indices,
        "gaussian_counts": [int(x) for x in gaussian_counts],
        "fps": [round(x, 4) for x in fps_history],
        "wall_times_s": [round(x, 4) for x in wall_times_s],
    }
    time_series_path = os.path.join(save_dir, "metrics_time_series.json")
    with open(time_series_path, "w", encoding="utf-8") as f:
        json.dump(time_series_data, f, indent=4)
    Log(f"Saved metrics time-series to {time_series_path}", tag="Eval")

    # Generate PNG graph with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(frame_indices, gaussian_counts, color="#2196F3", linewidth=1.5)
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Gaussian Count")
    ax1.set_title("Gaussian Count over Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(frame_indices, fps_history, color="#4CAF50", linewidth=1.5)
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("FPS")
    ax2.set_title("FPS over Time")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    graph_path = os.path.join(save_dir, "metrics_graph.png")
    plt.savefig(graph_path, dpi=150)
    plt.close(fig)
    Log(f"Saved metrics graph to {graph_path}", tag="Eval")
