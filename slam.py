import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

try:
    import wandb
except ImportError:
    class _DisabledWandb:
        class Table:
            def __init__(self, *args, **kwargs):
                pass

            def add_data(self, *args, **kwargs):
                pass

        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def define_metric(*args, **kwargs):
            pass

        @staticmethod
        def finish(*args, **kwargs):
            pass

    wandb = _DisabledWandb()
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_eval_summary, save_metrics_graphs
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


def make_save_dir(config, config_path):
    mkdir_p(config["Results"]["save_dir"])
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dataset_path = config["Dataset"].get("dataset_path", config["Dataset"]["type"])
    path = [p for p in dataset_path.split("/") if p]
    dataset_name = path[-2] + "_" + path[-1] if len(path) >= 2 else "dataset"
    if len(path) == 1:
        dataset_name = path[-1]
    save_dir = os.path.join(
        config["Results"]["save_dir"], dataset_name, current_datetime
    )
    config["Results"]["save_dir"] = save_dir
    mkdir_p(save_dir)
    with open(os.path.join(save_dir, "config.yml"), "w", encoding="utf-8") as file:
        yaml.dump(config, file)
    Log("saving results in " + save_dir)
    run_name = f"{config_path.split('.')[0]}_{current_datetime}"
    return save_dir, run_name


class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.record_optimization_video = self.config["Results"].get(
            "record_optimization_video", False
        )
        self.use_visualizer = self.use_gui or self.record_optimization_video
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_visualizer else FakeQueue()
        q_vis2main = mp.Queue() if self.use_visualizer else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background.cpu()
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background.cpu()
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background.cpu(),
            gaussians=None,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            record_video=self.record_optimization_video,
            record_video_dir=self.config["Results"].get("record_video_dir"),
            record_video_fps=self.config["Results"].get("record_video_fps", 15),
            record_video_interval=self.config["Results"].get(
                "record_video_interval", 1
            ),
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_visualizer:
            # Open3D's Filament backend segfaults (SIGSEGV) when running in a
            # mp.spawn child process under native Wayland.  Force XWayland via
            # the X11 display so the GUI process survives.
            if os.environ.get("WAYLAND_DISPLAY"):
                os.environ.pop("WAYLAND_DISPLAY", None)
                os.environ["XDG_SESSION_TYPE"] = "x11"
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        N_frames = len(self.frontend.cameras)
        total_time_s = start.elapsed_time(end) * 0.001
        FPS = N_frames / total_time_s
        Log("Total time", total_time_s, tag="Eval")
        Log("Total FPS", FPS, tag="Eval")

        gaussian_counts = self.frontend.gaussian_counts
        fps_history = self.frontend.fps_history
        wall_times = self.frontend.wall_times
        export_graph = self.config["Results"].get("export_metrics_graph", True)
        if export_graph and gaussian_counts and fps_history:
            save_metrics_graphs(
                self.save_dir, gaussian_counts, fps_history, wall_times,
                mapping_losses=self.frontend.mapping_losses
            )

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            gaussian_count = self.gaussians.get_xyz.shape[0]
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
                gaussian_count=gaussian_count,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            depth_l1_cm = rendering_result.get("mean_depth_l1")
            save_eval_summary(
                self.save_dir,
                rmse_ate_m=ATE,
                total_time_s=total_time_s,
                total_fps=FPS,
                gaussian_count=gaussian_count,
                rendering_result=rendering_result,
                depth_l1_cm=depth_l1_cm,
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # # re-used the frontend queue to retrive the gaussians from the backend.
            # while not frontend_queue.empty():
            #     frontend_queue.get()
            # backend_queue.put(["color_refinement"])
            # while True:
            #     if frontend_queue.empty():
            #         time.sleep(0.01)
            #         continue
            #     data = frontend_queue.get()
            #     if data[0] == "sync_backend" and frontend_queue.empty():
            #         gaussians = data[1]
            #         self.gaussians = gaussians
            #         break
            #
            # rendering_result = eval_rendering(
            #     self.frontend.cameras,
            #     self.gaussians,
            #     self.dataset,
            #     self.save_dir,
            #     self.pipeline_params,
            #     self.background,
            #     kf_indices=kf_indices,
            #     iteration="after_opt",
            # )
            # metrics_table.add_data(
            #     "After",
            #     rendering_result["mean_psnr"],
            #     rendering_result["mean_ssim"],
            #     rendering_result["mean_lpips"],
            #     ATE,
            #     FPS,
            # )
            # wandb.log({"Metrics": metrics_table})
            # save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_visualizer:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--record-optimization-video", action="store_true")
    parser.add_argument("--record-video-fps", type=int, default=15)
    parser.add_argument("--record-video-interval", type=int, default=1)

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running 2dgslam in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=False")
        config["Results"]["use_wandb"] = False

    config["Results"]["record_optimization_video"] = args.record_optimization_video
    config["Results"]["record_video_fps"] = max(1, args.record_video_fps)
    config["Results"]["record_video_interval"] = max(1, args.record_video_interval)

    if config["Results"]["save_results"] or args.record_optimization_video:
        save_dir, run_name = make_save_dir(config, args.config)
        config["Results"]["record_video_dir"] = os.path.join(
            save_dir, "optimization_videos"
        )
        with open(os.path.join(save_dir, "config.yml"), "w", encoding="utf-8") as file:
            yaml.dump(config, file)
        run = wandb.init(
            project="2dgslam",
            name=run_name,
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
    os._exit(0)

