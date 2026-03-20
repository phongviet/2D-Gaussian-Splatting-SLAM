import os
import queue
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.renderer_utils import resolve_renderer_mode
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None):
        start_time = time.perf_counter()

        self.config = config
        self.renderer_mode = self.config["Training"].get("renderer", "2dgs")
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
        if self.use_gui and not self.live_mode:
            display = os.environ.get("DISPLAY")
            wayland_display = os.environ.get("WAYLAND_DISPLAY")
            if not display and not wayland_display:
                Log("GUI disabled: no display detected")
                self.use_gui = False

        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = None
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32)

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

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

        self.backend.gaussians = None
        self.backend.background = self.background.cpu()
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()
        self.backend.init_sh_degree = model_params.sh_degree

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background.cpu(),
            gaussians=None,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            renderer_mode=self.renderer_mode,
        )

        backend_process = mp.Process(target=self.backend.run)
        backend_process.start()
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(2)
        self.frontend.run()
        backend_queue.put(["pause"])

        # empty the frontend queue
        elapsed = max(time.perf_counter() - start_time, 1e-6)
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / elapsed
        Log("Total time", elapsed, tag="Eval")
        Log("Total FPS", FPS, tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
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
                renderer_mode=self.renderer_mode,
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

            # re-used the frontend queue to retrive the gaussians from the backend.
            while True:
                try:
                    frontend_queue.get_nowait()
                except queue.Empty:
                    break
            #backend_queue.put(["color_refinement"]) #no need
            while True:
                try:
                    data = frontend_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if data[0] == "sync_backend":
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

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
            #save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
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
    parser.add_argument(
        "--renderer",
        type=str,
        choices=["2dgs", "3dgs"],
        default="2dgs",
    )

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    config = load_config(args.config)
    resolve_renderer_mode(config, args.renderer)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
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
