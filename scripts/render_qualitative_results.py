#!/usr/bin/env python3
"""Render and compose the qualitative thesis comparisons.

The 2DGSLAM and MonoGS renderers are launched in separate Python processes so
their CUDA extensions and identically named packages never share an interpreter.
The compositor then verifies the ground-truth inputs, computes per-frame depth
errors, and exports publication-ready PDF figures with 300-dpi PNG previews.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_TUM_STEM = "ket_qua_dinh_tinh_tum_fr3"
FIGURE_REPLICA_STEM = "ket_qua_dinh_tinh_replica"
METRICS_NAME = "ket_qua_dinh_tinh_metrics.json"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="Render matched 2DGSLAM/MonoGS frames and build thesis figures."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "qualitative_manifest.json",
        help="Pinned scene/checkpoint manifest.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=workspace_root,
        help="Workspace containing 2dgslam and Original_MonoGS_Code.",
    )
    parser.add_argument(
        "--ours-python",
        type=Path,
        required=True,
        help="Python interpreter from the 2DGSLAM environment.",
    )
    parser.add_argument(
        "--monogs-python",
        type=Path,
        required=True,
        help="Python interpreter from the MonoGS environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for final PDF/PNG assets and the metrics record.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory for compressed intermediate render arrays.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse existing intermediate arrays instead of rerendering them.",
    )
    parser.add_argument("--preview-dpi", type=int, default=300)
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema_version')}")
    scenes = manifest.get("scenes", [])
    if len(scenes) != 3:
        raise ValueError("The qualitative figure manifest must contain exactly three scenes.")
    return manifest


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def validate_result_dir(result_dir: Path) -> None:
    require_file(result_dir / "config.yml", "saved run configuration")
    require_file(result_dir / "plot" / "trj_final.json", "saved trajectory")
    require_file(
        result_dir / "point_cloud" / "final" / "point_cloud.ply",
        "final Gaussian map",
    )


def run_dump(
    interpreter: Path,
    dump_script: Path,
    repo: Path,
    result_dir: Path,
    frame: int,
    output_path: Path,
) -> str:
    command = [
        str(interpreter),
        str(dump_script),
        "--repo",
        str(repo),
        "--res",
        str(result_dir),
        "--frame",
        str(frame),
        "--out",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def load_render(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.array(data[key]) for key in data.files}


def verify_ground_truth(ours: dict, monogs: dict, scene_id: str) -> None:
    for key in ("gt_rgb", "gt_depth"):
        if ours[key].shape != monogs[key].shape:
            raise ValueError(
                f"{scene_id}: {key} shape mismatch: "
                f"{ours[key].shape} vs {monogs[key].shape}"
            )
        if not np.allclose(ours[key], monogs[key], rtol=1e-5, atol=1e-5):
            maximum = float(np.nanmax(np.abs(ours[key] - monogs[key])))
            raise ValueError(
                f"{scene_id}: renderer ground truth differs (max |delta|={maximum:.3g})."
            )


def depth_statistics(gt_depth: np.ndarray, prediction: np.ndarray) -> dict:
    gt = np.asarray(gt_depth, dtype=np.float64).squeeze()
    pred = np.asarray(prediction, dtype=np.float64).squeeze()
    gt_valid = np.isfinite(gt) & (gt > 0)
    pred_valid = np.isfinite(pred) & (pred > 0)
    valid = gt_valid & pred_valid
    if not np.any(valid):
        raise ValueError("No valid pixels remain for the displayed-frame depth error.")
    error = np.full_like(gt, np.nan, dtype=np.float64)
    error[valid] = np.abs(pred[valid] - gt[valid])
    return {
        "mean_absolute_error_m": float(np.mean(error[valid])),
        "coverage": float(np.count_nonzero(valid) / np.count_nonzero(gt_valid)),
        "error": error,
        "valid": valid,
    }


def robust_range(values: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot derive a color scale from an empty array.")
    lower, upper = np.percentile(finite, [low, high])
    if math.isclose(float(lower), float(upper)):
        upper = lower + 1e-6
    return float(lower), float(upper)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#333333")


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def build_tum_figure(scene: dict, output_dir: Path, preview_dpi: int) -> None:
    ours, monogs = scene["ours"], scene["monogs"]
    gt_depth = ours["gt_depth"].squeeze()
    depth_min, depth_max = scene["depth_range_m"]
    figure, axes = plt.subplots(3, 3, figsize=(6.9, 4.9), constrained_layout=True)

    rows = [
        ("Ground Truth", ours["gt_rgb"], gt_depth, None),
        ("MonoGS", monogs["rgb"], monogs["depth"], monogs["normal"]),
        ("Phương pháp đề xuất", ours["rgb"], ours["depth"], ours["normal"]),
    ]
    titles = ["Ảnh RGB", "Độ sâu", "Pháp tuyến"]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontweight="bold")

    for row, (label, rgb, depth, normal) in enumerate(rows):
        axes[row, 0].imshow(np.clip(rgb, 0.0, 1.0))
        axes[row, 1].imshow(depth, cmap="viridis", vmin=depth_min, vmax=depth_max)
        if normal is None:
            axes[row, 2].set_axis_off()
        else:
            axes[row, 2].imshow(np.clip(normal, 0.0, 1.0))
        axes[row, 0].set_ylabel(
            label,
            fontweight="bold" if row == 2 else "normal",
            color="#1f4e79" if row == 2 else "black",
        )
        for column in range(3):
            if normal is not None or column != 2:
                clean_axis(axes[row, column])
    save_figure(figure, output_dir, FIGURE_TUM_STEM, preview_dpi)


def build_replica_figure(scenes: list[dict], output_dir: Path, preview_dpi: int) -> None:
    figure = plt.figure(figsize=(6.9, 8.0), constrained_layout=True)
    subfigures = figure.subfigures(2, 1)
    row_labels = ("Ground Truth", "MonoGS", "Phương pháp đề xuất")
    titles = ("Ảnh RGB", "Độ sâu", "Pháp tuyến")

    for subfigure, scene in zip(subfigures, scenes):
        ours, monogs = scene["ours"], scene["monogs"]
        depth_min, depth_max = scene["depth_range_m"]
        short_label = scene["label"].replace("Replica ", "")
        axes = subfigure.subplots(3, 3)
        subfigure.suptitle(short_label, fontweight="bold")

        rows = [
            (ours["gt_rgb"], ours["gt_depth"].squeeze(), None),
            (monogs["rgb"], monogs["depth"], monogs["normal"]),
            (ours["rgb"], ours["depth"], ours["normal"]),
        ]
        for column, title in enumerate(titles):
            axes[0, column].set_title(title, fontweight="bold")
        for row, (rgb, depth, normal) in enumerate(rows):
            axes[row, 0].imshow(np.clip(rgb, 0.0, 1.0))
            axes[row, 1].imshow(
                depth, cmap="viridis", vmin=depth_min, vmax=depth_max
            )
            if normal is None:
                axes[row, 2].set_axis_off()
            else:
                axes[row, 2].imshow(np.clip(normal, 0.0, 1.0))
            axes[row, 0].set_ylabel(
                row_labels[row],
                fontweight="bold" if row == 2 else "normal",
                color="#1f4e79" if row == 2 else "black",
            )
            for column in range(3):
                if normal is not None or column != 2:
                    clean_axis(axes[row, column])

    save_figure(figure, output_dir, FIGURE_REPLICA_STEM, preview_dpi)


def serializable_scene(scene: dict) -> dict:
    return {
        "id": scene["id"],
        "label": scene["label"],
        "dataset": scene["dataset"],
        "frame": scene["frame"],
        "ours_result": scene["ours_result"],
        "monogs_result": scene["monogs_result"],
        "gt_shape": list(scene["ours"]["gt_depth"].shape),
        "depth_range_m": scene["depth_range_m"],
        "error_range_m": scene["error_range_m"],
        "ours": {
            "mean_absolute_depth_error_m": scene["ours_stats"]["mean_absolute_error_m"],
            "coverage": scene["ours_stats"]["coverage"],
            "alignment_scale": float(scene["ours"]["alignment_scale"]),
        },
        "monogs": {
            "mean_absolute_depth_error_m": scene["monogs_stats"]["mean_absolute_error_m"],
            "coverage": scene["monogs_stats"]["coverage"],
            "alignment_scale": float(scene["monogs"]["alignment_scale"]),
        },
    }


def main() -> None:
    args = parse_args()
    configure_style()
    manifest = load_manifest(args.manifest.resolve())
    workspace_root = args.workspace_root.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dump_script = workspace_root / "2dgslam" / "scripts" / "dump_frame.py"
    ours_repo = workspace_root / "2dgslam"
    monogs_repo = workspace_root / "Original_MonoGS_Code"
    for path, description in (
        (args.ours_python.resolve(), "2DGSLAM Python interpreter"),
        (args.monogs_python.resolve(), "MonoGS Python interpreter"),
        (dump_script, "frame dumping script"),
    ):
        require_file(path, description)

    rendered_scenes = []
    for specification in manifest["scenes"]:
        scene = dict(specification)
        ours_result = workspace_root / scene["ours_result"]
        monogs_result = workspace_root / scene["monogs_result"]
        validate_result_dir(ours_result)
        validate_result_dir(monogs_result)

        ours_cache = cache_dir / f"{scene['id']}_ours.npz"
        monogs_cache = cache_dir / f"{scene['id']}_monogs.npz"
        logs = {}
        if not args.reuse_cache or not ours_cache.is_file():
            logs["ours"] = run_dump(
                args.ours_python.resolve(),
                dump_script,
                ours_repo,
                ours_result,
                scene["frame"],
                ours_cache,
            )
        if not args.reuse_cache or not monogs_cache.is_file():
            logs["monogs"] = run_dump(
                args.monogs_python.resolve(),
                dump_script,
                monogs_repo,
                monogs_result,
                scene["frame"],
                monogs_cache,
            )

        scene["ours"] = load_render(ours_cache)
        scene["monogs"] = load_render(monogs_cache)
        scene["render_logs"] = logs
        verify_ground_truth(scene["ours"], scene["monogs"], scene["id"])
        scene["ours_stats"] = depth_statistics(
            scene["ours"]["gt_depth"], scene["ours"]["depth"]
        )
        scene["monogs_stats"] = depth_statistics(
            scene["ours"]["gt_depth"], scene["monogs"]["depth"]
        )
        gt_valid = scene["ours"]["gt_depth"]
        gt_valid = gt_valid[np.isfinite(gt_valid) & (gt_valid > 0)]
        scene["depth_range_m"] = list(robust_range(gt_valid, 1.0, 99.0))
        combined_errors = np.concatenate(
            [
                scene["ours_stats"]["error"][np.isfinite(scene["ours_stats"]["error"])],
                scene["monogs_stats"]["error"][np.isfinite(scene["monogs_stats"]["error"])],
            ]
        )
        scene["error_range_m"] = [0.0, robust_range(combined_errors, 0.0, 95.0)[1]]
        rendered_scenes.append(scene)

    tum_scenes = [scene for scene in rendered_scenes if scene["dataset"] == "tum"]
    replica_scenes = [scene for scene in rendered_scenes if scene["dataset"] == "replica"]
    if len(tum_scenes) != 1 or len(replica_scenes) != 2:
        raise ValueError("Expected one TUM scene and two Replica scenes.")

    build_tum_figure(tum_scenes[0], output_dir, args.preview_dpi)
    build_replica_figure(replica_scenes, output_dir, args.preview_dpi)

    record = {
        "manifest": str(args.manifest.resolve()),
        "workspace_root": str(workspace_root),
        "figures": [
            f"{FIGURE_TUM_STEM}.pdf",
            f"{FIGURE_REPLICA_STEM}.pdf",
        ],
        "scenes": [serializable_scene(scene) for scene in rendered_scenes],
    }
    with (output_dir / METRICS_NAME).open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
