import os
import re
import signal
import subprocess
import time
from pathlib import Path


RUNS = [
    ("BASE", "configs/mono/tum/fr1_desk_cutoff_exp.yaml"),
    ("A1", "configs/mono/tum/fr1_desk_drift_A1.yaml"),
    ("A2", "configs/mono/tum/fr1_desk_drift_A2.yaml"),
    ("A3", "configs/mono/tum/fr1_desk_drift_A3.yaml"),
    ("B1", "configs/mono/tum/fr1_desk_drift_B1.yaml"),
    ("B2", "configs/mono/tum/fr1_desk_drift_B2.yaml"),
    ("B3", "configs/mono/tum/fr1_desk_drift_B3.yaml"),
    ("B4", "configs/mono/tum/fr1_desk_drift_B4.yaml"),
    ("B5", "configs/mono/tum/fr1_desk_drift_B5.yaml"),
    ("C1", "configs/mono/tum/fr1_desk_drift_C1.yaml"),
    ("C2", "configs/mono/tum/fr1_desk_drift_C2.yaml"),
    ("C3", "configs/mono/tum/fr1_desk_drift_C3.yaml"),
    ("D1", "configs/mono/tum/fr1_desk_drift_D1.yaml"),
    ("D2", "configs/mono/tum/fr1_desk_drift_D2.yaml"),
]

LOG_DIR = Path("/home/2DGS_SLAM/MonoGS/log")
SUMMARY_PATH = LOG_DIR / "drift_matrix_summary.tsv"

MAX_SECONDS = 4200
TARGET_FRAME = 242


def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def clear_processes():
    run_cmd('pkill -f "python .*slam.py" || true')
    time.sleep(2)
    smi = run_cmd("nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits")
    if smi.returncode == 0:
        for line in smi.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            pid, pname = parts
            if pid.isdigit() and "python" in pname:
                run_cmd(f"kill {pid} || true")
    time.sleep(2)


def parse_log(log_path: Path):
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    result_match = re.search(r"results/tum_rgbd_dataset_freiburg1_desk/[0-9\-]+", text)
    result_path = result_match.group(0) if result_match else ""
    reset_count = text.count("MonoGS: Resetting the system")
    overlap = "Keyframes lacks sufficient overlap" in text

    evals = []
    frames = re.findall(r"Evaluating ATE at frame:\s+(\d+)", text)
    rmses = re.findall(r"Eval: RMSE ATE \[m\]\s+([0-9eE+\-.]+)", text)
    for i, f in enumerate(frames):
        rmse = float(rmses[i]) if i < len(rmses) else None
        evals.append((int(f), rmse))

    first_frame = evals[0][0] if evals else None
    first_rmse = evals[0][1] if evals else None

    checkpoints = {110: None, 143: None, 188: None, 242: None}
    for target in checkpoints:
        exact = [r for f, r in evals if f == target]
        if exact:
            checkpoints[target] = exact[0]
        else:
            nearest = [r for f, r in evals if abs(f - target) <= 5]
            checkpoints[target] = nearest[0] if nearest else None

    max_frame = max([f for f, _ in evals], default=0)
    return {
        "result_path": result_path,
        "reset_count": reset_count,
        "overlap": overlap,
        "first_frame": first_frame,
        "first_rmse": first_rmse,
        "rmse110": checkpoints[110],
        "rmse143": checkpoints[143],
        "rmse188": checkpoints[188],
        "rmse242": checkpoints[242],
        "max_frame": max_frame,
        "evals": evals,
    }


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "ID\tCONFIG\tLOG\tRESULT_PATH\tRESET_COUNT\tOVERLAP_RESET\tFIRST_FRAME\tFIRST_RMSE\tRMSE110\tRMSE143\tRMSE188\tRMSE242\tMAX_FRAME\tSTATUS"
    ]

    for rid, cfg in RUNS:
        clear_processes()
        log_path = LOG_DIR / f"drift_{rid}.log"

        with log_path.open("w") as f:
            proc = subprocess.Popen(
                [
                    "conda",
                    "run",
                    "-n",
                    "MonoGS",
                    "--no-capture-output",
                    "python",
                    "-u",
                    "slam.py",
                    "--config",
                    cfg,
                ],
                cwd="/home/2DGS_SLAM/MonoGS",
                stdout=f,
                stderr=subprocess.STDOUT,
            )

            start = time.time()
            checks = 0
            status = "timeout"
            while True:
                elapsed = time.time() - start
                if proc.poll() is not None:
                    status = "completed"
                    break
                if elapsed > MAX_SECONDS:
                    status = "timeout"
                    break

                time.sleep(180 if elapsed < 600 else (300 if elapsed < 1800 else 600))
                checks += 1
                snapshot = parse_log(log_path)
                if snapshot["max_frame"] >= TARGET_FRAME:
                    status = "target_frame_reached"
                    break

            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                time.sleep(3)
                if proc.poll() is None:
                    proc.kill()

        clear_processes()
        m = parse_log(log_path)
        line = "\t".join(
            [
                rid,
                cfg,
                str(log_path.relative_to("/home/2DGS_SLAM/MonoGS")),
                m["result_path"],
                fmt(m["reset_count"]),
                fmt(m["overlap"]),
                fmt(m["first_frame"]),
                fmt(m["first_rmse"]),
                fmt(m["rmse110"]),
                fmt(m["rmse143"]),
                fmt(m["rmse188"]),
                fmt(m["rmse242"]),
                fmt(m["max_frame"]),
                status,
            ]
        )
        lines.append(line)

        perfect = (
            m["overlap"]
            and m["first_frame"] is not None
            and 70 <= m["first_frame"] <= 80
            and (m["first_rmse"] is not None and m["first_rmse"] < 0.015)
            and (m["rmse188"] is not None and m["rmse188"] < 0.04)
            and (m["rmse242"] is not None and m["rmse242"] < 0.06)
        )
        if perfect:
            break

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
