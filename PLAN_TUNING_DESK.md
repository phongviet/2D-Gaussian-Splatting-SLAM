# Desk Sequence Tuning Plan and Results

## Objective
- Match or exceed the Original MonoGS desk early trajectory behavior while improving long-horizon drift stability.
- Target reference for first evaluation: frame around 74 with low RMSE ATE (~0.009).

## Operational Protocol
- Before each run:
  - Check `nvidia-smi`.
  - Kill stale `slam.py` processes only.
  - Re-check `nvidia-smi` to confirm no stale SLAM process remains.
- Use `conda run -n MonoGS --no-capture-output` for all runs.
- Route heavy command output to `log/`.
- Monitor logs at intervals, not by waiting indefinitely:
  - first 10 min: every 3 min
  - 10 to 30 min: every 5 min
  - 30 to 70 min: every 10 min
- Never delete folders/files except temporary log files after extracting results.

## Experiment Stages

### Stage 1: `pcd_downsample` Sweep (Primary)
- Increase both values until no further late-drift improvement:
  - 128/64 (baseline)
  - 160/80
  - 192/96
  - 224/112
  - 256/128
- Stop sweep when two consecutive increases show no gain.

### Stage 2: Config Tuning (Only after Stage 1 winner)
- Growth controls:
  - `gaussian_update_every`
  - `densify_until_iter`
  - `gaussian_th`
- Pose stability:
  - `pose_window`
  - `lr.cam_rot_delta`
  - `lr.cam_trans_delta`

### Stage 3: Beyond Config (If required)
- If all config attempts fail to approach Original MonoGS trajectory quality, tune parameters outside config through controlled code-level changes.

## Results Table

| Attempt | Stage | Params Changed | Log Path | Result Path | First Eval Frame | First Eval RMSE | Mid/Late Drift Notes | Status |
|---|---|---|---|---|---:|---:|---|---|
| A0 | Stage 1 | pcd=128 init=64 | `log/desk_A0.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-02-26-17` | 45 | 0.02530 | Early eval too soon; no overlap-reset observed in captured interval | fail |
| A1 | Stage 1 | pcd=160 init=80 | `log/desk_A1.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-02-40-50` | 50 | 0.03455 | Worse early RMSE; frame 103 RMSE 0.04964 | fail |
| A2 | Stage 1 | pcd=192 init=96 | `log/desk_A2.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-02-54-59` | 49 | 0.02019 | Better than A1 but still early and above target trend | fail |
| A3 | Stage 1 | pcd=224 init=112 | `log/desk_A3.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-03-09-01` | 47 | 0.01690 | Best Stage-1 RMSE trend but still early eval frame | fail |
| A4 | Stage 1 | pcd=256 init=128 | `log/desk_A4.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-03-23-21` | 49 | 0.02064 | Regressed from A3; no gain from further increase | fail |
| B0 | Stage 2 | A3 + kf_interval=8, kf_overlap=0.95, kf_cutoff=0.35 | `log/desk_B0.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-03-38-43` | 122 | 0.04714 | Over-delayed keyframe creation; severe early RMSE | fail |
| B1 | Stage 2 | B0 + gaussian_update_every=200, gaussian_update_offset=80, gaussian_th=0.85, densify_until_iter=10000 | `log/desk_B1.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-04-03-38` | 66 | 0.03260 | Frame timing improved vs B0 but RMSE still poor | fail |
| B2 | Stage 2 | B1 + pose_window=2, cam_rot=0.002, cam_trans=0.0007 | `log/desk_B2.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-04-18-20` | 66 | 0.03466 | No improvement over B1; slight regression | fail |
| C0 | Stage 2 (Overlap reset hypothesis) | A3 + `init_kf_cutoff=0.4` (code path uses config for pre-init cutoff) | `log/cutoff_exp.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-20-01-50` | 72 | 0.00778 | Triggered one overlap-reset cycle then stable init; frame 115 RMSE 0.01108 | **match (early-stage)** |

## Best Candidate So Far
- Stage-1 best by early RMSE trend: **A3** (`pcd_downsample=224`, `pcd_downsample_init=112`) with frame 47 / RMSE 0.01690.
- This still does not meet target behavior (frame ~74 with ~0.009).

## Current Conclusion
- Increasing `pcd_downsample` beyond 224 does not improve performance (A4 regressed), so Stage-1 plateau reached.
- Stage-2 config-only tuning attempted here did not recover target trajectory quality.
- Next step allowed by plan: controlled parameter changes outside config (code-path parameters/loss routing) if proceeding further.

## New Verification Note (Overlap Hypothesis)
- The pre-init overlap cutoff value is high-impact. Restoring pre-init cutoff to `0.4` (instead of hardcoded `0.1`) produced behavior close to 3DGS baseline pattern:
  - explicit overlap-triggered reset once,
  - first eval around frame ~80 neighborhood (observed 72),
  - first ATE well below 0.1 (0.00778).

## Notes
- Keep links/paths to result folders in this file for traceability.
- Delete each `log/*.log` after its row is fully filled.
