# DRIFT_FIX.md

## Goal

Stabilize integrated 2DGS MonoGS on TUM desk sequence so that it matches 3DGS-style initialization behavior and avoids late severe drift.

### Hard acceptance criteria
1. **Init behavior**
   - Must trigger overlap-based map reset at least once in early phase (like Original MonoGS).
   - First ATE eval around frame **~74** (acceptable band: 70-80).
   - First eval RMSE ATE < **0.015** (minimum bar: < 0.1, target much tighter).

2. **Long-horizon stability**
   - RMSE at ~frame 188 < **0.04**
   - RMSE at ~frame 242 < **0.06**
   - No sharp divergence trend after ~140.

---

## Known facts from prior runs

- Pre-init overlap rejection is critical:
  - `init_kf_cutoff=0.4` restores expected early reset/init behavior.
- However late drift still occurs:
  - Example: frame 72 low RMSE, then drift grows by frame 143/188/242.
- So the remaining issue is primarily **post-init mapping stability**, not init keyframe gating.

---

## Execution protocol (must follow every run)

1. `nvidia-smi` check.
2. Kill stale `slam.py`-related python processes only.
3. `nvidia-smi` re-check (ensure clean).
4. Run with:
   - `conda run -n MonoGS --no-capture-output python -u slam.py --config <config>`
5. Heavy command output must go to `log/`.
6. Poll logs instead of waiting indefinitely:
   - first 10 min: every 3 min
   - 10-30 min: every 5 min
   - 30-70 min: every 10 min
7. After extracting metrics into this file, delete temporary log file.
8. Do **not** delete any folder/data/result directory.

---

## Baseline anchor run

- Config: `configs/mono/tum/fr1_desk_cutoff_exp.yaml`
- Purpose: confirm reproducible init behavior before tuning.
- Required observations:
  - reset count,
  - whether overlap-reset occurs,
  - first eval frame + RMSE,
  - RMSE at ~110, ~143, ~188, ~242.

---

## Experiment strategy (ordered)

### Stage A - Opacity reset timing (highest priority for late drift)

Hypothesis: late divergence is amplified after opacity reset events.

Tune one variable at a time:
- `Training.gaussian_reset` (delay/reset less often)
- `opt_params.opacity_reset_interval` (if active in current path)

Ablations:
- A0: current baseline
- A1: increase `gaussian_reset`
- A2: increase `opacity_reset_interval`
- A3: increase both

Pass condition to continue:
- better RMSE at ~188 and ~242 vs A0 without harming first-eval behavior.

---

### Stage B - Densification growth control

Hypothesis: map overgrowth drives long-horizon instability.

Tune:
- `Training.gaussian_update_every` (higher)
- `opt_params.densify_until_iter` (lower)
- `Training.gaussian_th` (higher pruning threshold)

Ablations:
- B1: slower updates only
- B2: earlier densify stop only
- B3: stronger pruning only
- B4: best pairwise combination
- B5: best triple combination

Monitor:
- gaussian count trend at eval checkpoints
- RMSE slope after frame ~140

---

### Stage C - Pose update stiffness

Hypothesis: post-init pose flexibility accumulates drift.

Tune:
- `Training.pose_window`
- `Training.lr.cam_rot_delta`
- `Training.lr.cam_trans_delta`

Ablations:
- C1: lower pose_window
- C2: lower pose LRs
- C3: combine C1+C2

Guardrail:
- reject if first-eval frame shifts badly (<65 or >90) or first RMSE degrades heavily.

---

### Stage D - Keyframe cadence refinement (only if needed)

Keep `init_kf_cutoff=0.4` fixed.
Tune late keyframe policy only:
- `Training.kf_interval`
- `Training.kf_overlap`
- `Training.kf_cutoff` (post-init)

Goal:
- avoid unstable keyframe accumulation while preserving coverage.

---

### Stage E - Beyond config (if all config sweeps fail)

Allowed only after Stage A-D exhausted:
- expose/tune additional non-config constants in code paths
- prioritize post-init growth/loss routing controls
- keep one-change-per-run discipline

---

## Experiment table (fill this for every run)

| ID | Stage | Config / Change | Command | Log Path | Result Path | Reset Count | Overlap Reset Triggered? | First Eval Frame | First RMSE | RMSE@~110 | RMSE@~143 | RMSE@~188 | RMSE@~242 | Gaussian Count Notes | Verdict |
|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| BASE | Baseline | fr1_desk_cutoff_exp | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_cutoff_exp.yaml` | `log/drift_BASE.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-22-06-30` | 2 | yes | 72 | 0.013986 | 0.016374 | 0.028732 (frame 148) | 0.041636 | 0.077352 (frame 232) | late drift climbs after ~188, severe by ~232 | fail |
| A1 | A | `gaussian_reset: 4000` | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_A1.yaml` | `log/drift_A1.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-22-48-38` | 2 | yes |  |  |  |  |  |  | run terminated before first eval; likely unstable/incomplete run | fail |
| A2 | A | `opacity_reset_interval: 6000` | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_A2.yaml` | `log/drift_A2.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-23-10-54` | 2 | yes | 72 | 0.013309 | 0.013471 (frame 111) | 0.020442 (frame 146) | 0.030517 (frame 186) | 0.043097 (frame 223) | best so far; drift controlled much better through ~269 | pass |
| A3 | A | `gaussian_reset: 4000` + `opacity_reset_interval: 6000` | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_A3.yaml` | `log/drift_A3.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-01-23-52-01` | 2 | yes |  |  |  |  |  |  | stalled after second reset (no initial BA/eval after >15 min); rejected early | fail |
| B1 | B | `gaussian_update_every: 200` (A3 base) | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_B1.yaml` | `log/drift_B1.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-00-12-31` | 2 | yes |  |  |  |  |  |  | stalled after second reset (no BA/eval in ~15 min) | fail |
| B2 | B | `densify_until_iter: 10000` (A3 base) | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_B2.yaml` | `log/drift_B2.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-00-27-30` | 2 | yes | 72 | 0.013419 | 0.014898 (frame 108) | 0.022585 (frame 142) | 0.048435 (frame 191) |  | drift at ~191 still above 0.04 gate; no perfect run | fail |
| B3 | B | `gaussian_th: 0.85` (A3 base) | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_B3.yaml` | `log/drift_B3.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-00-58-27` | 2 | yes |  |  |  |  |  |  | stalled after second reset (no BA/eval in ~15 min) | fail |
| B4 | B | `gaussian_update_every:200` + `gaussian_th:0.85` + `densify_until_iter:10000` (A3 base) | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_B4.yaml` | `log/drift_B4.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-01-13-39` | 2 | yes |  |  |  |  |  |  | stalled after second reset (no BA/eval in ~15 min) | fail |
| B5 | B | `B4` + `gaussian_update_offset: 80` | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_B5.yaml` | `log/drift_B5.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-01-28-49` | 2 | yes |  |  |  |  |  |  | stalled after second reset (no BA/eval in ~15 min) | fail |
| C1 | C | `pose_window: 2` (B5 base) | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_C1.yaml` | `log/drift_C1.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-01-44-00` | 3 | yes |  |  |  |  |  |  | unstable reset loop (repeated overlap resets), aborted early | fail |
| C2 | C | `cam_rot_delta:0.002`, `cam_trans_delta:0.0007` (B5 base) | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_C2.yaml` | `log/drift_C2.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-01-59-19` | 3 | yes |  |  |  |  |  |  | unstable reset loop (repeated overlap resets), aborted early | fail |
| C3 | C | `pose_window:2` + reduced pose LR (B5 base) | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_C3.yaml` | `log/drift_C3.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-02-14-31` | 2 | yes |  |  |  |  |  |  | stalled after second reset (no BA/eval in ~15 min) | fail |
| D1 | D | `kf_interval:6`, `kf_overlap:0.92`, `kf_cutoff:0.35` (C3 base) | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_D1.yaml` | `log/drift_D1.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-02-29-25` | 2 | yes |  |  |  |  |  |  | stalled after second reset (no BA/eval in ~15 min) | fail |
| D2 | D | `kf_interval:8`, `kf_overlap:0.95`, `kf_cutoff:0.35` (C3 base) | `conda run -n MonoGS --no-capture-output python -u slam.py --config configs/mono/tum/fr1_desk_drift_D2.yaml` | `log/drift_D2.log` | `results/tum_rgbd_dataset_freiburg1_desk/2026-04-02-02-44-14` | 3 | yes |  |  |  |  |  |  | unstable reset loop (repeated overlap resets), aborted early | fail |

---

## Decision rule after each stage

- Promote only candidates that improve **both**:
  - RMSE at ~188 and ~242
  - without breaking early init acceptance criteria.
- If no candidate in a stage beats baseline, revert to best prior config and move to next stage.
- Stop when one candidate satisfies all hard acceptance criteria.

---

## Final report template

### Best config
- Config path:
- Key params changed:
- Result path:
- First eval frame/RMSE:
- RMSE@143, @188, @242:
- Reset behavior summary:
- Gaussian count trend summary:

### Comparison vs Original MonoGS baseline
- Matching points:
- Remaining gaps:
- Risk assessment:
- Next recommended action:

---

## Run outcome summary (current session)

- No perfect run was found in this matrix.
- Best observed candidate remains **A2** (`opacity_reset_interval: 6000` on cutoff baseline):
  - frame 72 RMSE 0.013309
  - frame 186 RMSE 0.030517
  - frame 223 RMSE 0.043097
  - frame 269 RMSE 0.045639
- This improves late drift significantly vs baseline, but still not enough evidence yet for passing strict long-horizon gate at ~242 because nearest measured checkpoint was frame 223.
