# MonoGS Compatibility Fixes

## Goal

Run `python slam.py --config configs/mono/tum/fr3_office.yaml` without error on a system where CUDA IPC (Inter-Process Communication) is broken.

**Constraint:** Fix compatibility issues only — do not change algorithm logic.

---

## Root Cause

The system uses `mp.set_start_method("spawn")` with `mp.Queue` for inter-process communication between:
- **FrontEnd** (main process) ↔ **BackEnd** (subprocess via `mp.Process`)
- **FrontEnd** (main process) ↔ **GUI** (subprocess via `mp.Process`)

On this system, CUDA IPC is broken: any CUDA tensor crossing a process boundary via `mp.Queue` (which uses `ForkingPickler`) triggers:

```
RuntimeError: CUDA error: invalid resource handle
```

Or causes a **segmentation fault** in the receiving subprocess.

This is because PyTorch's `ForkingPickler` uses CUDA shared memory (IPC) to transfer tensors across processes. When IPC is unavailable, this fails. `nn.Parameter` objects are affected even when `__getstate__` converts them to CPU, because PyTorch's custom reducer for parameters bypasses the standard pickle protocol.

---

## Changes Made

### 1. `slam.py` — Pass CPU tensors to subprocesses

**Lines 76, 85, 97**

The `background` tensor was created on CUDA and passed directly to subprocess objects before they started. Changed to pass `.cpu()` copies so the tensor is CPU at subprocess start time.

```python
# Before
self.frontend.background = self.background
self.backend.background = self.background
# (params_gui also received CUDA background)

# After
self.frontend.background = self.background.cpu()
self.backend.background = self.background.cpu()
self.params_gui = gui_utils.ParamsGUI(..., background=self.background.cpu(), ...)
```

---

### 2. `utils/slam_frontend.py` — Restore background to CUDA in subprocess; serialize Camera for queue

**Line 320** (in `run()`): Move `background` back to CUDA after the subprocess starts:
```python
self.background = self.background.cuda()
```

**Lines 289–300** (in `request_keyframe`, `reqeust_mapping`, `request_init`): Serialize `Camera` objects to a CPU-only dict before putting them in `backend_queue`:
```python
# Before
msg = ["init", cur_frame_idx, viewpoint, depth_map]

# After
msg = ["init", cur_frame_idx, viewpoint.to_dict(), depth_map]
```

**Lines 302–313** (in `sync_backend`): Reconstruct `GaussianModel` from CPU dict received from backend, and move `occ_aware_visibility` back to CUDA:
```python
# Before
self.gaussians = data[1]

# After
self.gaussians = GaussianModel.from_dict(data[1])
self.occ_aware_visibility = {k: v.cuda() for k, v in occ_aware_visibility.items()}
```

**Import added:** `from gaussian_splatting.scene.gaussian_model import GaussianModel`

---

### 3. `utils/slam_backend.py` — Restore background to CUDA; deserialize Camera from queue; serialize GaussianModel for queue

**Line 372** (in `run()`): Move `background` back to CUDA after the subprocess starts:
```python
self.background = self.background.cuda()
```

**Lines 401–404, 415–419** (in `run()` handlers for `"init"` and `"keyframe"`): Reconstruct `Camera` from CPU dict received from `backend_queue`:
```python
# Before
viewpoint = data[2]

# After
viewpoint = Camera.from_dict(data[2])
```

**Lines 355–369** (in `push_to_frontend`): Serialize `GaussianModel` to CPU dict and `occ_aware_visibility` to CPU before putting them in `frontend_queue`:
```python
# Before
msg = [tag, clone_obj(self.gaussians), occ_aware_visibility, keyframes]

# After
occ_aware_visibility_cpu = {k: v.cpu() for k, v in self.occ_aware_visibility.items()}
keyframes = [(kf_idx, kf.R.clone().cpu(), kf.T.clone().cpu()) for ...]
msg = [tag, self.gaussians.to_dict(), occ_aware_visibility_cpu, keyframes]
```

**Import added:** `from utils.camera_utils import Camera`

---

### 4. `utils/camera_utils.py` — Add `to_dict` / `from_dict` for safe queue transfer

Replaced the broken `__getstate__`/`__setstate__` approach (which did not work because `ForkingPickler` bypasses `__getstate__` for `nn.Parameter` objects) with explicit serialization methods:

**`Camera.to_dict()`**: Serializes all tensors and parameters to CPU-only Python dict. This plain dict can safely cross process boundaries via `mp.Queue`.

**`Camera.from_dict(d)`** (static method): Reconstructs a full `Camera` object on CUDA from the CPU dict, restoring pose, grad_mask, and all `nn.Parameter` members.

---

### 5. `gaussian_splatting/scene/gaussian_model.py` — Add `to_dict` / `from_dict` for safe queue transfer

Added serialization methods to transfer Gaussian parameters across processes. The optimizer state is **not** transferred (only parameter data), since the frontend only uses the received gaussians for rendering, not for gradient updates.

**`GaussianModel.to_dict()`**: Serializes all CUDA tensors/parameters to CPU-only Python dict.

**`GaussianModel.from_dict(d)`** (static method): Reconstructs a `GaussianModel` on CUDA from the CPU dict. The resulting model has no optimizer.

---

### 6. `gui/slam_gui.py` — Restore background to CUDA in GUI subprocess

**In `run()` function**: Move `background` back to CUDA at the start of the GUI subprocess:
```python
params_gui.background = params_gui.background.cuda()
```

---

### 7. `gui/gui_utils.py` + `gui/slam_gui.py` + `slam.py` — Fix CUDA IPC error on `q_main2vis` (GUI queue)

**Root cause:** `GaussianPacket` objects sent from the frontend to the GUI subprocess via `q_main2vis` contained raw CUDA tensors and `Camera` objects with CUDA `nn.Parameter` members. Unpickling these in the GUI subprocess triggered PyTorch's CUDA IPC (`rebuild_cuda_tensor`), which fails with `RuntimeError: CUDA error: invalid resource handle`. This also caused the blank white GUI window and "error loading image" in the panel.

**Fix in `gui/gui_utils.py`:**
- Added `from utils.camera_utils import Camera` import.
- In `GaussianPacket.__init__`: all Gaussian tensors moved to CPU with `.cpu()`. `gtcolor`/`gtdepth`/`gtnormal` tensors also moved to CPU. `current_frame`, `keyframe`, and `keyframes` Camera objects serialized to CPU dicts via `Camera.to_dict()`.
- Added `GaussianPacket.to_cuda()`: moves all tensors back to CUDA and reconstructs Camera objects via `Camera.from_dict()`.

**Fix in `gui/slam_gui.py` (`receive_data`):**
- Called `gaussian_packet.to_cuda()` immediately after dequeuing, before the GUI uses any data.

**Fix in `slam.py`:**
- Changed `gaussians=self.gaussians` to `gaussians=None` in `ParamsGUI(...)`. The initial `GaussianModel` CUDA parameters were being pickled into the GUI subprocess at spawn time.

**Fix in `gui/slam_gui.py` (`__init__`):**
- Guarded `self.gaussian_cur` and `self.init = True` behind `if params_gui.gaussians is not None:`.

---

### 8. `gui/slam_gui.py` — Fix OpenGL context conflict between Open3D/Filament and GLFW

**In `init_glfw()`**: Add `glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)` before `glfw.create_window()`.

**Root cause:** PyOpenGL on this system uses the EGL platform (`eglGetCurrentContext`) to identify the active GL context. Open3D's Filament renderer initialises its own EGL context during `app.initialize()`, which becomes the current EGL context at the OS level. GLFW's default context creation API is GLX (X11 native), so the GLFW window's context is a GLX context — invisible to `eglGetCurrentContext()`. As a result, every subsequent PyOpenGL call fails with:

```
OpenGL.error.Error: Attempt to retrieve context when no valid context
```

**Fix:** Hint GLFW to use the EGL context creation API so that after `glfw.make_context_current(window)`, `eglGetCurrentContext()` returns a valid handle and PyOpenGL can proceed normally. The harmless `libEGL warning: DRI3 error` that appears on stderr is expected on this system and does not affect functionality.

---

## Summary of the Pattern

All CUDA tensor transfers across `mp.Queue` are replaced with a **serialize → send → deserialize** pattern:

1. **Before sending**: call `.to_dict()` to produce a CPU-only plain Python dict
2. **After receiving**: call `.from_dict()` to reconstruct the object on CUDA

This avoids PyTorch's CUDA IPC mechanism entirely and uses standard Python pickling of CPU tensors (which works reliably).

---

## Verification

Both modes complete successfully and produce valid SLAM results.

**Eval mode** (`--eval`):
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml --eval
```

**GUI mode** (default):
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml
```

Sample output (both modes):
```
MonoGS: Resetting the system
MonoGS: Initialized map
MonoGS: Performing initial BA for initialization
MonoGS: Initialized SLAM
MonoGS: Evaluating ATE at frame:  112
Eval: RMSE ATE [m] 0.0081
MonoGS: Evaluating ATE at frame:  205
Eval: RMSE ATE [m] 0.0181
```

The RMSE ATE values are in the expected range for the TUM fr3/office sequence.
