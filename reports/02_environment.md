# Environment & Toolchain Diagnostic Report: Prompt 02

**Date**: 2026-09-16  
**Environment**: Python 3.11.16 Virtual Environment (`.venv`)  
**Host & Distro**: Windows 11 (Host) / WSL2 Ubuntu 26.04 LTS (Guest, x86_64)  
**Project Path**: `/home/rohit/projects/insta360_mapper`  
**Overall Verdict**: `PASS` (All Prompt 02 criteria satisfied)

---

## 1. Environment & Dependency Architecture

### 1.1 Python & Dependency Manager
- **Python Version**: CPython `3.11.16` (pinned and isolated from system Python 3.14).
- **Virtual Environment**: `/home/rohit/projects/insta360_mapper/.venv`.
- **Dependency Manager**: `uv` (v0.12.15) with frozen lockfile (`uv.lock`).
- **Packaging Standard**: `pyproject.toml` managed with `hatchling` build backend; project package `vscan` installed in editable mode (`-e .`).

### 1.2 Pinned Package Toolchain
| Package | Version | Import Path / Wheel Details |
| :--- | :---: | :--- |
| `pycolmap-cuda12` | `4.2.0` | Published wheel with CUDA 12 support |
| `gtsam` | `4.2.2` | Native Georgia Tech Smoothing and Mapping library |
| `numpy` | `1.26.4` | C-ABI compatible with GTSAM and PyCOLMAP |
| `scipy` | `1.17.1` | Scientific computing & spatial rotation transforms |
| `opencv-python-headless`| `4.11.0.86` | Computer vision backend (headless, no X11 GUI dependencies) |
| `pillow` | `12.3.0` | Image ingestion and raster processing |
| `pydantic` | `2.13.5` | Data validation and configuration schemas |
| `typer` | `0.27.2` | CLI framework for `vscan` commands |
| `psutil` | `7.2.2` | System & process resource monitoring |
| `pytest` | `9.1.1` | Automated test suite framework |
| `hatchling` | `1.32.0` | Modern PEP 621 build backend |
| `nvidia-cuda-runtime-cu12` | `12.9.79` | CUDA runtime dynamic library (`libcudart.so.12`) |
| `nvidia-curand-cu12` | `10.3.10.19` | CUDA random number generator library (`libcurand.so.10`) |

### 1.3 System Packages & Distro Tools
- **FFmpeg**: `ffmpeg version 8.0.1-3ubuntu2` (`/usr/bin/ffmpeg`)
- **ffprobe**: `ffprobe version 8.0.1-3ubuntu2` (`/usr/bin/ffprobe`)
- **System Shared Libraries**: `libsm6`, `libice6` installed via Ubuntu `apt` to fulfill X11 session linkages required by PyCOLMAP's C++ bindings without installing a full desktop environment.

---

## 2. CUDA Runtime Resolution & Minimal Footprint

### 2.1 Inspection of Published PyCOLMAP Wheel
Dynamic symbol inspection (`ldd _core*.so`) revealed that `pycolmap-cuda12==4.2.0` requires:
- `libcudart.so.12`
- `libcurand.so.10`
- `libSM.so.6` & `libICE.so.6`

### 2.2 Minimal Fix (Zero Bloat Policy)
Rather than downloading and compiling a redundant multi-gigabyte CUDA development toolkit, we:
1. Installed lightweight pip wheels `nvidia-cuda-runtime-cu12` and `nvidia-curand-cu12` (~68 MiB total).
2. Installed tiny system packages `libsm6` and `libice6` (~61 KiB total).
3. Registered the virtual environment's CUDA runtime paths into `/etc/ld.so.conf.d/00-nvidia-pip-cuda.conf` and explicitly exported them in `scripts/bootstrap.sh`.
4. Result: Zero source compilation was required. The published wheel loads cleanly in any fresh shell.

---

## 3. Native Test Execution & Evidence

### 3.1 Numerical & Optimizer Tests (GTSAM & SciPy)
- **GTSAM Lie Group Operations**: Successfully evaluated `Rot3.Expmap` and `Pose3` construction.
- **Factor Graph Optimization**: Added `PriorFactorPose3` with `Isotropic.Sigma(6, 0.05)` noise model to a `NonlinearFactorGraph` and optimized via `GaussNewtonOptimizer`.
- **Result**: Translation preserved with an exact `0.0` error norm. No NumPy C-ABI crashes or segmentation faults.

### 3.2 PyCOLMAP SIFT GPU (CUDA) Feature Extraction
- **Input**: 512x512 reproducible test pattern (seeded noise + grid lines).
- **Execution**: Requested `device=pycolmap.Device.cuda`.
- **Resolved Device**: `Device.cuda` (Verified: silent automatic CPU fallback did NOT occur).
- **Keypoints Extracted**: `1,929` keypoints.
- **Descriptor Shape**: `(1929, 128)` float32 SIFT descriptors.
- **Timing**: `0.3072s` (well within the 120s probe limit).
- **Host RAM Delta**: ~88 MB allocated during GPU context initialization.

### 3.3 PyCOLMAP SIFT CPU Fallback Feature Extraction
- **Input**: Identical 512x512 test pattern.
- **Execution**: Requested `device=pycolmap.Device.cpu`.
- **Resolved Device**: `Device.cpu`.
- **Keypoints Extracted**: `2,030` keypoints.
- **Descriptor Shape**: `(2030, 128)`.
- **Timing**: `0.4089s`.
- **Fallback Assessment**: CPU fallback is fully operational and verified.

### 3.4 Fixed-Rig Construction & Serialization
- **Constructed Rig**: 2-camera rig consisting of `cam0_` (reference sensor) and `cam1_` (secondary sensor with `Rigid3d` translation `[0.1, 0, 0]` and identity rotation quaternion `[1, 0, 0, 0]`).
- **Serialization**: Round-tripped through COLMAP JSON specification and parsed via `pycolmap.read_rig_config`.
- **Result**: `PASS` (1 rig containing 2 cameras verified).

### 3.5 Video Decode Test (FFmpeg / ffprobe)
- Generated a 1-second 64x64 H.264 video using `ffmpeg` with `lavfi testsrc`.
- Probed stream parameters using `ffprobe`: confirmed `codec_name=h264`, `width=64`, `height=64`, `duration=1.000000`.
- **Result**: `PASS`.

---

## 4. PyCOLMAP Capability Audit & Explicit Distinctions

| Capability | Status | Notes |
| :--- | :---: | :--- |
| **CUDA Feature Extraction** | **VERIFIED** | SIFT extraction tested and operational on NVIDIA GeForce GTX 1650. |
| **CPU Fallback** | **VERIFIED** | CPU SIFT extraction tested and ready for low-memory scenarios. |
| **Fixed-Rig Constraints** | **VERIFIED** | `RigConfig`, `RigConfigCamera`, `apply_rig_config`, `read_rig_config` present and tested. |
| **Reconstruction Export** | **VERIFIED** | `Reconstruction.write`, `write_text`, `write_binary`, `export_PLY` present. |
| **Camera Undistortion** | **VERIFIED** | `undistort_images`, `undistort_image`, `undistort_camera` present. |
| **Dense Stereo (MVS)** | **DEFERRED** | PyCOLMAP bindings (`patch_match_stereo`, `stereo_fusion`) exist in library symbols, but **dense stereo is NOT tested or claimed operational here**. Dense stereo will be qualified on actual data in Prompt 09. |
| **Panorama Helpers** | **ABSENT** | Convenience panorama code is absent in PyCOLMAP 4.2.0. As planned in the build contract, this will be handled by our custom tested ray renderer in Prompt 05. |

---

## 5. Environment Scripts & CLI Interface

1. **`src/vscan` CLI**:
   - `vscan doctor`: Formatted, readable health summary.
   - `vscan doctor --json`: Pure JSON output suitable for automated pipelines and companion records.
2. **`scripts/bootstrap.sh`**:
   - Explicit environment activation (`source .venv/bin/activate`).
   - Thread limits exported: `OMP_NUM_THREADS=6`, `MKL_NUM_THREADS=6`, `NUMEXPR_NUM_THREADS=6`, `OPENBLAS_NUM_THREADS=6` (preserves Windows 11 host budget).
   - Idempotent: Subsequent runs verify the existing virtual environment without wiping or rebuilding.
   - Clean shutdown: No lingering background processes, daemons, or jobs.

---

## 6. Run / Stop / Resume Guide

- **Run Bootstrap**:
  ```bash
  bash /home/rohit/projects/insta360_mapper/scripts/bootstrap.sh
  ```
- **Run Doctor Directly**:
  ```bash
  /home/rohit/projects/insta360_mapper/.venv/bin/vscan doctor
  ```
- **Run Doctor JSON**:
  ```bash
  /home/rohit/projects/insta360_mapper/.venv/bin/vscan doctor --json
  ```
- **Stop**:
  - Interactive commands can be interrupted at any time using `Ctrl+C`.
  - No background reconstruction or server processes are left running.
- **Resume**:
  - Activate environment via `source /home/rohit/projects/insta360_mapper/.venv/bin/activate`.

---

## 7. Stage Status & Handoff

- **Prompt 02 Status**: Completed (`PASS`).
- **Prompt 03 Readiness**: Ready to proceed.
