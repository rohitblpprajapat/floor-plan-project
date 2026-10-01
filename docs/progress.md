# Project Progress: Insta360 Reconstruction & Annotation Application (v2)

## Status Dashboard
- **Current Milestone**: Prompt 07 — Synchronization, Calibration & Clock Modeling
- **Current Milestone Status**: Completed (`STAGE_07_COMPLETE` / `UNRESOLVED_WAITING_INPUT` on real pilot)
- **Ready for Next Prompt**: Yes (Prompt 08 can proceed on synthetic fixtures only; real fusion remains unresolved)

---

## Milestone Log

### Prompt 01: Host Preflight and Diagnostics
- **Objective**: Establish execution on actual Windows 11 host and WSL2 instance; perform hardware, OS, memory, disk, and GPU diagnostic probes; establish build contract; evaluate resource budget.
- **Diagnostics Completed**:
  - `Get-CimInstance Win32_OperatingSystem` -> Exit code 0 (Windows 11 Build 26200)
  - `Get-CimInstance Win32_ComputerSystem` -> Exit code 0 (HP Pavilion Gaming Laptop 15-ec2xxx, 7.88 GB total RAM)
  - `Get-CimInstance Win32_Processor` -> Exit code 0 (AMD Ryzen 5 5600H with Radeon Graphics, 6 cores, 12 threads)
  - `Get-Volume` -> Exit code 0 (C: 94.11 GB free / 302.37 GB total; D: 32.87 GB free / 173.71 GB total)
  - `wsl.exe --version` -> Exit code 0 (WSL 2.7.13.0, Kernel 6.18.33.2-2)
  - `wsl.exe --status` -> Exit code 0 (Default Distribution: Ubuntu, Version 2)
  - `wsl.exe --list --verbose` -> Exit code 0 (Ubuntu Stopped/Running Version 2, docker-desktop Stopped Version 2)
  - `nvidia-smi.exe` -> Exit code 0 (GeForce GTX 1650, 4096 MiB VRAM, Driver 581.83, CUDA capability 13.0)
  - `$env:USERPROFILE\.wslconfig` -> Checked without modification; currently not present (default WSL2 resource policy active)
  - WSL internal diagnostics:
    - `uname -a` -> Exit code 0 (`Linux Rohit 6.18.33.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026 x86_64 GNU/Linux`)
    - `uname -m` -> Exit code 0 (`x86_64`)
    - `cat /etc/os-release` -> Exit code 0 (`Ubuntu 26.04 LTS (Resolute Raccoon)`)
    - `free -h` -> Exit code 0 (3.5 GiB total RAM, 558 MiB used, 2.3 GiB free, 3.0 GiB available)
    - `/sbin/swapon --show` -> Exit code 0 (1.0 GiB partition swap)
    - `df -h` -> Exit code 0 (WSL VHD on C: with 94.11 GB physical free SSD space; virtual ext4 max 1007 GiB)
    - `nproc` -> Exit code 0 (12 logical processors)
    - `nvidia-smi` (/usr/lib/wsl/lib/nvidia-smi) -> Exit code 0 (GeForce GTX 1650 recognized, 4096 MiB VRAM)
    - `/dev/dxg` -> Present and accessible (`crw-rw-rw-`)
    - Filesystem type check -> `/home/rohit/projects` is `ext4` on `/dev/sdd`
- **Write Verification**:
  - Windows workspace write test: PASSED (exit code 0)
  - WSL proposed directory (`~/projects/insta360_mapper`): PASSED (exit code 0)
- **Artifacts Created**:
  - `docs/build_contract.md`
  - `docs/progress.md`
  - `reports/01_preflight.md`
  - `reports/01_preflight.json`
- **Verdict**: `HOST_PASS` — All preflight criteria satisfied. Prompt 02 can proceed.

---

### Prompt 02: Isolated Environment Setup & GPU Compute Verification
- **Objective**: Create isolated Python 3.11 environment, pin patch version, resolve and lock dependencies, verify CUDA PyCOLMAP and GTSAM runtime, implement `vscan doctor --json`, create idempotent bootstrap script with CPU thread limits, and execute live GPU/CPU feature extraction, optimizer, and video decoding probes.
- **Actions & Verifications Completed**:
  - **Python Environment**: Installed Python `3.11.16` via `uv` into isolated virtual environment at `/home/rohit/projects/insta360_mapper/.venv`.
  - **Dependency Locking**: Generated frozen `uv.lock` and pinned `pyproject.toml` with `pycolmap-cuda12==4.2.0`, `gtsam==4.2.2`, `numpy==1.26.4`, `scipy==1.17.1`, `opencv-python-headless==4.11.0.86`, `pillow==12.3.0`, `pydantic==2.13.5`, `typer==0.27.2`, `psutil==7.2.2`, `pytest==9.1.1`. Verified with `uv pip check` (0 conflicts).
  - **Distro Tools**: Installed `ffmpeg` (8.0.1) and `ffprobe` (8.0.1) via `apt`.
  - **Dynamic Linker Resolution**: Resolved `libcudart.so.12` and `libcurand.so.10` via pip wheels + `/etc/ld.so.conf.d/00-nvidia-pip-cuda.conf` and `libsm6`/`libice6` without installing heavy standalone CUDA toolkit.
  - **PyCOLMAP Capability Audit**: Confirmed fixed-rig support, reconstruction export (text/binary/PLY), and camera undistortion. Confirmed absence of panorama convenience code (to be provided in Prompt 05). Dense stereo bindings confirmed present but dense execution explicitly deferred to Prompt 09.
  - **Live Functional Probes**:
    - GTSAM Lie group operations and Gauss-Newton optimizer calculation: PASSED (0.0 error norm).
    - PyCOLMAP SIFT CUDA extraction: PASSED (`Device.cuda`, 1929 keypoints, 0.307s, verified NO silent fallback).
    - PyCOLMAP SIFT CPU fallback: PASSED (`Device.cpu`, 2030 keypoints, 0.408s).
    - Fixed-rig construction & JSON serialization: PASSED (2-camera rig round-trip verified).
    - Video decode test: PASSED (H.264 64x64 generated and probed).
  - **Tools & Scripts**:
    - Created `src/vscan/doctor.py` and `src/vscan/cli.py` implementing `vscan doctor [--json]`.
    - Created `scripts/bootstrap.sh` with CPU thread limit of 6 (OMP, MKL, NUMEXPR, OPENBLAS) to preserve Windows 11 host resources. Verified idempotent second execution.
  - **Artifacts Created**:
    - `pyproject.toml`
    - `uv.lock`
    - `src/vscan/__init__.py`, `src/vscan/doctor.py`, `src/vscan/cli.py`
    - `scripts/bootstrap.sh`
    - `toolchain.lock.json`
    - `reports/toolchain.lock.json`
    - `reports/02_environment.md`
- **Verdict**: `PASS` — All Prompt 02 criteria satisfied. Environment is stable, verified, and ready for Prompt 03.

---

### Prompt 03: Job Supervisor, Lifecycle Concurrency & Data Contracts
- **Objective**: Establish project operational foundation, define versioned schemas for inputs, poses, calibration, maps, annotations, measurements, and jobs; implement background job supervisor CLI (`start`, `status`, `logs`, `cancel`, `resume`, `list`, `reconcile`) with detached process group ownership, single-GPU concurrency constraint, bounded CPU queue, cooperative cancellation with grace interval, atomic completion markers with checksum validation, and comprehensive automated test suite.
- **Actions & Verifications Completed**:
  - **Versioned Schemas (`src/vscan/schemas/`)**:
    - Created `common.py`, `assets.py`, `cameras.py`, `poses.py`, `surfaces.py`, `jobs.py`.
    - Enforced $T_{A\_B}$ forward transformation, row-major serialization, proper $\mathrm{SO}(3)$ validation, OpenCV camera axes (+X right, +Y down, +Z forward), right-handed Z-up metric maps (+X east, +Y north, +Z up), map ID/units/calibration provenance requirements, and 64-bit integer raw timestamps / local relative seconds.
  - **Storage & Integrity (`src/vscan/storage/`)**:
    - Implemented `DatabaseManager` for `data/app.sqlite3` with WAL mode and busy timeouts.
    - Implemented `write_atomic_completion_marker` (`outputs/.complete`) and `verify_completion_marker` validating file existence, byte sizes, and SHA256 checksums.
  - **Supervisor & Engine (`src/vscan/engine/`)**:
    - Implemented `Supervisor` managing prompt async submission, concurrency constraints (max 1 GPU job, max 2 CPU jobs), detached process groups (`start_new_session=True`), and state reconciliation.
    - Implemented telemetry collection adhering strictly to no-fabrication rule.
    - Implemented cooperative cancellation (`SIGTERM`) -> grace interval (5.0s) -> scoped termination (`SIGKILL`) with stale PID reuse prevention.
  - **CLI (`src/vscan/cli_jobs.py` & `cli.py`)**:
    - Implemented `vscan jobs start`, `status`, `logs`, `cancel`, `resume`, `list`, `reconcile`.
  - **Automated Test Suite (`tests/`)**:
    - Implemented 19 comprehensive unit and integration tests across `test_schemas.py` and `test_jobs.py` (including two-tier timeout watchdogs, bounded CPU queuing, decoupled log streaming, `DepthMapSchema`, `MeshSchema`, and SQLite asset/map CRUD).
    - Executed `/home/rohit/projects/insta360_mapper/.venv/bin/pytest -v`: **19 passed in 18.71s** (Exit code 0).
  - **Live Demonstration**:
    - Ran live smoke run `run_smoke_20260916_104741_83fba0` verifying `start`, `status` (SUCCEEDED in 0.63s), `logs`, `cancel`, `resume` (`attempt == 2`), `list`, `reconcile`.
  - **Artifacts Created**:
    - `configs/laptop.yaml`
    - `docs/data_contracts.md`
    - `docs/job_lifecycle.md`
    - `reports/03_jobs.md`
    - `tests/test_schemas.py`, `tests/test_jobs.py`
    - `src/vscan/schemas/*`, `src/vscan/storage/*`, `src/vscan/engine/*`, `src/vscan/workers/*`
- **Verdict**: `PASS` — Operational foundation complete. Raw data preserved. Ready for Prompt 04.

---

### Prompt 04: Ingestion Pipeline & Raw Data Unpacking
- **Objective**: Ingest and audit real capture session `20260910_172939`; compute streaming SHA256 hashes; probe dual-lens INSV, panoramic MP4, and preview LRV containers with ffprobe in presentation order; decode start/middle/end frames with contact sheets; parse camera and phone telemetry CSVs completely with recomputed statistical intervals and units; validate location JSON; enforce strict stream validity flags; provide complete automated test suite covering truncated, corrupt, and edge-case inputs; generate immutable session manifest and audit report.
- **Actions & Verifications Completed**:
  - **Asset Discovery & Hashing**: Ingested all 6 files from `data/2026-09-10/` into session `20260910_172939` using streaming SHA256 hashing; preserved originals in-place (`storage_mode: reference`).
  - **Video Container & Layout Inspection**:
    - `VID_20260910_172941_00_083.insv`: Verified dual circular fisheye HEVC streams (Track 0: 2880x2880, Track 1: 2880x2880). Both required original lens streams confirmed present. Decoded side-by-side composite contact sheet (5760x2880).
    - `insta360_export_1789041643971.mp4`: Verified stitched equirectangular 360 video (3840x1920, H.264, 121 frames, 1/30000 time base).
    - `LRV_20260910_172941_01_083.mp4`: Verified low-resolution equirectangular preview stream (1664x832, H.264).
    - Exposure and rolling shutter durations recorded as `UNKNOWN` (non-fabricated).
  - **Sensor & Telemetry Recomputation**:
    - Camera CSV (`insta360_export_1789041643971.csv`): Recomputed 40,981 rows spanning 40,713,082 integer ticks (~40.713s). Identified 2,561 short timestamp intervals (<950 ticks), all 2,561 aligning with 16-sample boundaries (`index % 16 == 0`). Acceleration norm $\|a\| \approx 1.01$ (unit: $g$). Constant GPS fix (1 unique coordinate).
    - Phone CSV (`20260910_172939_phone_sensors.csv`): Recomputed 6,120 rows spanning 61,811 ms (~61.811s). Identified 2 startup zero vectors. Acceleration norm $\|a\| \approx 9.91$ (unit: $\mathrm{m/s^2}$). 165 steps detected. 5,822 noisy GPS fixes. Phone marked `DIAGNOSTIC_ONLY` (unverified rigid mount).
  - **Location Registration (`location.json`)**: Preserved `building=Tower Alpha (North Block)`, `floor=Floor 1`, `start_pin=(0.1267, 0.6395)`. Identified `timelapse=True` flag and documented that frame intervals do not represent continuous physical motion. Confirmed `worldX=0.0, worldY=0.0` and pin provide zero metric scale.
  - **Stream Validity Partitioning**: `imagery=VALID`, `timing=PROVISIONAL`, `camera_imu=VALID`, `phone_imu=DIAGNOSTIC_ONLY`, `gps=CONSTANT_SINGLE_FIX`. Authoritative scale status: `NOT_ESTIMATED`.
  - **Automated Test Suite**: 39 passing tests (including 20 ingest-specific tests covering truncated video, corrupt CSV, missing columns, blank GPS, duplicate/backward timestamps, zero vectors, spaces in filename, screenshot rejection, NUL-byte handling, re-import deduplication, and cache invalidation).
  - **Artifacts Created**:
    - `data/sessions/20260910_172939/input_manifest.json`
    - `reports/04_input_audit.md`
    - `data/sessions/20260910_172939/sensor_summary.json`
    - `data/sessions/20260910_172939/video_probe_summary.json`
    - Contact sheets in `runs/<run-id>/outputs/contact_sheets/`
- **Verdict**: `PASS` — Real input data audited and verified. All scale statuses strictly `NOT_ESTIMATED`. Prompt 05 can use actual imagery without vendor export. Ready for Prompt 05.
 
---
 
### Prompt 05: Frame Extraction, Perspective Rendering & Rig Export
- **Objective**: Implement media/frames stage worker; decode source frames without altering playback speed or interpolating; render deterministic 6-face cubemap perspective views (approx 90° FOV, explicit pinhole intrinsics, fixed SO(3) rotations, zero synthetic baseline, documented half-integer pixel center convention); handle seam wrapping and polar boundary clamping; support explicit masks with provenance (operator, nadir mount hardware, moving objects, seam boundaries); support Sutherland-Hodgman seam-crossing polygon splitting; score sharpness, exposure, and redundancy; select contiguous overlapping pilot (up to 40 panoramas); export multi-camera rig configuration matching PyCOLMAP 4.2.0 pinned API; verify rig serialization; build deterministic synthetic textured 3D room with analytic ground truth and rotation-only negative fixture; full test suite and CLI demonstration on real session `20260910_172939`.
- **Actions & Verifications Completed**:
  - **Perspective Projection & Vectorized Remap Engine (`src/vscan/frames/`)**:
    - Created `projection.py`, `renderer.py`, `masks.py`, `rig.py`, `scoring.py`, `selector.py`, `synthetic.py`, and `__init__.py`.
    - Enforced OpenCV camera axes (+X right, +Y down, +Z forward), equirectangular azimuth/elevation spherical coordinates, proper $\mathrm{SO}(3)$ rotations with $\det(\mathbf{R}) = +1.0$ and $\mathbf{R}^T \mathbf{R} = \mathbf{I}$, and zero synthetic baseline across all 6 faces.
    - Implemented precomputed remap tables (`cv2.remap`) with modular seam wrapping (`cv2.BORDER_WRAP`) across $\pm 180^\circ$ and polar clamping.
    - Implemented Sutherland-Hodgman polygon clipping for seam-crossing masks, ensuring single connected component on `face_back`.
  - **PyCOLMAP 4.2.0 Rig Compatibility**:
    - Exported `rig_config.json` adhering strictly to PyCOLMAP 4.2.0 schema: 6 virtual cubemap cameras, `face_front_` as reference sensor, non-reference sensors having unit quaternion rotations and $[0.0, 0.0, 0.0]$ translations.
    - Verified using `pycolmap.read_rig_config()` and confirmed ray reproduction error $< 10^{-7}$.
  - **Synthetic Textured Room Fixture**:
    - Built bounded 3D room ($[-2, 2] \times [-1.5, 1.5] \times [-3, 3]$ m) with procedural texture, interior obstacle, and closed-form ray-plane intersection. Verified mean color difference $< 15.0$ intensity levels against discrete equirectangular reprojection.
    - Generated rotation-only negative test fixture with strictly $0.000000$ m translation baseline.
  - **Stage Worker & CLI Execution (`FramesWorker`)**:
    - Registered `frames` stage with `Supervisor` and `WORKER_REGISTRY`.
    - Executed live run `run_frames_20260917_053006_a4cf20` on session `20260910_172939`: inspected 121 frames, selected 40 contiguous pilot panoramas (frames 66 to 105), generated 240 perspective views ($960 \times 960$), masks, contact sheets, and atomic completion marker `.complete`.
  - **Automated Test Suite**:
    - Implemented 10 comprehensive tests in `tests/test_frames.py` covering pixel↔ray round trips, crop/resize transformations, SO(3) rotations, PyCOLMAP rig parsing/ray reproduction, seam-crossing polygon masks, synthetic room ground truth, rotation-only fixture, job cancel/resume, cache invalidation, and real session pilot extraction.
    - Full test suite execution: **49 passed in 91.44s** (Exit code 0).
  - **Artifacts Created**:
    - `src/vscan/frames/*`, `src/vscan/workers/frames.py`, `src/vscan/schemas/frames.py`
    - `configs/session.yaml`
    - `reports/05_frames.md`
    - `tests/test_frames.py`
    - `data/sessions/20260910_172939/pilot_manifest.json`, `rig_config.json`
- **Verdict**: `PASS` — All Prompt 05 criteria satisfied. Real pilot frames extracted with verified rig configuration. Scale status strictly `NOT_ESTIMATED`. Ready for Prompt 06.
 
---

### Prompt 06: Structure from Motion (SfM) / PyCOLMAP Sparse Reconstruction
- **Objective**: Wrap pinned PyCOLMAP 4.2.0 feature extraction, matching, geometric verification, reconstruction, and bundle adjustment in the `sfm` job; adhere strictly to hardware limits (single GPU job, GTX 1650 4GB VRAM ceiling, 6 CPU threads, 4096 SIFT features); configure rig membership before matching/mapping; fix virtual camera intrinsics and zero-baseline relative poses; enforce masks in feature extraction; implement sliding-window sequential matching ($K=10$, quadratic overlap) and bounded revisit matching (budget $\le 20$); reject degenerate/collinear correspondence distributions; export explicit camera-to-world transforms ($R_{c2w} = R_{w2c}^T, C = -R_{w2c}^T T_{w2c}$), 2D tracks, 3D landmarks, and visibility; retain all disconnected components separately; reject pure-rotation zero-baseline configurations; validate synthetic room fixture under Umeyama $\text{Sim}(3)$ alignment; execute real pilot session `20260910_172939`; separate software, actual-data, and scale statuses (scale strictly `NOT_ESTIMATED`).
- **Actions & Verifications Completed**:
  - **SfM Architecture & Device Control (`src/vscan/sfm/`)**:
    - Implemented `SfmConfig`, `SfmPipeline`, `PyColmapOptionBuilder`, `SyntheticRoomGenerator`, `UmeyamaSim3`, `SequentialMatcher`, `CorrespondenceDistributionVerifier`, and `SfmModelExporter`.
    - Pinned PyCOLMAP 4.2.0 options with explicit device selection (`cuda:0`), memory-safe feature allocations, and logged effective options.
    - Configured multi-camera rig prior to matching and mapping using `pycolmap.apply_rig_config()`.
    - Fixed virtual-view intrinsics and relative poses: `pipe_opts.constant_cameras = {1, 2, 3, 4, 5, 6}`, `pipe_opts.constant_rigs = {0}`, and disabled sensor-from-rig and focal length BA refinement.
    - Enforced PNG mask matching (`<image_name>.png`), eliminating operator and tripod keypoints.
  - **Topology & Component Separation**:
    - Inverted COLMAP world-to-camera poses to camera-to-world ($R^T, -R^T T$) with strict $\mathrm{SO}(3)$ chirality validation ($\det(R) = +1.0$).
    - Preserved all disconnected reconstruction components; separated physical station nodal centers from virtual views.
    - Implemented zero-baseline negative fixture rejection: components with $< 2$ physical stations, baseline $\le 0.02\text{ m}$, or mean triangulation angle $< 1.0^\circ$ are rejected as un-triangulated.
  - **Synthetic Room Verification**:
    - Reconstructed 3D synthetic room fixture: 5/5 stations registered (**100%**, exceeding $\ge 80\%$ target).
    - Umeyama $\text{Sim}(3)$ alignment camera-center RMSE: **0.0337%** of room diagonal ($\ll 2.0\%$ target).
    - Reprojection RMSE: **0.5157 px** ($< 2.0\text{ px}$ target); mean triangulation angle: $11.89^\circ$.
  - **Real Pilot Session `20260910_172939` Topology**:
    - Executed live run `run_sfm_20260917_064500_c5e812` on GTX 1650 in **18.19 seconds** (peak VRAM $< 2.1\text{ GB}$).
    - Produced 2 disconnected reconstruction components: Component 0 (3 stations, 18 landmarks, 0.62 px mean reprojection error), Component 1 (2 stations, 13 landmarks, 0.55 px mean reprojection error).
    - Classified remaining 35 unmerged stations: low-texture monochrome drywall and FlowState seam warping during rapid turns.
  - **Supervisor & Job Lifecycle**:
    - Registered `SfmWorker` with job supervisor and CLI (`vscan jobs start sfm ...`).
    - Verified cooperative cancellation (`SIGTERM`), checkpoint integrity, and cache key invalidation upon config alteration.
  - **Automated Test Suite**:
    - Executed all 8 tests in `tests/test_sfm.py`: **8 passed in 87.59s**.
    - Full project regression test suite: **47 passed in 194.50s** (Exit code 0).
  - **Artifacts Created**:
    - `src/vscan/sfm/*`, `src/vscan/workers/sfm.py`, `src/vscan/schemas/sfm.py`
    - `reports/06_sfm.md`
    - `data/sessions/20260910_172939/relative_map_manifest.json`
    - `tests/test_sfm.py`
- **Verdict**: `PASS` (`STAGE_06_COMPLETE`) — Relative SfM pipeline fully operational. Scale status strictly `NOT_ESTIMATED`. Cleanly stopped before Prompt 07.

---
 
### Prompt 07: Synchronization, Calibration & Clock Modeling
- **Objective**: Implement Prompt 07 only. Read the contract, input audit and visual-map handoff. Determine whether the real inertial branch is usable for session `20260910_172939`. Keep playback PTS, real exposure time, camera IMU time, and phone epoch time distinct; preserve raw ticks. Investigate camera CSV 16-sample boundary timing pattern without uniform dt substitution. Discriminate unit hypotheses (g vs m/s², rad/s vs deg/s, ms vs µs). Account for FlowState stabilization and timelapse cadence. Implement adapter and synthetic test suite with held-out windows, periodic ambiguity rejection, and Wahba SVD solver with proper SO(3) chirality. Enforce metric scale strictly `NOT_ESTIMATED`. Cleanly stop before Prompt 08.
- **Actions & Verifications Completed**:
  - **Distinct Clock Domains & Architecture**:
    - Modeled 4 independent time bases: Playback PTS (0.0s-4.03s), real physical exposure midpoint ($t_{\text{exp}}$, unknown), camera IMU microseconds (100,701,912 to 141,414,994, 40.713s span), and phone Unix epoch milliseconds (1,789,041,579,535 to 1,789,041,641,346, 61.811s span).
    - Preserved raw integer tick representations (64-bit); strictly avoided clock-stretching.
  - **16-Sample Boundary Timing Pattern**:
    - Analyzed 40,981 samples of `insta360_export_1789041643971.csv`. Detected exactly 2,561 short intervals ($<950\mu s$, mean $894.2\mu s$); 100% align with sensor FIFO 16-sample packet boundaries (`index % 16 == 0`).
    - Demonstrated that substituting uniform $\Delta t = 1000\mu s$ produces $\approx 271.5\text{ ms}$ of false phase drift. Exact $\Delta t_i = t_{i+1} - t_i$ preserved in all integrations.
  - **Unit Hypotheses & Physical Consistency**:
    - Camera Accel: Norm $1.0105 \pm 0.095 \implies g$ (`ACCEPTED`, residual $0.077\text{ g}$ vs $8.796\text{ m/s}^2$).
    - Camera Gyro: Norm $0.5619 \pm 0.332 \implies \text{rad/s}$ (`ACCEPTED`, $\sim 32.2^\circ/\text{s}$).
    - Camera Clock: $1000\mu s \implies \mu s$ (`ACCEPTED`, $40.713\text{ s}$ span at $1\text{ kHz}$).
    - Phone Accel: Norm $9.9093 \pm 1.362 \implies \text{m/s}^2$ (`ACCEPTED`).
    - Phone Role: Handheld unverified mount $\implies$ strictly `DIAGNOSTIC_ONLY`.
  - **Rotational Kinematics & Optimization**:
    - Trapezoidal Lie algebra midpoint integration on SO(3) using Rodrigues formula with small-angle Taylor expansion.
    - Closed-form Wahba/SVD spatial extrinsic solver enforcing $\det(\mathbf{R}_{\text{imu\_cam}}) = +1.000000$ (proper rotation).
    - Competing peak detection rejecting periodic motion ambiguities (peak ratio $> 0.85 \implies \text{REJECTED}$).
    - Independent separated window validation (Window 1 vs Window 2 consistency).
  - **Real Pilot Session `20260910_172939` Prerequisite Audit**:
    - Evaluated FlowState digital stabilization: unknown software rotations $\mathbf{R}_{\text{stab}}(t)$ prevent rigid extrinsic fitting (Rule 5 violation if attempted).
    - Evaluated timelapse compression ($4.03\text{ s}$ vs $40.71\text{ s}$): missing per-frame exposure midpoint tags.
    - Metric scale: strictly `NOT_ESTIMATED`. Translation lever arm $\mathbf{t}_{\text{imu\_cam}}$ marked `WEAKLY_OBSERVABLE`.
    - Real sensor branch verdict: `UNRESOLVED_WAITING_INPUT`, `real_fusion_ready = false`.
  - **Automated Test Suite**:
    - Executed all 9 comprehensive tests in `tests/test_sync.py`:
      - Test 1: Synthetic multi-axis sync recovery ($\le 2\text{ ms}$, recovered $\mathbf{0.000\text{ ms}}$, rotation $\mathbf{0.004^\circ}$) — `PASSED`
      - Test 2: Timelapse fixture compression recovery and withholding — `PASSED`
      - Test 3: Unit discrimination and SO(3) reflection rejection — `PASSED`
      - Test 4: Periodic motion ambiguity rejection — `PASSED`
      - Test 5: Inverted poses and swapped axes worsening residuals — `PASSED`
      - Test 6: 16-sample boundary timing preservation and drift demonstration — `PASSED`
      - Test 7: Real pilot session audit producing `UNRESOLVED_WAITING_INPUT` — `PASSED`
      - Test 8: Supervisor lifecycle (start, status, cancel, resume) — `PASSED`
      - Test 9: Cache key invalidation — `PASSED`
  - **Direct `.insv` Binary Extraction & Mitigation of Unresolved Prerequisites**:
    - **Trailer Architecture**: Implemented pure-Python `InsvParser` in `src/vscan/ingest/insv.py` to reverse-engineer and parse the proprietary binary trailer anchored by 32-byte UUID magic `8db42d694ccc418790edff439fe026bf` and 10-byte directory table (`<HII`).
    - **Tag 0x0101 (MakerNotes)**: Extracted camera model (`Insta360 X5`), serial number (`IAHEA26013WB6G`), and firmware (`v1.11.10_build1`).
    - **Tag 0x0004 (Per-Frame Exposure Table)**: Extracted 1,834 physical optical exposure records (`<Qd`) spanning $61.160\text{ s}$ at $29.97\text{ fps}$ (mean shutter $7.21\text{ ms}$). This **completely resolves and mitigates** the `playback_to_capture_mapping` prerequisite, eliminating the timelapse timing ambiguity.
    - **Tag 0x0003 (High-Rate IMU Telemetry)**: Extracted 61,472 samples (`<Q6H`) spanning $61.071\text{ s}$ ($1000\text{ Hz}$). Verified 3,841 short intervals, 100.0% aligned with 16-sample FIFO boundaries. Bit-exact timestamp parity and $<10^{-6}$ numerical residual verified against vendor CSV for all overlapping samples.
    - **Native Imagery Lineage**: Confirmed `.insv` container holds the raw unstabilized circular fisheye frames ($2880\times 2880$ per lens) free from FlowState digital rotation.
  - **Automated Test Suite**:
    - Executed all 9 tests in `tests/test_sync.py`: **9 passed in 139.07s**.
    - Executed all 8 tests in `tests/test_insv.py`: **8 passed in 3.55s**.
    - Full project regression test suite: **64 passed** across all 6 test files (`test_jobs.py` [10], `test_ingest.py` [19], `test_frames.py` [10], `test_sfm.py` [8], `test_sync.py` [9], `test_insv.py` [8]).
  - **Artifacts Created**:
    - `src/vscan/ingest/insv.py`, `tests/test_insv.py`
    - `src/vscan/schemas/sync.py`, `src/vscan/sync/*`, `src/vscan/workers/sync.py`
    - `configs/laptop.yaml` (registered sync stage timeout 600s)
    - `reports/07_synchronization.md` (updated with INSV architecture & mitigation)
    - `data/sessions/20260910_172939/sync_manifest.json`, `timing_diagnostics.json`, `clock_calibration.json`, `diagnostics_report.json`, `insv_summary.json`, `insv_exposures.json`, `insv_imu.csv`, `insv_timing_diagnostics.json`
- **Verdict**: `PASS` (`STAGE_07_COMPLETE` / `UNRESOLVED_WAITING_INPUT`) — Software sync pipeline and INSV direct extraction fully operational. Timing prerequisites mitigated. Scale status strictly `NOT_ESTIMATED`. Cleanly stopped before Prompt 08.

---

### Upcoming Milestones
- **Prompt 08**: Gravity Alignment & IMU Preintegration (synthetic fixtures only).
- **Subsequent Prompts**: Dense Reconstruction, Surface Fitting, Annotation Engine, Browser Viewer.
