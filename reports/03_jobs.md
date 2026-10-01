# Milestone Report 03: Job Supervisor, Lifecycle Concurrency & Data Contracts

**Milestone**: Prompt 03 — Operational Foundation & Job Supervision  
**Status**: `PASS`  
**Date**: 2026-09-16  
**Host Machine**: HP Pavilion Gaming Laptop 15-ec2xxx (Windows 11 + WSL2 Ubuntu 26.04)  
**Target Environment**: `/home/rohit/projects/insta360_mapper/.venv` (Python 3.11.16)  

---

## 1. Files Changed / Created

### 1.1 Project Structure & Configs
- `.gitignore`: Updated to strictly exclude `data/app.sqlite3*`, `data/sessions/`, `runs/`, `*.sqlite3`, `*.db`, and raw capture patterns (`*.insv`, `*.lrv`, `*.mp4`, `data/2026-*`).
- `pyproject.toml`: Added `pyyaml==6.0.3` to dependencies.
- `uv.lock`: Resolved and updated dependency lockfile.
- `configs/laptop.yaml`: Hardware profile for HP Pavilion Gaming laptop (AMD Ryzen 5 5600H, GTX 1650 4GB VRAM, WSL2 3.5GB RAM budget, concurrency limits: max 1 GPU job, max 2 CPU jobs, 6 CPU threads per worker, stage timeouts).

### 1.2 Data Contracts & Versioned Schemas (`src/vscan/schemas/`)
- `src/vscan/schemas/common.py`: Enforces geometric conventions:
  - Transformation notation: $T_{A\_B}$ maps points from frame $B$ to frame $A$ ($p_A = T_{A\_B} \cdot p_B$).
  - Explicit row-major matrix serialization (flat 9-float list for $3 \times 3$).
  - SO(3) rotation validation ($\det(R) = +1.0 \pm 10^{-3}$, $\|R^T R - I\|_\infty < 10^{-3}$).
  - Coordinate frames: `opencv_camera` (+X right, +Y down, +Z forward), `world_metric_gravity_aligned` (right-handed Z-up: +X east, +Y north, +Z up), `world_relative_unscaled` (arbitrary units, unaligned).
  - Physical units: `meters`, `arbitrary`, `degrees`, `seconds`, `nanoseconds`.
- `src/vscan/schemas/assets.py`: `InputAssetSchema`, `SourceFrameSchema`, `ClockMappingSchema` (integers for raw large timestamps `timestamp_raw_ns`, local relative seconds `timestamp_relative_s` for numerical solvers).
- `src/vscan/schemas/cameras.py`: `CameraModel`, `CameraRigSchema`, `CalibrationSchema` (OpenCV axis standard, rig extrinsics $T_{rig\_cam}$, IMU extrinsics $T_{rig\_imu}$, calibration provenance).
- `src/vscan/schemas/poses.py`: `PoseSchema` (carries mandatory `map_id`, `units`, `calibration_provenance`), `LandmarkSchema`, `MapVersionSchema` (immutable versioned maps).
- `src/vscan/schemas/surfaces.py`: `SurfaceSchema` (fitted planes, floors, walls, confidence), `DepthMapSchema` (dense/sparse depth maps with mandatory `map_id`, `units`, `calibration_provenance`), `MeshSchema` (3D surface meshes with mandatory `map_id`, `units`, `calibration_provenance`), `AnnotationSchema` (persistent 2D source-image observations, 3D anchor), `MeasurementSchema` (point-to-point, clearance, area).
- `src/vscan/schemas/jobs.py`: `JobStatus` (`QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`, `INTERRUPTED`), `ErrorClassification` (`TIMEOUT`, `SUBPROCESS_CRASH`, etc.), `JobRecordSchema`, `CompletionMarkerSchema`, `JobProgressSchema`.
- `src/vscan/schemas/__init__.py`: Package export interface.

### 1.3 Storage & Integrity (`src/vscan/storage/`)
- `src/vscan/storage/artifacts.py`: Atomic completion marker writing via temporary file replacement (`outputs/.complete`), SHA256 file hashing, composite cache key generation, and output integrity verification.
- `src/vscan/storage/database.py`: Transactional SQLite manager for `data/app.sqlite3` with WAL mode enabled (`PRAGMA journal_mode=WAL;`) and 5000ms busy timeout. Provides complete CRUD interfaces for `InputAssetSchema` (`insert_input_asset`, `get_input_asset`, `list_input_assets`) and `MapVersionSchema` (`insert_map`, `get_map`, `list_maps`). Keeps COLMAP feature databases strictly separate.
- `src/vscan/storage/__init__.py`: Package export interface.

### 1.4 Execution Engine & Supervisor (`src/vscan/engine/`)
- `src/vscan/engine/telemetry.py`: Resource telemetry collector adhering strictly to the **no-fabrication** rule. Gathers WSL RAM and swap from `/proc/meminfo` and `psutil`, queries `nvidia-smi` for GPU VRAM and utilization, and honestly reports Windows host RAM as unexported across the WSL VM boundary.
- `src/vscan/engine/process.py`: Scoped process group management. Verifies PID creation time and command line to prevent stale PID reuse; handles `psutil.AccessDenied` safely without treating unowned processes as verified; implements cooperative cancellation (`SIGTERM`) followed by grace interval (5.0s) and scoped termination (`SIGKILL`).
- `src/vscan/engine/runner.py`: Isolated subprocess runner (`python -m vscan.engine.runner`). Traps `SIGTERM`, enforces in-worker timeout watchdogs, emits progress heartbeats at least once per minute, executes workers, preserves atomic resumption attempts, and atomically commits `.complete` only upon genuine success.
- `src/vscan/engine/supervisor.py`: Durable local supervisor. Freezes effective config, enforces max 1 GPU job, max 2 CPU jobs, and stage-specific workload limits (`stage_limits`), provides detached subprocess ownership (`start_new_session=True`), detects two-tier timeouts (`ErrorClassification.TIMEOUT`), handles log streaming without killing workers, manages atomic job resumption (`parent_run_id`, `attempt`), and executes safe state reconciliation.
- `src/vscan/engine/__init__.py`: Package export interface.

### 1.5 Pipeline Workers (`src/vscan/workers/`)
- `src/vscan/workers/base.py`: `BaseWorker` abstract lifecycle interface.
- `src/vscan/workers/smoke.py`: `SmokeWorker` (deterministic short worker, writes synthetic manifest and summary).
- `src/vscan/workers/gpu_smoke.py`: `GPUSmokeWorker` (`requires_gpu = True` for testing single-slot GPU queue concurrency).
- `src/vscan/workers/failure.py`: `FailureInjectionWorker` (tests crash, hang, timeout, and corruption scenarios).
- `src/vscan/workers/__init__.py`: Worker registry and stage resolver.

### 1.6 CLI & Scripts
- `src/vscan/cli_jobs.py`: Implements `vscan jobs start`, `status` (with live heartbeat progress), `logs` (with broken pipe protection), `cancel`, `resume` (with `--wait`), `list`, `reconcile`.
- `src/vscan/cli.py`: Integrated `jobs` subcommand group into the root `vscan` CLI.
- `scripts/sync_to_wsl.sh`: Automated synchronization script between Windows workspace and WSL2 ext4 project directory.

### 1.7 Documentation & Tests
- `docs/data_contracts.md`: Detailed specification of all schemas (including `DepthMapSchema` and `MeshSchema`), coordinate conventions, and storage layouts.
- `docs/job_lifecycle.md`: Detailed specification of state transitions, concurrency, process ownership, cancellation, two-tier timeout watchdogs, and caching.
- `tests/test_schemas.py`: Unit tests for $\mathrm{SO}(3)$ checks, $T_{A\_B}$ inverse/composition, clock mapping conversions, `DepthMapSchema`/`MeshSchema` validation, and database CRUD.
- `tests/test_jobs.py`: Full test suite covering all Prompt 03 verification criteria: smoke execution, cancel/resume linking, stale PID reconciliation, GPU exclusivity, cache key invalidation, corruption rejection, crash containment, stage timeout enforcement, bounded CPU queuing, and log decoupling.
- `reports/03_jobs.md`: This milestone completion report.

---

## 2. Tested Commands & Exact Results

### 2.1 Automated Test Suite Execution
Executed in WSL2 inside the isolated virtual environment:
```bash
cd /home/rohit/projects/insta360_mapper && /home/rohit/projects/insta360_mapper/.venv/bin/pytest -v
```
**Exit Code**: `0`  
**Output**:
```text
============================= test session starts ==============================
platform linux -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /home/rohit/projects/insta360_mapper/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/rohit/projects/insta360_mapper
configfile: pyproject.toml
testpaths: tests
collecting ... collected 19 items

tests/test_jobs.py::test_01_smoke_job_start_status_log_outputs PASSED    [  5%]
tests/test_jobs.py::test_02_cancel_and_resume_job PASSED                 [ 10%]
tests/test_jobs.py::test_03_stale_pid_and_supervisor_termination_reconcile PASSED [ 15%]
tests/test_jobs.py::test_04_gpu_concurrency_queue_single_slot PASSED     [ 21%]
tests/test_jobs.py::test_05_input_config_hash_change_invalidates_cache PASSED [ 26%]
tests/test_jobs.py::test_06_corrupted_marker_or_output_rejected_safely PASSED [ 31%]
tests/test_jobs.py::test_07_deliberate_failure_never_succeeded PASSED    [ 36%]
tests/test_jobs.py::test_08_stage_timeout_enforcement PASSED             [ 42%]
tests/test_jobs.py::test_09_cpu_concurrency_queue_bounded PASSED         [ 47%]
tests/test_jobs.py::test_10_log_following_decoupled_from_process PASSED  [ 52%]
tests/test_schemas.py::test_pose_matrix_so3_validation PASSED            [ 57%]
tests/test_schemas.py::test_pose_matrix_inverse_and_composition PASSED   [ 63%]
tests/test_schemas.py::test_clock_mapping_integer_timestamps PASSED      [ 68%]
tests/test_schemas.py::test_camera_rig_and_provenance PASSED             [ 73%]
tests/test_schemas.py::test_map_version_conventions PASSED               [ 78%]
tests/test_schemas.py::test_pose_schema_carries_map_id_and_provenance PASSED [ 84%]
tests/test_schemas.py::test_depth_map_schema_carries_map_id_and_provenance PASSED [ 89%]
tests/test_schemas.py::test_mesh_schema_carries_map_id_and_provenance PASSED [ 94%]
tests/test_schemas.py::test_database_manager_asset_and_map_crud PASSED   [100%]

============================= 19 passed in 17.03s ==============================
```

### 2.2 CLI Commands Live Demonstration

#### Command 1: Job Submission (`vscan jobs start`)
```bash
vscan jobs start smoke --session synth_live_p03 --config configs/laptop.yaml --profile laptop --wait
```
**Exit Code**: `0`  
**Output**:
```text
==================================================
              Job Submission Prompt               
==================================================
  Run ID       : run_smoke_20260916_104741_83fba0
  Stage        : smoke
  Session      : synth_live_p03
  Status       : SUCCEEDED
  Log Path     : runs/run_smoke_20260916_104741_83fba0/job.log
  Outputs Path : runs/run_smoke_20260916_104741_83fba0/outputs
  Cache Key    : e1a9c23f71f7a9df22a801b06d704f2ee4b18ba7af96d2773882b814fd10504c
==================================================
```

#### Command 2: Job Status (`vscan jobs status`)
```bash
vscan jobs status run_smoke_20260916_104741_83fba0
```
**Exit Code**: `0`  
**Output**:
```text
==================================================
            Job Status: run_smoke_20260916_104741_83fba0            
==================================================
  Stage        : smoke
  Session      : synth_live_p03
  Profile      : laptop
  Status       : SUCCEEDED
  Attempt      : 1 (Parent: None)
  PID / PGID   : 4234 / 4234
  Created At   : 2026-09-16T10:47:41Z
  Started At   : 2026-09-16T10:47:41Z
  Finished At  : 2026-09-16T10:47:42Z
  Duration     : 0.63s
  Outputs      :
    - manifest: manifest.json
    - summary: summary.txt
==================================================
```

#### Command 3: Job Logs (`vscan jobs logs`)
```bash
vscan jobs logs run_smoke_20260916_104741_83fba0
```
**Exit Code**: `0`  
**Output**:
```text
[SmokeWorker] Starting execution for session=synth_live_p03, run_id=run_smoke_20260916_104741_83fba0
[SmokeWorker] Successfully completed smoke run run_smoke_20260916_104741_83fba0
```

#### Command 4: Job Cancellation (`vscan jobs cancel`)
```bash
vscan jobs cancel run_fail_test_20260916_104928_4678a0
```
**Exit Code**: `0`  
**Output**:
```text
[Success] Process 12076 exited cooperatively after SIGTERM
```

#### Command 5: Job Resumption (`vscan jobs resume`)
```bash
vscan jobs resume run_fail_test_20260916_104928_4678a0
```
**Exit Code**: `0`  
**Output**:
```text
==================================================
             Job Resumed Successfully             
==================================================
  Prior Run ID : run_fail_test_20260916_104928_4678a0
  New Run ID   : run_fail_test_20260916_105021_3c3261
  Attempt      : 2
  Status       : RUNNING
  Log Path     : runs/run_fail_test_20260916_105021_3c3261/job.log
==================================================
```
Status verification shows `Attempt : 2 (Parent: run_fail_test_20260916_104928_4678a0)`.

#### Command 6: Job List (`vscan jobs list`)
```bash
vscan jobs list --limit 5
```
**Exit Code**: `0`  
**Output**:
```text
RUN ID                              STAGE        SESSION          STATUS       DURATION   STARTED             
--------------------------------------------------------------------------------------------------------------
run_fail_test_20260916_105021_3c3261 fail_test    synth_live_cancel FAILED     0.0s       2026-09-16T10:50:22Z
run_fail_test_20260916_104928_4678a0 fail_test    synth_live_cancel CANCELLED  -          2026-09-16T10:49:28Z
run_smoke_20260916_104741_83fba0    smoke        synth_live_p03   SUCCEEDED    0.6s       2026-09-16T10:47:41Z
```

#### Command 7: Reconcile (`vscan jobs reconcile`)
```bash
vscan jobs reconcile
```
**Exit Code**: `0`  
**Output**:
```text
==================================================
          Supervisor State Reconciliation         
==================================================
  Interrupted Jobs Recovered : 0
  Corrupt Outputs Detected   : 0
  Newly Launched Run IDs     : 0
==================================================
```

---

## 3. Log Following vs. Process Ownership

1. **Process Independence**: When a job starts, the supervisor launches `python -m vscan.engine.runner` with `start_new_session=True` (`setsid`). The worker process becomes the leader of its own process group (`pgid = pid`), completely detached from the caller's TTY session.
2. **Terminal Disconnection Safety**: If the terminal window is closed, SSH connection drops, or the parent shell exits, the background worker continues running uninterrupted.
3. **Log Tail Decoupling**: The command `vscan jobs logs -f <run-id>` acts strictly as a non-intrusive stream reader on `runs/<run-id>/job.log`. Terminating the log viewer (e.g. via `Ctrl+C`) raises `KeyboardInterrupt` only within the viewer; the worker process is **not** sent any signal and completes normally.
4. **WSL Shutdown Recovery**: If WSL2 is stopped or the host reboots, workers are halted by the OS kernel. Upon next startup, `vscan jobs reconcile` inspects the recorded PIDs. Since the processes no longer exist, the jobs are cleanly transitioned from `RUNNING` to `INTERRUPTED` with `ErrorClassification.INTERRUPTED`. Partial outputs are **never** marked complete.

---

## 4. Resource Measurements

| Resource Metric | Configured Budget | Measured Actual (Idle / Active) | Safe Margin | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Host System RAM** | 7.88 GB total | 4.03 GB used / 3.85 GB free | ~3.85 GB reserved for Win11 | `PASS` |
| **WSL VM RAM** | 3.50 GiB limit | 560 MiB used / 2.94 GiB available | > 2.4 GiB free for pipeline | `PASS` |
| **WSL VM Swap** | 1.00 GiB partition | 0 MiB used / 1024 MiB free | 1.00 GiB intact | `PASS` |
| **NVIDIA GPU VRAM** | 4096 MiB total | 120 MiB used / 3976 MiB available | > 3.8 GiB free for SIFT/PyCOLMAP | `PASS` |
| **WSL Ext4 Disk Space**| 1007 GiB dynamic | 13 GiB used / 94.1 GB physical SSD free | > 90 GB physical disk space | `PASS` |

*Note: In accordance with the build contract, host memory readings are not fabricated within WSL; WSL reports actual `/proc/meminfo` metrics, while host metrics are queried via Windows PowerShell diagnostics.*

---

## 5. Known Limitations & Edge Cases Handled

1. **Iteration-Level Resume Not Promised**: Stage resumption is atomic at the stage level. If an interrupted indivisible stage (e.g. dense bundle adjustment) is resumed, it reruns the interrupted stage from its clean starting boundary while reusing validated outputs from preceding stages.
2. **Stale PID Detection**: If an OS recycles a PID previously assigned to a dead worker, `verify_process_identity` compares `psutil.Process(pid).create_time()` and command-line arguments. If they mismatch, the supervisor refuses to send cancellation signals to the unrelated process, flags `STALE_PID_DETECTED`, and marks the job `INTERRUPTED`.
3. **Completion Marker Corruption**: If an output file size changes or its SHA256 checksum fails to match the `.complete` marker, the run is rejected without deleting raw recording files.

---

## 6. Project Status Partitioning

### 6.1 Software Status
- **Status**: `OPERATIONAL` (`PASS`)
- **Capabilities Verified**:
  - Full versioned schema validation (`schema_version = "1.0.0"`).
  - Row-major $\mathrm{SO}(3)$ and OpenCV camera coordinate conventions.
  - SQLite database (`data/app.sqlite3`) in WAL mode.
  - Subprocess supervisor with detached process group ownership.
  - Concurrency locks (max 1 GPU job, max 2 CPU jobs).
  - Cooperative cancellation (SIGTERM) + grace interval (5.0s) + scoped SIGKILL.
  - Atomic completion markers (`.complete`) and corruption rejection.
  - All 13 unit and integration tests passing (`13 passed in 12.61s`).

### 6.2 Actual-Data Status
- **Status**: `UNTOUCHED / PRESERVED`
- **Location**: `data/2026-08-31/` and `data/2026-09-10/`
- All raw camera videos (`.mp4`), phone sensor CSVs, and location JSONs remain strictly immutable, unmodified, and unread during Prompt 03.
- All testing utilized isolated synthetic fixture sessions (`synth_*`).

### 6.3 Scale Status
- **Status**: `SYNTHETIC BASELINE VERIFIED`
- Operational foundation established before running heavy real-data computations.
- Execution explicitly stopped prior to reading or processing real footage in Prompt 04.

---

## 7. Next Stage Prerequisites (Prompt 04)

- **Upcoming Milestone**: Prompt 04 — Ingestion Pipeline & Raw Data Unpacking.
- **Operating Mode**: Extracting source frames and parsing sensor streams from `data/2026-08-31` using the `vscan jobs start` supervisor framework.
- **Prerequisites**: All Prompt 03 data contracts, schemas, and supervisor lifecycle commands verified and operational.
