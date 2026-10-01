# Job Lifecycle & Execution Architecture Specification (v2)

**Document**: `docs/job_lifecycle.md`  
**Application**: `vscan` Background Supervisor Engine  
**Version**: `1.0.0`  

---

## 1. Lifecycle State Machine

Every background job managed by `vscan` exists in exactly one of the six authoritative lifecycle states:

```text
    ┌───────────────────────────┐
    │          QUEUED           │◄─────────────────────┐
    └─────────────┬─────────────┘                      │
                  │ (Concurrency slot available)       │
                  ▼                                    │
    ┌───────────────────────────┐                      │
    │          RUNNING          │                      │
    └──────┬──────┬──────┬──────┘                      │
           │      │      │                             │
 (Success) │      │      │ (SIGTERM/User Cancel)       │ (Resume)
           ▼      │      ▼                             │
    ┌──────────┐  │ ┌───────────┐                      │
    │SUCCEEDED │  │ │ CANCELLED │──────────────────────┤
    └──────────┘  │ └───────────┘                      │
                  │                                    │
    (Worker Error)│      (Supervisor Crash/WSL Reboot) │
                  ▼      ▼                             │
            ┌─────────┐ ┌─────────────┐                │
            │ FAILED  │ │ INTERRUPTED │────────────────┘
            └─────────┘ └─────────────┘
```

### 1.1 State Definitions
| State | Description | Reusable? | Outputs Validated? |
| :--- | :--- | :---: | :---: |
| `QUEUED` | Job submitted, effective config frozen, input hashes computed, waiting for resource slot. | No | N/A |
| `RUNNING` | Subprocess executing in detached process group; live heartbeat and progress reported. | No | Partial / Temporary |
| `SUCCEEDED` | Subprocess exited with code 0; atomic completion marker `.complete` verified with sha256 checksums. | **Yes** | **Enforced** |
| `FAILED` | Subprocess exited with non-zero exit code or raised unhandled exception. | No | Never marked complete |
| `CANCELLED` | Subprocess terminated cooperatively via `vscan jobs cancel` or SIGTERM/SIGKILL escalation. | No | Never marked complete |
| `INTERRUPTED` | Subprocess terminated unexpectedly (WSL shutdown, host reboot, supervisor termination, stale PID). | No | Never marked complete |

---

## 2. Process Ownership & Log Streaming Architecture

### 2.1 Detached Process Group Ownership
- When `vscan jobs start` executes, it forks the runner into a detached session (`start_new_session=True` / `os.setsid()` on POSIX, `CREATE_NEW_PROCESS_GROUP` on Windows).
- The worker's process group ID (`pgid`) is isolated from the interactive CLI shell.
- Output streams (`stdout` and `stderr`) are redirected directly to `runs/<run-id>/job.log`.
- **Terminal Disconnect Invariance**: If the invoking terminal session closes, SSH drops, or a tmux/screen session exits, the worker subprocess continues running unaffected.
- **Log Streaming Decoupling**: Executing `vscan jobs logs -f <run-id>` opens a read-only tail reader on `runs/<run-id>/job.log`. Pressing `Ctrl+C` terminates only the log viewer process; the background worker subprocess continues executing without interruption.

### 2.2 WSL Interruption Recovery
- If the WSL2 virtual machine terminates or restarts (`wsl --shutdown`), running processes vanish.
- Upon next invocation of `vscan jobs reconcile` or `vscan jobs status`, the supervisor detects the absence of the recorded PID.
- The job state transitions safely to `INTERRUPTED`. Partial or interrupted outputs are **never** marked complete.

---

## 3. Concurrency Model & Resource Safeguards

Designed specifically for the HP Pavilion Gaming Laptop (AMD Ryzen 5 5600H, 6 cores/12 threads, GTX 1650 4GB VRAM, 3.5 GiB WSL memory allocation):

### 3.1 Strict GPU Concurrency: Max 1 GPU Job
- Any stage declaring `requires_gpu = True` (e.g. `gpu_smoke`, feature extraction) must acquire the project's exclusive GPU slot.
- If a GPU job is currently `RUNNING`, all subsequent GPU jobs remain in `QUEUED` state.
- GPU jobs cannot starve the system: when the active GPU job finishes, the supervisor queue dispatcher dequeues the next eligible job.

### 3.2 Bounded CPU Worker Queue
- Bounded to a maximum of **2 concurrent CPU jobs** (`max_cpu_jobs: 2`).
- Each worker subprocess is bound to **6 CPU worker threads** via environment variables:
  `OMP_NUM_THREADS=6`, `MKL_NUM_THREADS=6`, `NUMEXPR_NUM_THREADS=6`, `OPENBLAS_NUM_THREADS=6`.
- Preserves 3.85 GiB host RAM for Windows 11 host responsiveness and prevents WSL OOM (Out Of Memory) crashes within the 3.5 GiB WSL budget.

### 3.3 Two-Tier Stage Timeout Enforcement
Stage execution timeouts (e.g. `smoke: 30s`, `unpack: 600s`, `features: 1200s`, `sfm: 1800s`) are enforced by two complementary watchdogs:
1. **In-Worker Watchdog Thread**: The runner executes a background watchdog thread checking `time.time() - start_time >= stage_timeout`. Upon timeout, it sets `self.timed_out = True`, flags cancellation, and exits with code 124, persisting `ErrorClassification.TIMEOUT`.
2. **Authoritative Supervisor Watchdog**: During `vscan jobs reconcile` or `advance_queue()`, if a worker process has exceeded its configured timeout plus grace buffer, the supervisor directly signals the owned process group, terminates it, and transitions the state to `FAILED` with `ErrorClassification.TIMEOUT`.

### 3.4 Stage-Specific Workload Limits (`stage_limits`)
In addition to global concurrency limits, stages declare stage-specific workload limits in `configs/laptop.yaml`:
- Per-stage concurrency bounds (`max_concurrent: 1` for heavy GPU/SfM tasks).
- Per-stage CPU thread allocations (`cpu_threads: 2` for light workers, `cpu_threads: 6` for multi-threaded solvers).

### 3.5 Progress Heartbeats & Live Status Display
Workers emit progress updates at least once per minute (default every 15 seconds) into `runs/<run-id>/progress.json`, recording elapsed time, step progress, percentage, ETA, and live resource telemetry. Executing `vscan jobs status <run-id>` reads this heartbeat to display live percentage, step, item count, and ETA.

---

## 4. Scoped Cancellation & Stale PID Protection

Cancellation targets **only** verified owned process groups through a strict two-stage protocol:

```text
[vscan jobs cancel <run-id>]
            │
            ▼
[Verify Process Identity]
  - Is PID alive?
  - Does psutil.Process(pid).create_time() match recorded create_time within 2.0s?
  - Does proc.cmdline() match 'vscan'?
            ├── (Identity Mismatch / Reused PID) ──► [Flag STALE_PID_DETECTED, Mark INTERRUPTED, Do NOT kill]
            │
            └── (Verified Owned Process)
                     │
                     ▼
             [Stage 1: Cooperative Cancellation]
               - Send SIGTERM to process group (os.killpg(pgid, SIGTERM))
               - Touch runs/<run-id>/cancel_requested
               - Await exit during grace interval (default 5.0 seconds)
                     │
                     ├── (Exited cleanly within grace interval) ──► [Mark CANCELLED]
                     │
                     └── (Still alive after grace interval)
                              │
                              ▼
                      [Stage 2: Scoped Termination]
                        - Send SIGKILL to process group (os.killpg(pgid, SIGKILL))
                        - Verify final process termination
                        - Mark CANCELLED
```

---

## 5. Atomic Completion Markers & Checksum Verification

### 5.1 The `.complete` Marker
Outputs are **never** declared complete by loose file presence. A stage run is only reusable if `outputs/.complete` exists and is verified:

```json
{
  "schema_version": "1.0.0",
  "run_id": "run_smoke_20260916_102935_0a9544",
  "stage": "smoke",
  "session_id": "synth_live_demo",
  "completed_at_utc": "2026-09-16T10:29:36Z",
  "cache_key": "7ca5efb733ba02d3ffb7b93af84592e979c851cd21807c712c3b4889a40d8a55",
  "output_checksums": {
    "manifest.json": "8f42ab...",
    "summary.txt": "c3e109..."
  },
  "output_sizes_bytes": {
    "manifest.json": 164,
    "summary.txt": 160
  },
  "marker_sha256": "4b68ef..."
}
```

### 5.2 Atomic Write Guarantee
The marker is written to `.complete.tmp.<pid>` and committed via `os.replace` (atomic file rename) only after all worker computation has completed and file handles are flushed and synced (`os.fsync`).

### 5.3 Corruption Rejection Policy
During reconciliation or cache checking, if any output file:
- Is missing,
- Has an unexpected file size, or
- Fails SHA256 checksum verification against `.complete`:

The output is **immediately rejected**, and the run is transitioned to `FAILED` with `CORRUPT_OUTPUT`. Raw input data is **strictly preserved and never deleted**.

---

## 6. Deterministic Caching & Resumability

### 6.1 Cache Key Generation
The cache key deterministically captures all factors influencing a stage output:
$$\mathrm{CacheKey} = \mathrm{SHA256}\Big(\mathrm{stage} \,\|\, \mathrm{config\_hash} \,\|\, \mathrm{input\_hashes} \,\|\, \mathrm{calibration\_hash} \,\|\, \mathrm{pose\_hash}\Big)$$
Any alteration to input files, camera calibration, or config settings generates a completely distinct key, invalidating downstream cache reuse.

### 6.2 Indivisible Stage Resumption
- Stages are indivisible execution units.
- Resuming an interrupted or failed stage (`vscan jobs resume <run-id>`) launches a new run attempt (`parent_run_id = prior_run_id`, `attempt = prior_attempt + 1`).
- Valid verified outputs from preceding stages with matching cache keys are reused; the interrupted stage is re-run from its clean starting boundary.
