"""Durable local supervisor and job lifecycle coordinator.

Guarantees:
- Durable SQLite + filesystem persistence.
- Job submission returns ID and log paths promptly.
- Subprocesses execute in detached process groups (start_new_session=True).
- Max 1 GPU job at a time across project workers; bounded CPU queue.
- Scoped cooperative cancellation and grace intervals.
- Stale PID reuse detection and safe reconciliation.
- Atomic completion verification and output checksum validation.
"""

from __future__ import annotations

import calendar
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import psutil

from vscan.engine.process import cancel_process_group, verify_process_identity
from vscan.engine.telemetry import collect_full_telemetry
from vscan.schemas.jobs import (
    ErrorClassification,
    JobRecordSchema,
    JobStatus,
)
from vscan.storage.artifacts import (
    compute_cache_key,
    compute_sha256_dict,
    compute_sha256_file,
    verify_completion_marker,
)
from vscan.storage.database import DatabaseManager, DEFAULT_DB_PATH
from vscan.workers import get_worker_class


class Supervisor:
    """Coordinates job submission, queue scheduling, cancellation, and lifecycle state."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        runs_dir: Path = Path("runs"),
        sessions_dir: Path = Path("data/sessions"),
        config_path: Optional[Path] = None,
    ) -> None:
        self.default_config_path = Path(config_path) if config_path else Path("configs/laptop.yaml")
        if config_path and Path(config_path).is_file():
            try:
                cfg = self.load_config(Path(config_path))
                storage = cfg.get("storage", {})
                if db_path == DEFAULT_DB_PATH and "app_db_path" in storage:
                    db_path = Path(storage["app_db_path"])
                if runs_dir == Path("runs") and "runs_dir" in storage:
                    runs_dir = Path(storage["runs_dir"])
                if sessions_dir == Path("data/sessions") and "sessions_dir" in storage:
                    sessions_dir = Path(storage["sessions_dir"])
            except Exception:
                pass

        self.db_path = Path(db_path)
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.db = DatabaseManager(self.db_path)

    def load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load YAML or JSON configuration."""
        target = Path(config_path) if config_path else self.default_config_path
        if not target.is_file():
            # Return standard laptop defaults if file not yet written
            return {
                "profile": "laptop",
                "concurrency": {"max_gpu_jobs": 1, "max_cpu_jobs": 2, "worker_cpu_threads": 6},
                "process_management": {"cooperative_grace_period_seconds": 5.0, "heartbeat_interval_seconds": 15.0},
                "timeouts": {"default_seconds": 1800, "stages": {"smoke": 30, "gpu_smoke": 30, "fail_test": 30}},
            }
        if target.suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(target, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except ImportError:
                pass
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)

    def start_job(
        self,
        stage: str,
        session_id: str,
        config_path: Optional[Path] = None,
        profile: str = "laptop",
        force: bool = False,
        wait: bool = False,
        parent_run_id: Optional[str] = None,
        attempt: int = 1,
    ) -> Dict[str, Any]:
        """Submit a job to the supervisor queue promptly.

        Returns:
            Dictionary containing run_id, status, log_path, and cache_key.
        """
        cfg_path = Path(config_path) if config_path else self.default_config_path
        config = self.load_config(cfg_path)
        config_hash = compute_sha256_dict(config)

        # Worker capability inspection
        worker_cls = get_worker_class(stage)
        requires_gpu = worker_cls.requires_gpu

        # Resolve input hashes for session
        input_hashes = self._resolve_input_hashes(stage, session_id)
        cache_key = compute_cache_key(stage, config_hash, input_hashes)

        # Cache check: Reuse existing SUCCEEDED run if valid
        if not force:
            cached_job = self.db.find_cached_successful_job(cache_key, session_id=session_id)
            if cached_job:
                outputs_dir = self.runs_dir / cached_job.run_id / "outputs"
                is_valid, reason, _ = verify_completion_marker(outputs_dir, expected_cache_key=cache_key)
                if is_valid:
                    return {
                        "run_id": cached_job.run_id,
                        "status": JobStatus.SUCCEEDED.value,
                        "log_path": str(self.runs_dir / cached_job.run_id / "job.log"),
                        "outputs_path": str(outputs_dir),
                        "cache_key": cache_key,
                        "cached": True,
                        "message": "Reusing cached verified successful execution",
                    }

        # Generate unique run ID
        now = time.time()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime(now))
        short_id = uuid.uuid4().hex[:6]
        run_id = f"run_{stage}_{timestamp_str}_{short_id}"

        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "job.log"

        # Freeze effective configuration
        effective_cfg_path = run_dir / "effective_config.json"
        with open(effective_cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Build initial job record
        cmd_args = [
            sys.executable,
            "-m",
            "vscan.engine.runner",
            "--run-id",
            run_id,
            "--stage",
            stage,
            "--session",
            session_id,
            "--config",
            str(effective_cfg_path.resolve()),
            "--profile",
            profile,
            "--db-path",
            str(self.db_path.resolve()),
            "--runs-dir",
            str(self.runs_dir.resolve()),
        ]

        record = JobRecordSchema(
            run_id=run_id,
            stage=stage,
            session_id=session_id,
            profile=profile,
            status=JobStatus.QUEUED,
            parent_run_id=parent_run_id,
            attempt=attempt,
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            cmd_args=cmd_args,
            requires_gpu=requires_gpu,
            input_hashes=input_hashes,
            config_hash=config_hash,
            cache_key=cache_key,
        )

        # Insert into DB and write run.json
        self.db.insert_job(record)
        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(record.model_dump(mode="json"), f, indent=2)

        # Advance queue to start execution if slot is available
        self.advance_queue()

        # Reload updated status
        updated_rec = self.db.get_job(run_id) or record

        if wait:
            self._wait_for_job(run_id)
            updated_rec = self.db.get_job(run_id) or updated_rec

        return {
            "run_id": run_id,
            "status": updated_rec.status.value,
            "log_path": str(log_path),
            "outputs_path": str(outputs_dir),
            "cache_key": cache_key,
            "cached": False,
        }

    def _resolve_input_hashes(self, stage: str, session_id: str) -> Dict[str, str]:
        """Resolve hashes for all inputs belonging to session/stage."""
        hashes: Dict[str, str] = {"session_id": session_id}
        # 1. Registered input assets from database
        try:
            db_assets = self.db.list_input_assets(session_id)
            for asset in db_assets:
                hashes[f"asset/{asset.asset_id}"] = asset.file_sha256
        except Exception:
            pass

        # 2. Files in session directory
        session_dir = self.sessions_dir / session_id
        if session_dir.is_dir():
            for root, _, files in os.walk(session_dir):
                for fname in sorted(files):
                    if fname.startswith("."):
                        continue
                    fpath = Path(root) / fname
                    rel = str(fpath.relative_to(session_dir))
                    try:
                        hashes[f"session/{rel}"] = compute_sha256_file(fpath)
                    except Exception:
                        pass

        # 3. For ingest stage, if assets not yet registered, hash raw files from input candidates
        if stage == "ingest" and len(hashes) <= 1:
            date_cand = f"{session_id[:4]}-{session_id[4:6]}-{session_id[6:8]}" if len(session_id) >= 8 and session_id[:8].isdigit() else ""
            cand_dirs = [
                self.sessions_dir.parent / date_cand if date_cand else None,
                self.sessions_dir / session_id / "inputs",
                self.sessions_dir / session_id,
            ]
            for cand in cand_dirs:
                if cand and cand.is_dir():
                    for f in sorted(cand.iterdir()):
                        if f.is_file() and not f.name.startswith("."):
                            try:
                                hashes[f"raw_input/{f.name}"] = compute_sha256_file(f)
                            except Exception:
                                pass
                    if len(hashes) > 1:
                        break

        return hashes

    def advance_queue(self) -> List[str]:
        """Process queued jobs adhering to GPU (max 1), CPU, and stage workload limits."""
        active_jobs = self.db.get_running_or_queued_jobs()
        config = self.load_config()
        concurrency = config.get("concurrency", {})
        stage_limits = config.get("stage_limits", {})
        max_gpu = int(concurrency.get("max_gpu_jobs", 1))
        max_cpu = int(concurrency.get("max_cpu_jobs", 2))

        # Check running jobs: reconcile dead processes or timed out jobs
        running_jobs: List[JobRecordSchema] = []
        queued_jobs: List[JobRecordSchema] = []

        for job in active_jobs:
            if job.status == JobStatus.RUNNING:
                is_alive = False
                if job.pid:
                    verified, _, _ = verify_process_identity(job.pid, job.process_create_time, "vscan")
                    is_alive = verified

                    # Check if stage has exceeded its configured timeout
                    job_cfg = config
                    job_cfg_path = self.runs_dir / job.run_id / "effective_config.json"
                    if job_cfg_path.is_file():
                        try:
                            with open(job_cfg_path, "r", encoding="utf-8") as f:
                                job_cfg = json.load(f)
                        except Exception:
                            pass

                    stage_timeout = float(
                        job_cfg.get("timeouts", {}).get("stages", {}).get(
                            job.stage, job_cfg.get("timeouts", {}).get("default_seconds", 1800)
                        )
                    )
                    is_timed_out = False
                    if job.started_at_utc:
                        try:
                            started_epoch = calendar.timegm(time.strptime(job.started_at_utc, "%Y-%m-%dT%H:%M:%SZ"))
                            if (time.time() - started_epoch) >= stage_timeout:
                                is_timed_out = True
                        except Exception:
                            pass

                    if is_timed_out:
                        if job.pid:
                            cancel_process_group(job.pid, job.pgid, job.process_create_time, grace_interval_seconds=1.0)
                        job.status = JobStatus.FAILED
                        job.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        job.error_classification = ErrorClassification.TIMEOUT
                        job.error_message = f"Supervisor detected job exceeded stage timeout limit of {stage_timeout}s"
                        self.db.update_job(job)
                        self._update_run_json(job)
                    else:
                        running_jobs.append(job)
                else:
                    # Process died without updating status (Interruption / crash)
                    job.status = JobStatus.INTERRUPTED
                    job.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    job.error_classification = ErrorClassification.INTERRUPTED
                    job.error_message = f"Process {job.pid} disappeared unexpectedly"
                    self.db.update_job(job)
                    self._update_run_json(job)
            elif job.status == JobStatus.QUEUED:
                queued_jobs.append(job)

        active_gpu_count = sum(1 for j in running_jobs if j.requires_gpu)
        active_cpu_count = len(running_jobs)

        launched_ids: List[str] = []

        for job in queued_jobs:
            if job.requires_gpu:
                if active_gpu_count >= max_gpu:
                    continue  # GPU slot full
                if active_cpu_count >= max_cpu:
                    continue  # CPU queue full
            else:
                if active_cpu_count >= max_cpu:
                    continue  # CPU queue full

            # Enforce stage-specific concurrency limits if configured
            stage_max = int(stage_limits.get(job.stage, {}).get("max_concurrent", 999))
            active_stage_count = sum(1 for j in running_jobs if j.stage == job.stage)
            if active_stage_count >= stage_max:
                continue

            # Launch job subprocess
            self._spawn_worker(job)
            launched_ids.append(job.run_id)
            running_jobs.append(job)
            active_cpu_count += 1
            if job.requires_gpu:
                active_gpu_count += 1

        return launched_ids

    def _spawn_worker(self, job: JobRecordSchema) -> None:
        """Spawn worker in its own process group with detached session."""
        run_dir = self.runs_dir / job.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(run_dir / "job.log", "a", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        src_dir = str(Path(__file__).resolve().parent.parent.parent)
        repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{repo_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else f"{src_dir}{os.pathsep}{repo_root}"

        # Set CPU thread limits
        config = self.load_config()
        threads = str(config.get("concurrency", {}).get("worker_cpu_threads", 6))
        env["OMP_NUM_THREADS"] = threads
        env["MKL_NUM_THREADS"] = threads
        env["NUMEXPR_NUM_THREADS"] = threads
        env["OPENBLAS_NUM_THREADS"] = threads

        # Spawn detached process group
        # start_new_session=True creates new session and process group (setsid on POSIX)
        # creationflags=subprocess.CREATE_NEW_PROCESS_GROUP on Windows
        kwargs: Dict[str, Any] = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "cwd": repo_root,
            "env": env,
        }
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(job.cmd_args, **kwargs)

        pid = proc.pid
        pgid = os.getpgid(pid) if hasattr(os, "getpgid") else pid
        try:
            ps_proc = psutil.Process(pid)
            create_time = ps_proc.create_time()
        except Exception:
            create_time = time.time()

        job.status = JobStatus.RUNNING
        job.started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        job.pid = pid
        job.pgid = pgid
        job.process_create_time = create_time
        job.hostname = os.uname().nodename if hasattr(os, "uname") else "localhost"

        self.db.update_job(job)
        self._update_run_json(job)

    def cancel_job(self, run_id: str) -> Tuple[bool, str]:
        """Cancel a running or queued job.

        Enforces cooperative cancellation first, followed by scoped termination.
        Detects and guards against stale PID reuse.
        """
        record = self.db.get_job(run_id)
        if record is None:
            return False, f"Job {run_id} not found"

        if record.status == JobStatus.QUEUED:
            record.status = JobStatus.CANCELLED
            record.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            record.error_classification = ErrorClassification.CANCELLED
            record.error_message = "Cancelled while waiting in queue"
            self.db.update_job(record)
            self._update_run_json(record)
            self.advance_queue()
            return True, f"Queued job {run_id} cancelled"

        if record.status != JobStatus.RUNNING:
            return False, f"Job {run_id} is in terminal status {record.status.value}, cannot cancel"

        if not record.pid:
            record.status = JobStatus.CANCELLED
            self.db.update_job(record)
            self._update_run_json(record)
            return True, f"Job {run_id} has no associated PID; marked CANCELLED"

        config = self.load_config()
        grace_seconds = float(config.get("process_management", {}).get("cooperative_grace_period_seconds", 5.0))

        success, msg = cancel_process_group(
            pid=record.pid,
            pgid=record.pgid,
            expected_create_time=record.process_create_time,
            grace_interval_seconds=grace_seconds,
        )

        record.status = JobStatus.CANCELLED
        record.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record.error_classification = ErrorClassification.CANCELLED
        record.error_message = msg
        self.db.update_job(record)
        self._update_run_json(record)

        # Advance queue in case slot opened up
        self.advance_queue()
        return success, msg

    def resume_job(self, run_id: str, wait: bool = False) -> Dict[str, Any]:
        """Resume an interrupted, cancelled, or failed stage run.

        Re-runs the stage while reusing valid verified predecessors.
        Creates a new attempt linked by parent_run_id directly in the new record.
        """
        prior = self.db.get_job(run_id)
        if prior is None:
            raise ValueError(f"Job {run_id} not found to resume")

        if prior.status == JobStatus.SUCCEEDED:
            raise ValueError(f"Job {run_id} already SUCCEEDED; resume not applicable")

        # Check if prior run stored an effective configuration
        prior_run_dir = self.runs_dir / prior.run_id
        prior_cfg = prior_run_dir / "effective_config.json"
        cfg_to_use = prior_cfg if prior_cfg.is_file() else None

        # Submit new run directly initialized with parent_run_id and attempt counter
        submission = self.start_job(
            stage=prior.stage,
            session_id=prior.session_id,
            config_path=cfg_to_use,
            profile=prior.profile,
            force=True,
            wait=wait,
            parent_run_id=prior.run_id,
            attempt=prior.attempt + 1,
        )
        submission["parent_run_id"] = prior.run_id
        submission["attempt"] = prior.attempt + 1
        return submission

    def get_status(self, run_id: str) -> Optional[JobRecordSchema]:
        """Retrieve full status record and verify live process state."""
        record = self.db.get_job(run_id)
        if record is None:
            # Check filesystem fallback
            rf = self.runs_dir / run_id / "run.json"
            if rf.is_file():
                with open(rf, "r", encoding="utf-8") as f:
                    record = JobRecordSchema.model_validate_json(f.read())
        return record

    def get_progress(self, run_id: str) -> Optional[JobProgressSchema]:
        """Read latest progress heartbeat from runs/<run-id>/progress.json."""
        prog_file = self.runs_dir / run_id / "progress.json"
        if prog_file.is_file():
            try:
                with open(prog_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return JobProgressSchema.model_validate(data)
            except Exception:
                pass
        return None

    def list_jobs(
        self,
        session_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[JobRecordSchema]:
        """List jobs matching query."""
        return self.db.list_jobs(session_id=session_id, status=status, limit=limit)

    def reconcile(self) -> Dict[str, Any]:
        """Reconcile active jobs and outputs against real system state.

        Actions:
        1. Detects dead or stale PIDs for jobs marked RUNNING -> marks INTERRUPTED.
        2. Detects timed out jobs -> terminates and marks FAILED with TIMEOUT.
        3. Detects corrupted outputs or missing markers on claimed SUCCEEDED jobs -> marks FAILED.
        4. Advances queued jobs into freed slots.
        """
        reconciled_interrupted: List[str] = []
        reconciled_timed_out: List[str] = []
        reconciled_corrupt: List[str] = []

        all_active = self.db.get_running_or_queued_jobs()
        config = self.load_config()

        for job in all_active:
            if job.status == JobStatus.RUNNING:
                is_alive = False
                if job.pid:
                    verified, reason, _ = verify_process_identity(
                        job.pid, job.process_create_time, "vscan"
                    )
                    is_alive = verified

                if not is_alive:
                    job.status = JobStatus.INTERRUPTED
                    job.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    job.error_classification = ErrorClassification.INTERRUPTED
                    job.error_message = "Supervisor detected worker process termination or stale PID reuse"
                    self.db.update_job(job)
                    self._update_run_json(job)
                    reconciled_interrupted.append(job.run_id)
                else:
                    # Check timeout
                    stage_timeout = float(
                        config.get("timeouts", {}).get("stages", {}).get(
                            job.stage, config.get("timeouts", {}).get("default_seconds", 1800)
                        )
                    )
                    is_timed_out = False
                    if job.started_at_utc:
                        try:
                            started_epoch = calendar.timegm(time.strptime(job.started_at_utc, "%Y-%m-%dT%H:%M:%SZ"))
                            if (time.time() - started_epoch) > stage_timeout + 2.0:
                                is_timed_out = True
                        except Exception:
                            pass

                    if is_timed_out:
                        if job.pid:
                            cancel_process_group(job.pid, job.pgid, job.process_create_time, grace_interval_seconds=1.0)
                        job.status = JobStatus.FAILED
                        job.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        job.error_classification = ErrorClassification.TIMEOUT
                        job.error_message = f"Supervisor detected job exceeded stage timeout limit of {stage_timeout}s"
                        self.db.update_job(job)
                        self._update_run_json(job)
                        reconciled_timed_out.append(job.run_id)

        # Also inspect recently claimed SUCCEEDED jobs for completion marker integrity
        recent_succeeded = self.db.list_jobs(status=JobStatus.SUCCEEDED, limit=20)
        for job in recent_succeeded:
            outputs_dir = self.runs_dir / job.run_id / "outputs"
            is_valid, reason, _ = verify_completion_marker(outputs_dir, expected_cache_key=job.cache_key)
            if not is_valid:
                job.status = JobStatus.FAILED
                job.error_classification = ErrorClassification.CORRUPT_OUTPUT
                job.error_message = f"Completion marker corruption detected: {reason}"
                self.db.update_job(job)
                self._update_run_json(job)
                reconciled_corrupt.append(job.run_id)

        # Advance queue after reconciliation
        newly_launched = self.advance_queue()

        return {
            "status": "RECONCILED",
            "interrupted_recovered_count": len(reconciled_interrupted),
            "interrupted_run_ids": reconciled_interrupted,
            "timed_out_count": len(reconciled_timed_out),
            "timed_out_run_ids": reconciled_timed_out,
            "corrupt_detected_count": len(reconciled_corrupt),
            "corrupt_run_ids": reconciled_corrupt,
            "newly_launched_run_ids": newly_launched,
        }

    def stream_logs(
        self,
        run_id: str,
        tail: Optional[int] = None,
        follow: bool = False,
    ) -> Generator[str, None, None]:
        """Read and optionally stream job logs.

        Closing this generator never kills the worker process!
        """
        log_path = self.runs_dir / run_id / "job.log"
        if not log_path.is_file():
            yield f"[Log] Log file not found at {log_path}\n"
            return

        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            if tail and tail > 0:
                lines = f.readlines()
                for line in lines[-tail:]:
                    yield line
            else:
                for line in f:
                    yield line

            if follow:
                while True:
                    line = f.readline()
                    if line:
                        yield line
                    else:
                        # Check if process is still active
                        status_rec = self.get_status(run_id)
                        if status_rec and status_rec.status not in (JobStatus.RUNNING, JobStatus.QUEUED):
                            # Read any remaining output then exit
                            remaining = f.read()
                            if remaining:
                                yield remaining
                            break
                        time.sleep(0.5)

    def _update_run_json(self, record: JobRecordSchema) -> None:
        run_file = self.runs_dir / record.run_id / "run.json"
        try:
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(record.model_dump(mode="json"), f, indent=2)
        except Exception:
            pass

    def _wait_for_job(self, run_id: str, poll_interval: float = 0.5) -> None:
        while True:
            rec = self.db.get_job(run_id)
            if rec and rec.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED):
                break
            self.advance_queue()
            time.sleep(poll_interval)
