"""Isolated worker subprocess runner and lifecycle executor.

Invoked as:
python -m vscan.engine.runner --run-id <id> --stage <stage> --session <id> --config <path> --profile laptop

Guarantees:
- Runs in its own detached process group (start_new_session=True).
- Traps SIGTERM for cooperative cancellation.
- Periodically reports progress heartbeats (at least once per minute).
- Outputs are marked complete ONLY atomically upon genuine success.
- On interruption or failure, outputs are NEVER marked complete.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
import psutil

# Ensure project src directory is in sys.path
_src_dir = Path(__file__).resolve().parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from vscan.engine.telemetry import collect_full_telemetry
from vscan.schemas.jobs import (
    ErrorClassification,
    JobProgressSchema,
    JobRecordSchema,
    JobStatus,
)
from vscan.storage.database import DatabaseManager
from vscan.workers import get_worker_class


class SubprocessRunner:
    """Manages worker execution inside the isolated subprocess."""

    def __init__(
        self,
        run_id: str,
        stage: str,
        session_id: str,
        config_path: Path,
        profile: str = "laptop",
        db_path: Path = Path("data/app.sqlite3"),
        runs_dir: Path = Path("runs"),
    ) -> None:
        self.run_id = run_id
        self.stage = stage
        self.session_id = session_id
        self.config_path = Path(config_path)
        self.profile = profile
        self.db = DatabaseManager(db_path)
        self.runs_dir = Path(runs_dir)
        self.run_dir = self.runs_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = self.run_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.cancelled = False
        self.timed_out = False
        self.heartbeat_stop = threading.Event()
        self.watchdog_stop = threading.Event()
        self.latest_progress = JobProgressSchema(
            run_id=self.run_id,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress_percent=0.0,
            current_step="Starting runner",
        )

        # Setup signal handlers for cooperative cancellation
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

    def _watchdog_loop(self, timeout_seconds: float) -> None:
        """Watchdog thread that triggers timeout cancellation if exceeded."""
        deadline = time.time() + timeout_seconds
        while not self.watchdog_stop.is_set():
            now = time.time()
            if now >= deadline:
                print(
                    f"[Runner] Stage '{self.stage}' exceeded timeout limit of {timeout_seconds:.1f}s. Halting...",
                    file=sys.stderr,
                )
                self.timed_out = True
                self.cancelled = True
                cancel_marker = self.run_dir / "cancel_requested"
                try:
                    cancel_marker.touch()
                except Exception:
                    pass
                break
            time.sleep(min(0.5, max(0.1, deadline - now)))

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        print(f"[Runner] Received signal {signum} (SIGTERM/SIGINT). Initiating cooperative cancellation...", file=sys.stderr)
        self.cancelled = True
        cancel_marker = self.run_dir / "cancel_requested"
        try:
            cancel_marker.touch()
        except Exception:
            pass

    def _load_effective_config(self) -> Dict[str, Any]:
        cfg_file = self.run_dir / "effective_config.json"
        if cfg_file.is_file():
            with open(cfg_file, "r", encoding="utf-8") as f:
                return json.load(f)
        if self.config_path.is_file():
            if self.config_path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}
                except ImportError:
                    pass
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _heartbeat_loop(self, interval_seconds: float = 15.0) -> None:
        """Background thread updating progress.json at least once per minute."""
        while not self.heartbeat_stop.is_set():
            try:
                telemetry = collect_full_telemetry()
                progress_data = self.latest_progress.model_dump(mode="json")
                progress_data["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                progress_data["telemetry"] = telemetry

                progress_file = self.run_dir / "progress.json"
                tmp_file = self.run_dir / f"progress.json.tmp.{os.getpid()}"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(progress_data, f, indent=2)
                os.replace(tmp_file, progress_file)
            except Exception as exc:
                print(f"[Runner Heartbeat] Warning: {exc}", file=sys.stderr)

            self.heartbeat_stop.wait(interval_seconds)

    def _on_progress_update(self, progress: JobProgressSchema) -> None:
        self.latest_progress = progress
        try:
            progress_data = progress.model_dump(mode="json")
            progress_file = self.run_dir / "progress.json"
            tmp_file = self.run_dir / f"progress.json.tmp.{os.getpid()}"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=2)
            os.replace(tmp_file, progress_file)
        except Exception:
            pass

    def run(self) -> int:
        start_time = time.time()
        pid = os.getpid()
        pgid = os.getpgid(0) if hasattr(os, "getpgid") else pid
        create_time = psutil.Process(pid).create_time()

        # Load existing record or create new
        record = self.db.get_job(self.run_id)
        if record is None:
            record = JobRecordSchema(
                run_id=self.run_id,
                stage=self.stage,
                session_id=self.session_id,
                profile=self.profile,
                status=JobStatus.RUNNING,
                created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            )

        # Transition to RUNNING
        record.status = JobStatus.RUNNING
        record.started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time))
        record.pid = pid
        record.pgid = pgid
        record.process_create_time = create_time
        record.hostname = os.uname().nodename if hasattr(os, "uname") else "localhost"
        self.db.update_job(record)

        # Write run.json
        run_json_path = self.run_dir / "run.json"
        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump(record.model_dump(mode="json"), f, indent=2)

        # Load config
        config = self._load_effective_config()

        # Start heartbeat
        heartbeat_interval = float(config.get("process_management", {}).get("heartbeat_interval_seconds", 15.0))
        hb_thread = threading.Thread(target=self._heartbeat_loop, args=(heartbeat_interval,), daemon=True)
        hb_thread.start()

        # Start stage timeout watchdog
        stage_timeout = float(
            config.get("timeouts", {}).get("stages", {}).get(
                self.stage, config.get("timeouts", {}).get("default_seconds", 1800)
            )
        )
        wd_thread = threading.Thread(target=self._watchdog_loop, args=(stage_timeout,), daemon=True)
        wd_thread.start()

        exit_code = 0
        try:
            worker_cls = get_worker_class(self.stage)
            worker = worker_cls(
                run_id=self.run_id,
                session_id=self.session_id,
                config=config,
                run_dir=self.run_dir,
                progress_callback=self._on_progress_update,
                cancel_checker=lambda: self.cancelled,
            )

            # Check if cancelled before start
            if self.cancelled:
                raise InterruptedError("Cancellation requested before worker execution")

            produced_outputs = worker.execute()

            # Verify and write completion marker atomically
            worker.write_completion_marker(produced_outputs, cache_key=record.cache_key)

            # Record success
            finish_time = time.time()
            record.status = JobStatus.SUCCEEDED
            record.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finish_time))
            record.duration_seconds = finish_time - start_time
            record.outputs = produced_outputs
            record.error_classification = ErrorClassification.NONE
            record.error_message = None
            record.telemetry = collect_full_telemetry()
            exit_code = 0

        except (InterruptedError, KeyboardInterrupt):
            finish_time = time.time()
            if self.timed_out:
                record.status = JobStatus.FAILED
                record.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finish_time))
                record.duration_seconds = finish_time - start_time
                record.error_classification = ErrorClassification.TIMEOUT
                record.error_message = f"Job exceeded stage timeout limit of {stage_timeout}s"
                exit_code = 124
            else:
                record.status = JobStatus.CANCELLED
                record.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finish_time))
                record.duration_seconds = finish_time - start_time
                record.error_classification = ErrorClassification.CANCELLED
                record.error_message = "Execution cancelled cooperatively by user request"
                exit_code = 143

        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            finish_time = time.time()
            if code == 0:
                record.status = JobStatus.SUCCEEDED
                record.error_classification = ErrorClassification.NONE
                exit_code = 0
            else:
                record.status = JobStatus.FAILED
                record.error_classification = ErrorClassification.SUBPROCESS_CRASH
                record.error_message = f"Process exited with non-zero exit code: {code}"
                exit_code = code
            record.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finish_time))
            record.duration_seconds = finish_time - start_time

        except Exception as exc:
            finish_time = time.time()
            record.status = JobStatus.FAILED
            record.finished_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finish_time))
            record.duration_seconds = finish_time - start_time
            if self.timed_out:
                record.error_classification = ErrorClassification.TIMEOUT
                record.error_message = f"Job timed out ({stage_timeout}s): {exc}"
                exit_code = 124
            else:
                record.error_classification = ErrorClassification.SUBPROCESS_CRASH
                record.error_message = f"Unhandled worker exception: {type(exc).__name__}: {exc}"
                exit_code = 1
            print(f"[Runner] Execution failed: {exc}", file=sys.stderr)

        finally:
            self.watchdog_stop.set()
            self.heartbeat_stop.set()
            # Retain fresh parent_run_id and attempt if updated by supervisor
            fresh_rec = self.db.get_job(self.run_id)
            if fresh_rec:
                if fresh_rec.parent_run_id:
                    record.parent_run_id = fresh_rec.parent_run_id
                if fresh_rec.attempt > record.attempt:
                    record.attempt = fresh_rec.attempt
            # Update DB and run.json
            self.db.update_job(record)
            with open(run_json_path, "w", encoding="utf-8") as f:
                json.dump(record.model_dump(mode="json"), f, indent=2)

        return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="vscan isolated subprocess runner")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="laptop")
    parser.add_argument("--db-path", default="data/app.sqlite3")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    runner = SubprocessRunner(
        run_id=args.run_id,
        stage=args.stage,
        session_id=args.session,
        config_path=Path(args.config),
        profile=args.profile,
        db_path=Path(args.db_path),
        runs_dir=Path(args.runs_dir),
    )
    code = runner.run()
    sys.exit(code)


if __name__ == "__main__":
    main()
