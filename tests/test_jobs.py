"""Comprehensive integration test suite for vscan jobs supervisor, lifecycle, and concurrency.

Tests specified in Prompt 03:
1. Start/status/log a smoke job; verify exit code and completed outputs.
2. Cancel a running job, restart it from validated checkpoints, and compare its result to an uninterrupted execution.
3. Simulate supervisor termination and a stale PID; reconcile safely.
4. Submit two GPU-labelled fixtures: only one may execute at once.
5. Change one input/config hash: downstream output must not be reused.
6. Corrupt a completion marker/output: reject it without deleting raw data.
7. Confirm a deliberately failing command never becomes SUCCEEDED.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
import pytest
import psutil

from vscan.engine.supervisor import Supervisor
from vscan.schemas.jobs import ErrorClassification, JobStatus
from vscan.storage.artifacts import compute_sha256_file, verify_completion_marker
from vscan.storage.database import DatabaseManager


@pytest.fixture
def test_env(tmp_path: Path):
    """Create an isolated test environment with custom DB and runs dir."""
    db_path = tmp_path / "app.sqlite3"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a test config with max 1 GPU job and max 2 CPU jobs
    config_path = tmp_path / "test_config.yaml"
    cfg_data = {
        "profile": "test_profile",
        "concurrency": {"max_gpu_jobs": 1, "max_cpu_jobs": 2, "worker_cpu_threads": 2},
        "process_management": {"cooperative_grace_period_seconds": 2.0, "heartbeat_interval_seconds": 1.0},
        "timeouts": {"default_seconds": 60, "stages": {"smoke": 10, "gpu_smoke": 10, "fail_test": 10}},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f)

    supervisor = Supervisor(
        db_path=db_path,
        runs_dir=runs_dir,
        sessions_dir=sessions_dir,
        config_path=config_path,
    )
    return {
        "tmp_path": tmp_path,
        "db_path": db_path,
        "runs_dir": runs_dir,
        "sessions_dir": sessions_dir,
        "config_path": config_path,
        "supervisor": supervisor,
    }


def test_01_smoke_job_start_status_log_outputs(test_env):
    """1. Start/status/log a smoke job; verify exit code and completed outputs."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_smoke_01"

    # Start smoke job
    submission = supervisor.start_job(
        stage="smoke",
        session_id=session_id,
        config_path=test_env["config_path"],
        wait=True,
    )
    run_id = submission["run_id"]
    assert run_id.startswith("run_smoke_")

    # Status check
    record = supervisor.get_status(run_id)
    assert record is not None
    assert record.status == JobStatus.SUCCEEDED
    assert record.error_classification == ErrorClassification.NONE
    assert record.duration_seconds is not None and record.duration_seconds >= 0.0

    # Log check
    log_content = "".join(supervisor.stream_logs(run_id))
    assert "[SmokeWorker] Successfully completed smoke run" in log_content

    # Output verification
    outputs_dir = test_env["runs_dir"] / run_id / "outputs"
    assert (outputs_dir / "manifest.json").is_file()
    assert (outputs_dir / "summary.txt").is_file()
    assert (outputs_dir / ".complete").is_file()

    # Completion marker verification
    is_valid, msg, marker = verify_completion_marker(outputs_dir, expected_cache_key=record.cache_key)
    assert is_valid, msg
    assert marker is not None
    assert "manifest.json" in marker.output_checksums
    assert "summary.txt" in marker.output_checksums


def test_02_cancel_and_resume_job(test_env):
    """2. Cancel a running job, restart it from validated checkpoints, and compare its result."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_cancel_02"

    # Start a smoke job
    sub = supervisor.start_job(stage="smoke", session_id=session_id, config_path=test_env["config_path"], wait=False)
    run_id = sub["run_id"]

    # Promptly cancel the running job
    time.sleep(0.1)
    success, msg = supervisor.cancel_job(run_id)
    assert success, msg

    rec_after = supervisor.get_status(run_id)
    assert rec_after.status == JobStatus.CANCELLED
    assert rec_after.error_classification == ErrorClassification.CANCELLED

    # Ensure incomplete outputs are NEVER marked complete
    outputs_dir = test_env["runs_dir"] / run_id / "outputs"
    assert not (outputs_dir / ".complete").exists()

    # Resume the job via supervisor.resume_job
    resumed_sub = supervisor.resume_job(run_id, wait=True)
    resumed_rec = supervisor.get_status(resumed_sub["run_id"])
    assert resumed_rec.status == JobStatus.SUCCEEDED
    assert resumed_rec.parent_run_id == run_id
    assert resumed_rec.attempt == 2

    # Compare to uninterrupted smoke execution
    uninterrupted = supervisor.start_job(stage="smoke", session_id=session_id, config_path=test_env["config_path"], force=True, wait=True)
    unint_rec = supervisor.get_status(uninterrupted["run_id"])
    assert unint_rec.status == JobStatus.SUCCEEDED

    # Compare output content
    out1 = (test_env["runs_dir"] / resumed_sub["run_id"] / "outputs" / "manifest.json").read_text()
    out2 = (test_env["runs_dir"] / uninterrupted["run_id"] / "outputs" / "manifest.json").read_text()
    data1 = json.loads(out1)
    data2 = json.loads(out2)
    assert data1["synthetic_frames_count"] == data2["synthetic_frames_count"]


def test_03_stale_pid_and_supervisor_termination_reconcile(test_env):
    """3. Simulate supervisor termination and a stale PID; reconcile safely."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_stale_03"

    sub = supervisor.start_job(stage="smoke", session_id=session_id, config_path=test_env["config_path"], wait=False)
    run_id = sub["run_id"]

    # Simulate supervisor termination while job is recorded as RUNNING with a stale/non-existent PID
    record = supervisor.get_status(run_id)
    record.status = JobStatus.RUNNING
    record.pid = 9999999  # Guaranteed non-existent or unowned PID
    record.process_create_time = 100.0
    supervisor.db.update_job(record)

    # Reconcile safely
    res = supervisor.reconcile()
    assert res["status"] == "RECONCILED"
    assert res["interrupted_recovered_count"] >= 1
    assert run_id in res["interrupted_run_ids"]

    # Status must now be INTERRUPTED, not RUNNING
    reconciled_rec = supervisor.get_status(run_id)
    assert reconciled_rec.status == JobStatus.INTERRUPTED
    assert reconciled_rec.error_classification == ErrorClassification.INTERRUPTED


def test_04_gpu_concurrency_queue_single_slot(test_env):
    """4. Submit two GPU-labelled fixtures: only one may execute at once."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_gpu_04"

    # Start first GPU job
    sub1 = supervisor.start_job(stage="gpu_smoke", session_id=session_id, config_path=test_env["config_path"], wait=False)
    run_id_1 = sub1["run_id"]

    # Immediately submit second GPU job
    sub2 = supervisor.start_job(stage="gpu_smoke", session_id=session_id, config_path=test_env["config_path"], wait=False)
    run_id_2 = sub2["run_id"]

    # Check status: Job 1 should be RUNNING, Job 2 MUST be QUEUED (only 1 GPU job allowed)
    rec1 = supervisor.get_status(run_id_1)
    rec2 = supervisor.get_status(run_id_2)

    assert rec1.status in (JobStatus.RUNNING, JobStatus.SUCCEEDED)
    # If job1 is still running, job2 must be strictly QUEUED
    if rec1.status == JobStatus.RUNNING:
        assert rec2.status == JobStatus.QUEUED

    # Wait for completion of both jobs
    supervisor._wait_for_job(run_id_1)
    supervisor._wait_for_job(run_id_2)

    assert supervisor.get_status(run_id_1).status == JobStatus.SUCCEEDED
    assert supervisor.get_status(run_id_2).status == JobStatus.SUCCEEDED


def test_05_input_config_hash_change_invalidates_cache(test_env):
    """5. Change one input/config hash: downstream output must not be reused."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_hash_05"

    cfg1_path = test_env["tmp_path"] / "cfg1.json"
    with open(cfg1_path, "w", encoding="utf-8") as f:
        json.dump({"param_a": 100, "concurrency": {"max_gpu_jobs": 1, "max_cpu_jobs": 2}}, f)

    sub1 = supervisor.start_job(stage="smoke", session_id=session_id, config_path=cfg1_path, wait=True)
    rec1 = supervisor.get_status(sub1["run_id"])
    assert rec1.status == JobStatus.SUCCEEDED
    cache_key_1 = rec1.cache_key

    # Submitting identical job should reuse cache
    sub1_cached = supervisor.start_job(stage="smoke", session_id=session_id, config_path=cfg1_path, force=False)
    assert sub1_cached.get("cached") is True
    assert sub1_cached["run_id"] == sub1["run_id"]

    # Now alter config parameter
    cfg2_path = test_env["tmp_path"] / "cfg2.json"
    with open(cfg2_path, "w", encoding="utf-8") as f:
        json.dump({"param_a": 999, "concurrency": {"max_gpu_jobs": 1, "max_cpu_jobs": 2}}, f)

    sub2 = supervisor.start_job(stage="smoke", session_id=session_id, config_path=cfg2_path, force=False, wait=True)
    rec2 = supervisor.get_status(sub2["run_id"])
    assert rec2.status == JobStatus.SUCCEEDED
    cache_key_2 = rec2.cache_key

    # Keys MUST be different and sub2 must NOT have reused sub1
    assert cache_key_1 != cache_key_2
    assert sub2["run_id"] != sub1["run_id"]
    assert sub2.get("cached") is False


def test_06_corrupted_marker_or_output_rejected_safely(test_env):
    """6. Corrupt a completion marker/output: reject it without deleting raw data."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_corrupt_06"

    # Place raw mock file in session directory
    session_dir = test_env["sessions_dir"] / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    raw_file = session_dir / "raw_sensor.csv"
    raw_file.write_text("timestamp,acc_x,acc_y,acc_z\n100,0.1,0.2,9.8\n")
    raw_checksum_before = compute_sha256_file(raw_file)

    # Run smoke job
    sub = supervisor.start_job(stage="smoke", session_id=session_id, config_path=test_env["config_path"], wait=True)
    run_id = sub["run_id"]
    rec = supervisor.get_status(run_id)
    assert rec.status == JobStatus.SUCCEEDED

    # Corrupt an output file in the run outputs directory
    out_path = test_env["runs_dir"] / run_id / "outputs" / "summary.txt"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\nMALICIOUS_CORRUPTING_EXTRA_LINE\n")

    # Verify marker rejection
    outputs_dir = test_env["runs_dir"] / run_id / "outputs"
    is_valid, reason, _ = verify_completion_marker(outputs_dir, expected_cache_key=rec.cache_key)
    assert not is_valid
    assert ("Checksum corruption" in reason or "Size mismatch" in reason)

    # Reconcile: must mark the job FAILED due to corruption
    rec_res = supervisor.reconcile()
    assert rec_res["corrupt_detected_count"] >= 1
    assert run_id in rec_res["corrupt_run_ids"]

    corrupted_rec = supervisor.get_status(run_id)
    assert corrupted_rec.status == JobStatus.FAILED
    assert corrupted_rec.error_classification == ErrorClassification.CORRUPT_OUTPUT

    # CRITICAL: Raw session data must remain intact and NOT deleted!
    assert raw_file.is_file()
    assert compute_sha256_file(raw_file) == raw_checksum_before


def test_07_deliberate_failure_never_succeeded(test_env):
    """7. Confirm a deliberately failing command never becomes SUCCEEDED."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_fail_07"

    cfg_path = test_env["tmp_path"] / "fail_cfg.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fail_mode": "crash",
                "concurrency": {"max_gpu_jobs": 1, "max_cpu_jobs": 2},
                "process_management": {"cooperative_grace_period_seconds": 2.0},
            },
            f,
        )

    sub = supervisor.start_job(stage="fail_test", session_id=session_id, config_path=cfg_path, wait=True)
    run_id = sub["run_id"]
    rec = supervisor.get_status(run_id)

    # Must be FAILED, never SUCCEEDED
    assert rec.status == JobStatus.FAILED
    assert rec.status != JobStatus.SUCCEEDED
    assert rec.error_classification == ErrorClassification.SUBPROCESS_CRASH
    assert rec.error_message is not None
    assert "Deliberate worker failure injected" in rec.error_message

    # Completion marker must never exist
    outputs_dir = test_env["runs_dir"] / run_id / "outputs"
    assert not (outputs_dir / ".complete").exists()


def test_08_stage_timeout_enforcement(test_env):
    """8. Enforce timeouts: a hanging job must be terminated and marked FAILED with TIMEOUT."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_timeout_08"

    cfg_path = test_env["tmp_path"] / "timeout_cfg.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fail_mode": "hang",
                "concurrency": {"max_gpu_jobs": 1, "max_cpu_jobs": 2},
                "process_management": {"cooperative_grace_period_seconds": 1.0, "heartbeat_interval_seconds": 0.5},
                "timeouts": {"default_seconds": 60, "stages": {"fail_test": 1.0}},
            },
            f,
        )

    sub = supervisor.start_job(stage="fail_test", session_id=session_id, config_path=cfg_path, wait=False)
    run_id = sub["run_id"]

    # Wait for the watchdog/timeout to trigger (timeout=1.0s)
    time.sleep(2.5)

    # Trigger queue advance or reconciliation
    supervisor.advance_queue()

    rec = supervisor.get_status(run_id)
    assert rec.status == JobStatus.FAILED
    assert rec.error_classification == ErrorClassification.TIMEOUT
    assert not (test_env["runs_dir"] / run_id / "outputs" / ".complete").exists()


def test_09_cpu_concurrency_queue_bounded(test_env):
    """9. Allow max 2 CPU jobs concurrently; 3rd job must remain QUEUED until slot frees."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_cpu_queue_09"

    # Submit 3 smoke jobs concurrently (max_cpu_jobs is 2 in test_config)
    sub1 = supervisor.start_job(stage="smoke", session_id=session_id, config_path=test_env["config_path"], wait=False)
    sub2 = supervisor.start_job(stage="smoke", session_id=session_id, config_path=test_env["config_path"], wait=False)
    sub3 = supervisor.start_job(stage="smoke", session_id=session_id, config_path=test_env["config_path"], wait=False)

    rec1 = supervisor.get_status(sub1["run_id"])
    rec2 = supervisor.get_status(sub2["run_id"])
    rec3 = supervisor.get_status(sub3["run_id"])

    # If first two are active, third must be strictly QUEUED
    running_or_done = sum(1 for r in [rec1, rec2] if r.status in (JobStatus.RUNNING, JobStatus.SUCCEEDED))
    if running_or_done == 2:
        assert rec3.status == JobStatus.QUEUED

    # Wait for all to complete
    supervisor._wait_for_job(sub1["run_id"])
    supervisor._wait_for_job(sub2["run_id"])
    supervisor._wait_for_job(sub3["run_id"])

    assert supervisor.get_status(sub1["run_id"]).status == JobStatus.SUCCEEDED
    assert supervisor.get_status(sub2["run_id"]).status == JobStatus.SUCCEEDED
    assert supervisor.get_status(sub3["run_id"]).status == JobStatus.SUCCEEDED


def test_10_log_following_decoupled_from_process(test_env):
    """10. Detaching / closing log stream must never terminate the worker subprocess."""
    supervisor: Supervisor = test_env["supervisor"]
    session_id = "synth_session_log_detach_10"

    sub = supervisor.start_job(stage="smoke", session_id=session_id, config_path=test_env["config_path"], wait=False)
    run_id = sub["run_id"]

    # Start following logs and immediately close stream (simulating user Ctrl+C or pipe close)
    stream = supervisor.stream_logs(run_id, follow=True)
    try:
        for chunk in stream:
            break
    finally:
        stream.close()

    # The background worker must continue executing independently to success
    supervisor._wait_for_job(run_id)
    rec = supervisor.get_status(run_id)
    assert rec.status == JobStatus.SUCCEEDED
    assert (test_env["runs_dir"] / run_id / "outputs" / ".complete").is_file()
