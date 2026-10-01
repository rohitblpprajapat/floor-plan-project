"""CLI subcommands for vscan jobs lifecycle and supervisor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
import typer

from vscan.engine.supervisor import Supervisor
from vscan.schemas.jobs import JobStatus


jobs_app = typer.Typer(
    help="Manage, schedule, and supervise vscan background execution jobs.",
    no_args_is_help=True,
)


@jobs_app.command(name="start")
def jobs_start(
    stage: str = typer.Argument(..., help="Pipeline stage to execute (e.g. smoke, gpu_smoke, fail_test, unpack)"),
    session: str = typer.Option(..., "--session", "-s", help="Target session identifier"),
    config: Path = typer.Option(Path("configs/laptop.yaml"), "--config", "-c", help="Path to profile or stage YAML/JSON config"),
    profile: str = typer.Option("laptop", "--profile", "-p", help="Execution hardware profile"),
    force: bool = typer.Option(False, "--force", help="Force execution even if cached verified output exists"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Block until job completes (default: prompt async return)"),
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Submit a job to the supervisor queue and return ID and log paths promptly."""
    supervisor = Supervisor(config_path=config)
    try:
        res = supervisor.start_job(
            stage=stage,
            session_id=session,
            config_path=config,
            profile=profile,
            force=force,
            wait=wait,
        )
    except Exception as exc:
        if as_json:
            typer.echo(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2))
        else:
            typer.secho(f"Error submitting job: {exc}", fg=typer.colors.RED, bold=True)
        sys.exit(1)

    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return

    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.secho("              Job Submission Prompt               ", fg=typer.colors.CYAN, bold=True)
    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Run ID       : {res['run_id']}")
    typer.echo(f"  Stage        : {stage}")
    typer.echo(f"  Session      : {session}")
    status_color = typer.colors.GREEN if res["status"] in ("RUNNING", "SUCCEEDED") else typer.colors.YELLOW
    typer.secho(f"  Status       : {res['status']}", fg=status_color, bold=True)
    typer.echo(f"  Log Path     : {res['log_path']}")
    typer.echo(f"  Outputs Path : {res['outputs_path']}")
    typer.echo(f"  Cache Key    : {res['cache_key']}")
    if res.get("cached"):
        typer.secho("  (Reused verified cached execution result)", fg=typer.colors.GREEN)
    typer.echo("==================================================")


@jobs_app.command(name="status")
def jobs_status(
    run_id: str = typer.Argument(..., help="Job run ID to query"),
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Query live execution status and telemetry for a job."""
    supervisor = Supervisor()
    rec = supervisor.get_status(run_id)
    if rec is None:
        if as_json:
            typer.echo(json.dumps({"status": "NOT_FOUND", "run_id": run_id}, indent=2))
        else:
            typer.secho(f"Job run '{run_id}' not found.", fg=typer.colors.RED)
        sys.exit(1)

    prog = supervisor.get_progress(run_id)
    data = rec.model_dump(mode="json")
    data["progress"] = prog.model_dump(mode="json") if prog else None
    if as_json:
        typer.echo(json.dumps(data, indent=2))
        return

    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.secho(f"            Job Status: {rec.run_id}            ", fg=typer.colors.CYAN, bold=True)
    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Stage        : {rec.stage}")
    typer.echo(f"  Session      : {rec.session_id}")
    typer.echo(f"  Profile      : {rec.profile}")
    status_color = {
        JobStatus.SUCCEEDED: typer.colors.GREEN,
        JobStatus.RUNNING: typer.colors.BLUE,
        JobStatus.QUEUED: typer.colors.YELLOW,
        JobStatus.FAILED: typer.colors.RED,
        JobStatus.CANCELLED: typer.colors.MAGENTA,
        JobStatus.INTERRUPTED: typer.colors.RED,
    }.get(rec.status, typer.colors.WHITE)
    typer.secho(f"  Status       : {rec.status.value}", fg=status_color, bold=True)
    typer.echo(f"  Attempt      : {rec.attempt} (Parent: {rec.parent_run_id or 'None'})")
    typer.echo(f"  PID / PGID   : {rec.pid or '-'} / {rec.pgid or '-'}")
    typer.echo(f"  Created At   : {rec.created_at_utc}")
    typer.echo(f"  Started At   : {rec.started_at_utc or '-'}")
    typer.echo(f"  Finished At  : {rec.finished_at_utc or '-'}")
    if rec.duration_seconds is not None:
        typer.echo(f"  Duration     : {rec.duration_seconds:.2f}s")
    if prog:
        typer.echo(f"  Progress     : {prog.progress_percent:.1f}% [{prog.current_step}]")
        if prog.items_total > 0:
            typer.echo(f"  Workload     : {prog.items_completed} / {prog.items_total} items")
        if prog.eta_seconds is not None:
            typer.echo(f"  ETA          : {prog.eta_seconds:.1f}s")
    if rec.error_classification.value != "NONE":
        typer.secho(f"  Error Class  : {rec.error_classification.value}", fg=typer.colors.RED)
        typer.echo(f"  Error Msg    : {rec.error_message or '-'}")
    if rec.outputs:
        typer.echo("  Outputs      :")
        for name, rel in rec.outputs.items():
            typer.echo(f"    - {name}: {rel}")
    typer.echo("==================================================")


@jobs_app.command(name="logs")
def jobs_logs(
    run_id: str = typer.Argument(..., help="Job run ID to view logs for"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output in real-time (Ctrl+C safely exits reader without killing worker)"),
    tail: Optional[int] = typer.Option(None, "--tail", "-n", help="Number of trailing lines to show"),
) -> None:
    """View or stream stdout/stderr execution logs for a job."""
    supervisor = Supervisor()
    try:
        for chunk in supervisor.stream_logs(run_id, tail=tail, follow=follow):
            sys.stdout.write(chunk)
            sys.stdout.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        # Closing log tail must never kill the worker subprocess
        pass


@jobs_app.command(name="cancel")
def jobs_cancel(
    run_id: str = typer.Argument(..., help="Job run ID to cancel"),
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Cooperatively cancel an owned job process group with scoped termination."""
    supervisor = Supervisor()
    success, msg = supervisor.cancel_job(run_id)
    if as_json:
        typer.echo(json.dumps({"run_id": run_id, "success": success, "message": msg}, indent=2))
        if not success:
            sys.exit(1)
        return

    if success:
        typer.secho(f"[Success] {msg}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"[Failed] {msg}", fg=typer.colors.RED)
        sys.exit(1)


@jobs_app.command(name="resume")
def jobs_resume(
    run_id: str = typer.Argument(..., help="Interrupted, failed, or cancelled job run ID to resume"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Block until resumed job completes"),
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Resume an interrupted stage run while reusing validated predecessors."""
    supervisor = Supervisor()
    try:
        res = supervisor.resume_job(run_id, wait=wait)
    except Exception as exc:
        if as_json:
            typer.echo(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2))
        else:
            typer.secho(f"Error resuming job: {exc}", fg=typer.colors.RED)
        sys.exit(1)

    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return

    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.secho(f"             Job Resumed Successfully             ", fg=typer.colors.CYAN, bold=True)
    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Prior Run ID : {res.get('parent_run_id')}")
    typer.echo(f"  New Run ID   : {res['run_id']}")
    typer.echo(f"  Attempt      : {res.get('attempt')}")
    typer.echo(f"  Status       : {res['status']}")
    typer.echo(f"  Log Path     : {res['log_path']}")
    typer.echo("==================================================")


@jobs_app.command(name="list")
def jobs_list(
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Filter by session ID"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status (QUEUED, RUNNING, SUCCEEDED, etc.)"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum jobs to display"),
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """List recent jobs with stage, session, status, and duration."""
    supervisor = Supervisor()
    st_filter = JobStatus(status.upper()) if status else None
    jobs = supervisor.list_jobs(session_id=session, status=st_filter, limit=limit)

    if as_json:
        typer.echo(json.dumps([j.model_dump(mode="json") for j in jobs], indent=2))
        return

    typer.secho(f"{'RUN ID':<35} {'STAGE':<12} {'SESSION':<16} {'STATUS':<12} {'DURATION':<10} {'STARTED':<20}", bold=True)
    typer.echo("-" * 110)
    for j in jobs:
        dur = f"{j.duration_seconds:.1f}s" if j.duration_seconds is not None else "-"
        started = j.started_at_utc or "-"
        typer.echo(f"{j.run_id:<35} {j.stage:<12} {j.session_id:<16} {j.status.value:<12} {dur:<10} {started:<20}")


@jobs_app.command(name="reconcile")
def jobs_reconcile(
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Reconcile supervisor state: detect stale PIDs, recover interrupted jobs, verify outputs."""
    supervisor = Supervisor()
    res = supervisor.reconcile()

    if as_json:
        typer.echo(json.dumps(res, indent=2))
        return

    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.secho("          Supervisor State Reconciliation         ", fg=typer.colors.CYAN, bold=True)
    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  Interrupted Jobs Recovered : {res['interrupted_recovered_count']}")
    if res["interrupted_run_ids"]:
        for rid in res["interrupted_run_ids"]:
            typer.echo(f"    - {rid}")
    typer.echo(f"  Corrupt Outputs Detected   : {res['corrupt_detected_count']}")
    if res["corrupt_run_ids"]:
        for rid in res["corrupt_run_ids"]:
            typer.echo(f"    - {rid}")
    typer.echo(f"  Newly Launched Run IDs     : {len(res['newly_launched_run_ids'])}")
    typer.echo("==================================================")
