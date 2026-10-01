"""Command line interface for vscan."""

from __future__ import annotations

import json
import os
import sys
import typer

app = typer.Typer(
    help="vscan - Local Insta360 reconstruction and annotation CLI.",
    no_args_is_help=True,
)

from vscan.cli_jobs import jobs_app
app.add_typer(jobs_app, name="jobs")


@app.callback()
def main() -> None:
    """vscan CLI root entrypoint."""
    pass


@app.command(name="doctor")
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Output machine-readable JSON format."),
) -> None:
    """Run environment, package, GPU, and native computation diagnostics."""
    if as_json:
        os.environ["GLOG_minloglevel"] = "2"

    from vscan.doctor import run_full_diagnosis

    diagnosis = run_full_diagnosis()

    if as_json:
        typer.echo(json.dumps(diagnosis, indent=2))
        if diagnosis["status"] != "PASS":
            sys.exit(1)
        return

    # Formatted terminal output
    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)
    typer.secho("            vscan Environment Doctor              ", fg=typer.colors.CYAN, bold=True)
    typer.secho("==================================================", fg=typer.colors.CYAN, bold=True)

    env = diagnosis["environment"]
    typer.echo(f"Python: {env['python_version']} ({env['python_executable']})")
    typer.echo(f"Isolated venv: {env['is_virtual_env']} ({env['sys_prefix']})")

    typer.secho("\n--- Packages ---", bold=True)
    for pkg, pinfo in diagnosis["packages"].items():
        status_color = typer.colors.GREEN if pinfo.get("installed") else typer.colors.RED
        status_str = f"v{pinfo.get('version')}" if pinfo.get("installed") else f"FAILED ({pinfo.get('error')})"
        typer.secho(f"  {pkg:15}: {status_str}", fg=status_color)

    typer.secho("\n--- System Tools & Hardware ---", bold=True)
    tools = diagnosis["system_tools"]
    typer.echo(f"  ffmpeg      : {tools['ffmpeg'].get('version', 'Not found')}")
    typer.echo(f"  ffprobe     : {tools['ffprobe'].get('version', 'Not found')}")
    typer.echo(f"  nvidia-smi  : {tools['nvidia_smi'].get('path', 'Not found')} [{tools['nvidia_smi'].get('gpu_summary')}]")
    typer.echo(f"  /dev/dxg    : {'Present' if tools['dev_dxg_present'] else 'Missing'}")

    typer.secho("\n--- PyCOLMAP Capabilities ---", bold=True)
    caps = diagnosis["pycolmap_capabilities"]
    typer.echo(f"  CUDA Devices: {caps.get('cuda_devices_count')} (has_cuda={caps.get('has_cuda')})")
    typer.echo(f"  Fixed Rig   : {'Supported' if caps.get('fixed_rig_supported') else 'Missing'}")
    typer.echo(f"  Recon Export: {'Supported' if caps.get('reconstruction_export_supported') else 'Missing'}")
    typer.echo(f"  Undistortion: {'Supported' if caps.get('undistortion_supported') else 'Missing'}")
    typer.echo(f"  Dense Stereo: {'Supported' if caps.get('stereo_supported') else 'Missing'}")
    pano_status = "Built-in" if caps.get("panorama_helpers_present") else "Absent (Replaced by Prompt 05 ray renderer)"
    typer.echo(f"  Panorama    : {pano_status}")

    typer.secho("\n--- Live Test Results ---", bold=True)
    num_tests = diagnosis["numerical_tests"]
    gtsam_stat = num_tests.get("gtsam_native_calculation", {}).get("status", "FAIL")
    typer.secho(f"  GTSAM Native Optimizer : {gtsam_stat}", fg=typer.colors.GREEN if gtsam_stat == "PASS" else typer.colors.RED)

    sift = diagnosis["sift_tests"]
    cuda_sift = sift.get("cuda_sift", {})
    cuda_stat = cuda_sift.get("status", "FAIL")
    cuda_dt = cuda_sift.get("elapsed_seconds", 0)
    cuda_kps = cuda_sift.get("keypoints_count", 0)
    typer.secho(
        f"  PyCOLMAP SIFT CUDA     : {cuda_stat} ({cuda_kps} kps in {cuda_dt}s, device={cuda_sift.get('resolved_device')})",
        fg=typer.colors.GREEN if cuda_stat == "PASS" else typer.colors.RED,
    )

    cpu_sift = sift.get("cpu_sift", {})
    cpu_stat = cpu_sift.get("status", "FAIL")
    cpu_dt = cpu_sift.get("elapsed_seconds", 0)
    cpu_kps = cpu_sift.get("keypoints_count", 0)
    typer.secho(
        f"  PyCOLMAP SIFT CPU      : {cpu_stat} ({cpu_kps} kps in {cpu_dt}s, fallback ready)",
        fg=typer.colors.GREEN if cpu_stat == "PASS" else typer.colors.RED,
    )

    rig = diagnosis["rig_test"]
    rig_stat = rig.get("status", "FAIL")
    typer.secho(f"  Rig Transform/Serial.  : {rig_stat}", fg=typer.colors.GREEN if rig_stat == "PASS" else typer.colors.RED)

    video = diagnosis["video_decode_test"]
    video_stat = video.get("status", "FAIL")
    typer.secho(f"  FFmpeg/ffprobe Decode  : {video_stat}", fg=typer.colors.GREEN if video_stat == "PASS" else typer.colors.RED)

    typer.echo("==================================================")
    final_status = diagnosis["status"]
    if final_status == "PASS":
        typer.secho("Overall Diagnosis: PASS (Environment healthy)", fg=typer.colors.GREEN, bold=True)
    else:
        typer.secho("Overall Diagnosis: FAIL (See above for issues)", fg=typer.colors.RED, bold=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
