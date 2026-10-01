"""Script to execute Stage 07 synchronization and clock calibration for real session 20260910_172939."""

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vscan.sync.config import SyncConfig
from vscan.sync.pipeline import SyncPipeline


def main() -> None:
    session_id = "20260910_172939"
    now_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_id = f"run_sync_{now_str}_s7e912"

    runs_dir = Path("runs") / run_id
    outputs_dir = runs_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    sessions_dir = Path("data/sessions") / session_id
    sessions_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Stage 07 Synchronization for Session {session_id} ===")
    print(f"Run ID: {run_id}")
    print(f"Outputs: {outputs_dir}")

    cfg = SyncConfig(
        session_id=session_id,
        run_id=run_id,
        output_dir=outputs_dir,
        is_synthetic=False,
    )

    pipeline = SyncPipeline(cfg)
    result = pipeline.run(progress_callback=lambda pct, msg: print(f"[{pct:.1f}%] {msg}"))

    # Copy output artifacts to data/sessions/20260910_172939/
    for fname in [
        "sync_manifest.json",
        "timing_diagnostics.json",
        "clock_calibration.json",
        "diagnostics_report.json",
        "insv_summary.json",
        "insv_exposures.json",
        "insv_imu.csv",
        "insv_timing_diagnostics.json",
    ]:
        src_f = outputs_dir / fname
        if src_f.is_file():
            shutil.copy2(src_f, sessions_dir / fname)
            print(f"Copied {fname} to {sessions_dir / fname}")

    manifest = result["manifest"]
    print("\n=== Stage 07 Execution Summary ===")
    print(f"Status           : {manifest['status']}")
    print(f"Real Fusion Ready: {manifest['real_fusion_ready']}")
    print(f"Scale Status     : {manifest['scale_status']}")
    print(f"Prompt 08 Ready  : {manifest['prompt_08_readiness']}")
    print("\nAccepted Parameters:")
    for k, v in manifest["accepted_parameters"].items():
        print(f"  - {k}: {v}")
    print("\nUnresolved Parameters:")
    for k, v in manifest["unresolved_parameters"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
