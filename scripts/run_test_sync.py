"""Standalone runner for tests/test_sync.py to verify all Prompt 07 test mandates."""

import sys
import time
from pathlib import Path

# Add src and root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

import tests.test_sync as ts


def run_tests() -> None:
    print("==================================================")
    print("    Running Stage 07 Synchronization Test Suite   ")
    print("==================================================")

    tmp_path = root_dir / "runs" / "test_sync_scratch"
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = ts.create_sync_test_env(tmp_path)

    tests_list = [
        ("Test 1: Synthetic Multi-Axis Nonperiodic Sync Recovery (<=2ms)", lambda: ts.test_01_synthetic_multi_axis_nonperiodic_sync_recovery(tmp_path)),
        ("Test 2: Timelapse Fixture Compression Recovery & Unavailable", ts.test_02_timelapse_fixture_compression_recovery_and_unavailable),
        ("Test 3: Unit Discrimination & Physical Checks (g, ms, SO(3))", ts.test_03_unit_discrimination_and_physical_checks),
        ("Test 4: Periodic Motion Ambiguity Rejection (No false pass)", ts.test_04_periodic_motion_ambiguity_rejection),
        ("Test 5: Deliberately Wrong Mappings Worsen Residuals", ts.test_05_deliberately_wrong_mappings_worsen_residuals),
        ("Test 6: 16-Sample Boundary Timing Pattern Preservation", ts.test_06_16_sample_boundary_timing_preservation),
        ("Test 7: Real Pilot Session Audit & Unresolved Verdict", lambda: ts.test_07_real_pilot_session_audit_and_unresolved_verdict(env)),
        ("Test 8: Supervisor Lifecycle (Start, Status, Cancel, Resume)", lambda: ts.test_08_sync_job_lifecycle_cancel_and_resume(env)),
        ("Test 9: Cache Invalidation & Downstream Dependency", lambda: ts.test_09_cache_invalidation_and_downstream_dependency(env)),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests_list:
        t0 = time.time()
        print(f"Running {name}...", end=" ", flush=True)
        try:
            test_fn()
            dt = time.time() - t0
            print(f"PASSED ({dt:.3f}s)")
            passed += 1
        except Exception as exc:
            dt = time.time() - t0
            print(f"FAILED ({dt:.3f}s)")
            print(f"  Error: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("==================================================")
    print(f"Summary: {passed} PASSED, {failed} FAILED (Total: {len(tests_list)})")
    print("==================================================")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
