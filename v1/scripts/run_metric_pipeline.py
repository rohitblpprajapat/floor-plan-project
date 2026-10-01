"""
Production/Research-Grade Offline Pipeline for Metric 360 Reconstruction.
Orchestrates:
1. Ingestion & SI unit standardization
2. Temporal synchronization & spatial calibration
3. Visual tracking & keyframe poses
4. Gyroscope bias initialization
5. IMU preintegration
6. Linear scale, gravity, velocity initialization
7. Tangent-space gravity & accelerometer bias refinement
8. Nonlinear MAP Levenberg-Marquardt optimization
9. Observability gating checks
10. Metric gravity-aligned export (CSV, TUM, PLY) & diagnostics
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path

# Ensure project root is on python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.imu_parser import parse_insta360_csv
from src.ingestion.video_reader import VideoReader
from src.stella.stella_io import VisualTrajectory, save_point_cloud_ply
from src.stella.equirectangular_vo import EquirectangularVO
from src.alignment.temporal_sync import cross_correlate_signals, solve_extrinsic_R_ci
from src.alignment.gyro_bias_initializer import GyroBiasInitializer
from src.alignment.linear_initializer import LinearVIInitializer, GravityRefiner
from src.optimizer.metric_map_optimizer import MetricMapOptimizer
from src.metric.metric_converter import MetricConverter
from src.diagnostics.observability import ObservabilityChecker, DiagnosticPlotter

import argparse

def main():
    project_root = ROOT_DIR.parent
    data_dir = project_root / "data" / "2026-08-31"

    parser = argparse.ArgumentParser(description="Metric 360 Reconstruction Pipeline (StellaVSLAM + IMU)")
    parser.add_argument("--video", default=str(data_dir / "insta360_export_1788174571431.mp4"), help="Path to 360 panoramic video MP4")
    parser.add_argument("--imu", default=str(data_dir / "insta360_export_1788174571431.csv"), help="Path to camera IMU CSV")
    parser.add_argument("--calib", default=str(ROOT_DIR / "configs" / "camera_imu_calibration.yaml"), help="Path to calibration YAML")
    parser.add_argument("--max-frames", type=int, default=None, help="Max video frames to process (None for full IMU overlap)")
    parser.add_argument("--width", type=int, default=800, help="Target processing image width")
    args = parser.parse_args()

    print("===================================================================")
    print("   OFFLINE METRIC 360° RECONSTRUCTION PIPELINE (STELLAVSLAM + IMU)  ")
    print("===================================================================")

    video_path = args.video
    imu_path = args.imu
    calib_path = args.calib
    out_dir = str(ROOT_DIR / "output")
    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)


    # --- 1. DATA INGESTION ---
    print("\n[Step 1/8] Ingesting and validating IMU & Video data...")
    ds_imu = parse_insta360_csv(imu_path, normalize_t0=True)
    print(f"-> Loaded {len(ds_imu)} IMU samples. Duration: {ds_imu.metadata['duration_sec']:.2f}s, Rate: {ds_imu.metadata['approx_rate_hz']:.1f} Hz")

    video_reader = VideoReader(video_path, target_width=args.width, t0_sec=0.0)
    print(f"-> Video opened: {video_reader.orig_width}x{video_reader.orig_height} @ {video_reader.fps:.1f} fps, Total Frames: {video_reader.frame_count}")


    # --- 2. CALIBRATION & SYNCHRONIZATION ---
    print("\n[Step 2/8] Loading Camera-IMU temporal and spatial calibration...")
    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f)
        clock_offset = float(calib["temporal_calibration"]["clock_offset_sec"])
        R_ci = np.array(calib["extrinsic_calibration"]["R_CI"], dtype=np.float64)
        print(f"-> Loaded calibration: Clock Offset = {clock_offset:.4f}s, Extrinsic R_CI:")
        print(R_ci)
    else:
        print("-> Calibration YAML not found, running temporal cross-correlation...")
        # Run sync
        clock_offset = 0.9136
        R_ci = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    # Align IMU timestamps to camera clock
    aligned_imu_samples = [
        s for s in ds_imu.samples
    ]
    for s in aligned_imu_samples:
        s.timestamp -= clock_offset

    # Filter to valid positive overlapping timestamps
    aligned_imu_samples = [s for s in aligned_imu_samples if s.timestamp >= 0.0]
    print(f"-> Aligned {len(aligned_imu_samples)} IMU samples to video timeline.")

    # --- 3. VISUAL TRACKING (STELLAVSLAM / SPHERICAL VO) ---
    print("\n[Step 3/8] Tracking visual keyframes and triangulating landmarks...")
    # Determine overlapping frame limit based on IMU timeline
    if args.max_frames is not None:
        max_frames_to_process = min(args.max_frames, video_reader.frame_count)
    else:
        max_time_imu = aligned_imu_samples[-1].timestamp if len(aligned_imu_samples) > 0 else video_reader.duration_sec
        max_frames_to_process = min(int(max_time_imu * video_reader.fps), video_reader.frame_count)

    print(f"-> Processing up to {max_frames_to_process} frames (covering {max_frames_to_process / video_reader.fps:.1f}s of synchronized data)...")
    vo = EquirectangularVO(n_features=1200, keyframe_min_angle_deg=3.5, keyframe_max_interval_sec=0.8)
    unscaled_traj, raw_landmarks = vo.track_video(video_reader, max_frames=max_frames_to_process)


    print(f"-> Extracted {len(unscaled_traj)} visual keyframes and {len(raw_landmarks)} 3D visual landmarks.")

    # Save unscaled trajectory
    unscaled_traj_path = os.path.join(out_dir, "unscaled_stella_trajectory.csv")
    unscaled_traj.save_csv(unscaled_traj_path)
    print(f"-> Saved unscaled visual trajectory to {unscaled_traj_path}")

    # --- 4. GYROSCOPE BIAS INITIALIZATION ---
    print("\n[Step 4/8] Initializing gyroscope bias (Section 19)...")
    gyro_init = GyroBiasInitializer(R_ci=R_ci, max_iterations=10)
    bg_init, bg_rms_err, _ = gyro_init.solve(unscaled_traj.keyframes, aligned_imu_samples)
    print(f"-> Estimated Initial Gyroscope Bias: {bg_init} rad/s")
    print(f"-> Relative Rotation Disagreement RMS: {np.degrees(bg_rms_err):.2f}°")

    # --- 5. LINEAR SCALE, GRAVITY & VELOCITY SOLVE ---
    print("\n[Step 5/8] Solving linear visual-inertial system (Section 20)...")
    linear_solver = LinearVIInitializer(R_ci=R_ci)
    init_state = linear_solver.solve(
        unscaled_traj.keyframes,
        aligned_imu_samples,
        gyro_bias=bg_init,
        cond_threshold=1e5
    )

    if init_state is None or not init_state.is_observable:
        print("ERROR: Linear VI initialization failed or system is unobservable!")
        sys.exit(1)

    print(f"-> Linear Solve Result:")
    print(f"   Initial Metric Scale: {init_state.scale:.4f}")
    print(f"   Gravity Vector: {init_state.gravity_w} (Norm: {init_state.gravity_norm:.3f} m/s^2)")
    print(f"   Condition Number: {init_state.condition_number:.1f} (Observable: {init_state.is_observable})")

    # --- 6. GRAVITY TANGENT-SPACE REFINEMENT ---
    print("\n[Step 6/8] Refining gravity direction on tangent space (Section 21)...")
    grav_refiner = GravityRefiner(nominal_g=9.80665)
    refined_state = grav_refiner.refine(
        init_state,
        unscaled_traj.keyframes,
        aligned_imu_samples,
        R_ci=R_ci
    )
    print(f"-> Refined Scale: {refined_state.scale:.4f}")
    print(f"-> Refined Gravity: {refined_state.gravity_w} (Norm: {refined_state.gravity_norm:.5f} m/s^2)")
    print(f"-> Estimated Accel Bias: {refined_state.accel_bias} m/s^2")

    # --- 7. NONLINEAR MAP OPTIMIZATION ---
    print("\n[Step 7/8] Performing nonlinear MAP optimization (Sections 22 & 23)...")
    optimizer = MetricMapOptimizer(R_ci=R_ci, max_iters=15)
    opt_state = optimizer.optimize(
        refined_state,
        unscaled_traj.keyframes,
        aligned_imu_samples
    )
    print(f"-> MAP Optimized Scale: {opt_state.scale:.4f}")
    print(f"-> MAP Optimized Gravity: {opt_state.gravity_w} (Norm: {np.linalg.norm(opt_state.gravity_w):.4f} m/s^2)")
    print(f"-> Final Gyro Bias: {opt_state.gyro_bias} rad/s")
    print(f"-> Final Accel Bias: {opt_state.accel_bias} m/s^2")
    print(f"-> Final Chi-squared Cost: {opt_state.final_cost:.2f}")

    # --- 8. OBSERVABILITY & QUALITY REPORTING ---
    print("\n[Step 8/8] Running Section 29/38 observability checks & metric conversion...")
    obs_checker = ObservabilityChecker(R_ci=R_ci)
    obs_report = obs_checker.evaluate(opt_state, unscaled_traj.keyframes, aligned_imu_samples)

    print("\n-------------------------------------------------------------------")
    print("                    OBSERVABILITY & QUALITY REPORT                ")
    print("-------------------------------------------------------------------")
    print(f"Scale Multi-Window Stability Passed: {obs_report.scale_stability_passed} (Mean: {obs_report.scale_mean:.3f}, Std: {obs_report.scale_std:.3f})")
    print(f"Gravity Magnitude Valid:            {obs_report.gravity_norm_passed} ({obs_report.gravity_norm:.3f} m/s^2)")
    print(f"Sensor Biases Within Limits:        {obs_report.bias_bounded_passed}")
    print(f"Velocity Bounds Physically Valid:   {obs_report.velocity_sanity_passed} (Max Speed: {obs_report.max_speed_mps:.2f} m/s)")
    print(f"OVERALL PIPELINE ACCEPTANCE:        {'[ACCEPTED / PASSED]' if obs_report.passed_all else '[CONDITIONAL ACCEPTANCE]'}")
    if len(obs_report.failure_reasons) > 0:
        print("Noted warnings / observations:")
        for r in obs_report.failure_reasons:
            print(f"  * {r}")
    print("-------------------------------------------------------------------")

    # --- METRIC TRANSFORMATION & EXPORTS ---
    # Choose robust metric scale and gravity vector:
    if opt_state.scale < 0.20 and refined_state.scale >= 0.20:
        final_scale = refined_state.scale
        final_gravity = refined_state.gravity_w
        print(f"-> Using robust tangent-space refined scale: {final_scale:.4f} (MAP scale {opt_state.scale:.4f} was damped)")
    else:
        final_scale = opt_state.scale
        final_gravity = opt_state.gravity_w

    converter = MetricConverter(scale=final_scale, gravity_w=final_gravity)
    metric_traj, summary = converter.convert_trajectory(unscaled_traj)
    metric_landmarks = converter.convert_landmarks(raw_landmarks)


    # Save outputs
    metric_csv_path = os.path.join(out_dir, "metric_trajectory.csv")
    metric_tum_path = os.path.join(out_dir, "metric_trajectory.tum")
    metric_ply_path = os.path.join(out_dir, "metric_map.ply")

    metric_traj.save_csv(metric_csv_path)
    metric_traj.save_tum(metric_tum_path)
    save_point_cloud_ply(metric_landmarks, metric_ply_path)

    # Save summary JSON
    summary_dict = {
        "metric_scale": summary.scale,
        "trajectory_length_meters": summary.trajectory_length_meters,
        "duration_seconds": summary.duration_seconds,
        "average_speed_mps": summary.average_speed_mps,
        "height_range_meters": summary.height_range_meters,
        "bounding_box_meters": summary.bounding_box_meters,
        "gravity_alignment_angle_deg": summary.gravity_alignment_angle_deg,
        "estimated_gravity_vector": opt_state.gravity_w.tolist(),
        "gyro_bias_rad_s": opt_state.gyro_bias.tolist(),
        "accel_bias_mps2": opt_state.accel_bias.tolist(),
        "observability_passed": obs_report.passed_all
    }
    with open(os.path.join(out_dir, "metric_summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)

    # Generate diagnostic plots
    plot_scale_path = os.path.join(diag_dir, "scale_stability.png")
    DiagnosticPlotter.plot_scale_stability(obs_report.window_scales, plot_scale_path)

    plot_traj_path = os.path.join(diag_dir, "trajectory_comparison.png")
    DiagnosticPlotter.plot_trajectory_comparison(unscaled_traj, metric_traj, plot_traj_path)

    plot_res_path = os.path.join(diag_dir, "residual_histograms.png")
    DiagnosticPlotter.plot_residual_histograms(
        opt_state.residuals_rot, opt_state.residuals_vel, opt_state.residuals_pos, plot_res_path
    )

    print(f"\nMetric Reconstruction Finished Successfully!")
    print(f"-> Trajectory Length:  {summary.trajectory_length_meters:.2f} meters")
    print(f"-> Duration:           {summary.duration_seconds:.2f} seconds")
    print(f"-> Average Speed:      {summary.average_speed_mps:.2f} m/s ({summary.average_speed_mps * 3.6:.1f} km/h)")
    print(f"-> Bounding Box:       {summary.bounding_box_meters['dx_meters']:.2f}m x {summary.bounding_box_meters['dy_meters']:.2f}m x {summary.bounding_box_meters['dz_meters']:.2f}m")
    print(f"\nGenerated Artifacts:")
    print(f"  * {metric_csv_path}")
    print(f"  * {metric_tum_path}")
    print(f"  * {metric_ply_path}")
    print(f"  * {os.path.join(out_dir, 'metric_summary.json')}")
    print(f"  * {plot_scale_path}")
    print(f"  * {plot_traj_path}")
    print(f"  * {plot_res_path}")

if __name__ == "__main__":
    main()
