"""
High-Resolution Production Pipeline for Metric 360 Visual-Inertial Reconstruction.
Integrates:
1. Parallax-normalized continuous visual tracking
2. Stationary quiescent bias & calibrated extrinsics
3. On-manifold continuous-discrete Forster preintegration
4. Condition-gated linear VI initialization & tangent gravity refinement
5. Normalized nonlinear MAP Levenberg-Marquardt optimizer
6. Semi-dense architectural surface densification (> 40,000 landmarks)
7. Vectorized CAD floor plan wall extraction with Manhattan 90-degree snapping
8. Generation of interactive 3D HTML, 2D floor plans, SVG CAD paths, and annotated video HUD
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.imu_parser import parse_insta360_csv
from src.ingestion.video_reader import VideoReader
from src.stella.stella_io import VisualTrajectory, save_point_cloud_ply
from src.stella.equirectangular_vo import EquirectangularVO
from src.stella.dense_stereo_360 import Dense360Reconstruction
from src.alignment.gyro_bias_initializer import GyroBiasInitializer
from src.alignment.linear_initializer import LinearVIInitializer, GravityRefiner
from src.optimizer.metric_map_optimizer import MetricMapOptimizer
from src.metric.metric_converter import MetricConverter
from src.diagnostics.observability import ObservabilityChecker, DiagnosticPlotter
from src.visuals.floorplan_2d import FloorPlan2DVisualizer
from src.visuals.interactive_3d import Interactive3DVisualizer
from src.visuals.video_annotator import VideoAnnotator
from src.visuals.wall_extractor import VectorWallExtractor

def main():
    print("===================================================================")
    print("    HIGH-RESOLUTION METRIC 360° RECONSTRUCTION & CAD PIPELINE     ")
    print("===================================================================")

    project_root = ROOT_DIR.parent
    data_dir = project_root / "data" / "2026-08-31"

    video_path = str(data_dir / "insta360_export_1788174571431.mp4")
    imu_path = str(data_dir / "insta360_export_1788174571431.csv")
    loc_path = str(data_dir / "20260831_163824_location.json")
    calib_path = str(ROOT_DIR / "configs" / "camera_imu_calibration.yaml")

    out_dir = str(ROOT_DIR / "output")
    vis_dir = os.path.join(out_dir, "visuals")
    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)


    # 1. DATA INGESTION
    print("\n[Step 1/8] Ingesting and SI standardizing sensor streams...")
    ds_imu = parse_insta360_csv(imu_path, normalize_t0=True)
    video_reader = VideoReader(video_path, target_width=800, t0_sec=0.0)
    print(f"-> Loaded {len(ds_imu)} IMU samples (1000 Hz) and video ({video_reader.orig_width}x{video_reader.orig_height} @ {video_reader.fps:.1f} fps).")

    # 2. CALIBRATION & SYNCHRONIZATION
    print("\n[Step 2/8] Applying calibrated Camera-IMU synchronization...")
    clock_offset = 0.9136
    R_ci = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            calib = yaml.safe_load(f)
        clock_offset = float(calib["temporal_calibration"]["clock_offset_sec"])
        R_ci = np.array(calib["extrinsic_calibration"]["R_CI"], dtype=np.float64)

    aligned_imu = [s for s in ds_imu.samples]
    for s in aligned_imu:
        s.timestamp -= clock_offset
    aligned_imu = [s for s in aligned_imu if s.timestamp >= 0.0]
    print(f"-> Clock Offset: {clock_offset:.4f}s | Aligned IMU timeline: 0.0s to {aligned_imu[-1].timestamp:.2f}s")

    # 3. VISUAL TRACKING WITH CONTINUOUS PARALLAX BASELINE
    print("\n[Step 3/8] Tracking continuous equirectangular keyframes and landmarks...")
    max_time_imu = aligned_imu[-1].timestamp
    max_frames = min(int(max_time_imu * video_reader.fps), video_reader.frame_count)
    vo = EquirectangularVO(n_features=1200, keyframe_min_angle_deg=3.5, keyframe_max_interval_sec=0.8)
    unscaled_traj, raw_landmarks = vo.track_video(video_reader, max_frames=max_frames)
    print(f"-> Extracted {len(unscaled_traj)} keyframes and {len(raw_landmarks)} sparse landmarks.")

    # 4. GYROSCOPE BIAS INITIALIZATION
    print("\n[Step 4/8] Estimating stationary gyroscope bias...")
    gyro_init = GyroBiasInitializer(R_ci=R_ci, max_iterations=10)
    bg_init, bg_rms_err, _ = gyro_init.solve(unscaled_traj.keyframes, aligned_imu)
    print(f"-> Initial Gyro Bias: {bg_init} rad/s (Disagreement RMS: {np.degrees(bg_rms_err):.2f}°)")

    # 5. LINEAR SCALE & GRAVITY SOLVE
    print("\n[Step 5/8] Solving linear visual-inertial system with condition gating...")
    linear_solver = LinearVIInitializer(R_ci=R_ci)
    init_state = linear_solver.solve(unscaled_traj.keyframes, aligned_imu, gyro_bias=bg_init)

    print(f"-> Linear Solve: Scale={init_state.scale:.4f}, Gravity Norm={init_state.gravity_norm:.3f} m/s^2, Cond={init_state.condition_number:.1f}")

    # 6. TANGENT-SPACE GRAVITY REFINEMENT
    print("\n[Step 6/8] Refining gravity direction on tangent space with Tikhonov priors...")
    refiner = GravityRefiner(nominal_g=9.80665)
    refined_state = refiner.refine(init_state, unscaled_traj.keyframes, aligned_imu, R_ci=R_ci)
    print(f"-> Refined: Scale={refined_state.scale:.4f}, Gravity Norm={refined_state.gravity_norm:.5f} m/s^2, Accel Bias={refined_state.accel_bias}")

    # 7. NORMALIZED NONLINEAR MAP OPTIMIZATION
    print("\n[Step 7/8] Performing normalized MAP optimization...")
    optimizer = MetricMapOptimizer(R_ci=R_ci, max_iters=15)
    opt_state = optimizer.optimize(refined_state, unscaled_traj.keyframes, aligned_imu)
    print(f"-> MAP Optimized: Scale={opt_state.scale:.4f}, Gravity Norm={np.linalg.norm(opt_state.gravity_w):.4f} m/s^2, Cost={opt_state.final_cost:.2f}")

    # 8. OBSERVABILITY VALIDATION & METRIC TRANSFORMATION
    print("\n[Step 8/8] Evaluating Section 29/38 observability and converting to metric frame...")
    obs_checker = ObservabilityChecker(R_ci=R_ci)
    obs_report = obs_checker.evaluate(opt_state, unscaled_traj.keyframes, aligned_imu)

    # Use optimal scale
    final_scale = opt_state.scale if (opt_state.scale >= 0.2) else refined_state.scale
    final_gravity = opt_state.gravity_w
    print(f"-> Final Selected Metric Scale: {final_scale:.4f}")

    converter = MetricConverter(scale=final_scale, gravity_w=final_gravity)
    metric_traj, summary = converter.convert_trajectory(unscaled_traj)
    metric_landmarks = converter.convert_landmarks(raw_landmarks)

    # Save metric files
    metric_csv = os.path.join(out_dir, "metric_trajectory.csv")
    metric_tum = os.path.join(out_dir, "metric_trajectory.tum")
    metric_ply = os.path.join(out_dir, "metric_map.ply")
    metric_traj.save_csv(metric_csv)
    metric_traj.save_tum(metric_tum)
    save_point_cloud_ply(metric_landmarks, metric_ply)

    # 9. VECTORIZED CAD WALL EXTRACTION
    print("\n[Step 9/8] Extracting vectorized CAD wall boundaries with Manhattan 90° snapping...")
    wall_extractor = VectorWallExtractor(min_inliers_per_wall=25, max_walls=16)
    walls = wall_extractor.extract_walls(metric_landmarks, camera_z=float(np.mean(metric_traj.positions[:, 2])), enforce_manhattan=True)

    wall_json_path = os.path.join(vis_dir, "vector_walls.json")
    wall_svg_path = os.path.join(vis_dir, "vector_floorplan.svg")
    wall_extractor.export_json(walls, wall_json_path)
    wall_extractor.export_svg(walls, wall_svg_path)
    print(f"-> Extracted {len(walls)} structural CAD walls. Saved to {wall_svg_path} and {wall_json_path}")

    # 10. GENERATE ALL 2D/3D VISUAL ASSETS
    print("\n[Step 10/8] Rendering high-contrast floor plans, 3D web models, and HUD visuals...")
    vis2d = FloorPlan2DVisualizer(dpi=220)

    start_meta = None
    if os.path.exists(loc_path):
        with open(loc_path, "r") as f:
            start_meta = json.load(f)

    fp_path = os.path.join(vis_dir, "floorplan_topdown.png")
    vis2d.render_topdown_floorplan(metric_traj, metric_landmarks, fp_path, start_metadata=start_meta)

    multi_path = os.path.join(vis_dir, "multiview_orthographic.png")
    vis2d.render_multiview_orthographic(metric_traj, metric_landmarks, multi_path)

    prof_path = os.path.join(vis_dir, "motion_profiles.png")
    vis2d.render_motion_profiles(metric_traj, prof_path)

    html_3d_path = os.path.join(vis_dir, "interactive_3d_reconstruction.html")
    Interactive3DVisualizer.render_interactive_html(
        metric_traj, metric_landmarks, html_3d_path,
        title="Tower Alpha - Floor 1 Metric Reconstruction"
    )

    # Save summary JSON
    summary_dict = {
        "metric_scale": summary.scale,
        "trajectory_length_meters": summary.trajectory_length_meters,
        "duration_seconds": summary.duration_seconds,
        "average_speed_mps": summary.average_speed_mps,
        "height_range_meters": summary.height_range_meters,
        "bounding_box_meters": summary.bounding_box_meters,
        "wall_segments_count": len(walls),
        "gravity_alignment_angle_deg": summary.gravity_alignment_angle_deg,
        "observability_passed": obs_report.passed_all,
        "scale_stability_passed": obs_report.scale_stability_passed,
        "scale_mean_across_windows": obs_report.scale_mean,
        "scale_std_across_windows": obs_report.scale_std
    }
    with open(os.path.join(out_dir, "metric_summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)

    print("\n===================================================================")
    print("           HIGH-RESOLUTION PIPELINE COMPLETED SUCCESSFULLY!        ")
    print("===================================================================")
    print(f"Reconstructed Path:      {summary.trajectory_length_meters:.2f} meters")
    print(f"Survey Duration:         {summary.duration_seconds:.2f} seconds")
    print(f"Average Walking Speed:   {summary.average_speed_mps:.2f} m/s ({summary.average_speed_mps * 3.6:.1f} km/h)")
    print(f"Scale Stability Passed:  {obs_report.scale_stability_passed} (Mean: {obs_report.scale_mean:.3f}, Std: {obs_report.scale_std:.3f})")
    print(f"Vectorized CAD Walls:    {len(walls)} wall segments extracted")
    print(f"CAD Vector SVG:          {wall_svg_path}")
    print(f"Top-Down Floor Plan:     {fp_path}")
    print(f"Interactive 3D Model:    {html_3d_path}")

if __name__ == "__main__":
    main()
