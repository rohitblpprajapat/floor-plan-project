"""
Script to evaluate temporal synchronization and Camera-IMU extrinsic rotation
between panoramic video and Insta360 IMU export.
Generates diagnostic plots in output/diagnostics/
"""

import os
import sys
from pathlib import Path

# Ensure project root is on python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.ingestion.imu_parser import parse_insta360_csv

from src.ingestion.video_reader import VideoReader, SphericalVisualGyro
from src.alignment.temporal_sync import cross_correlate_signals, solve_extrinsic_R_ci
from src.inertial.imu_types import so3_log

def main():
    project_root = ROOT_DIR.parent
    data_dir = project_root / "data" / "2026-08-31"

    diag_dir = ROOT_DIR / "output" / "diagnostics"
    os.makedirs(diag_dir, exist_ok=True)
    video_path = str(data_dir / "insta360_export_1788174571431.mp4")
    imu_path = str(data_dir / "insta360_export_1788174571431.csv")



    print("=== STEP 1: Load IMU Data ===")
    ds = parse_insta360_csv(imu_path, normalize_t0=True)
    t_imu = ds.timestamps
    gyro_imu = ds.gyros
    gyro_imu_norm = np.linalg.norm(gyro_imu, axis=1)
    print(f"Loaded {len(ds)} IMU samples across {ds.metadata['duration_sec']:.2f}s")

    print("\n=== STEP 2: Extract Visual Angular Rates from 360 Video ===")
    # Target width 800 for fast robust feature extraction
    v_reader = VideoReader(video_path, target_width=800, t0_sec=0.0)
    visual_gyro = SphericalVisualGyro(n_features=1200)

    # We will sample frames (up to 750 frames = 30 seconds)
    max_frames = min(750, v_reader.frame_count)
    t_vis = []
    omega_vis = []
    omega_vis_norm = []

    prev_frame = None
    prev_t = 0.0

    print(f"Processing first {max_frames} video frames...")
    for idx, t, frame in tqdm(v_reader.frames(start_frame=0, max_frames=max_frames), total=max_frames):
        if prev_frame is not None:
            R_rel, inliers = visual_gyro.estimate_relative_rotation(prev_frame, frame)
            dt = t - prev_t
            if R_rel is not None and inliers >= 15 and dt > 1e-4:
                rot_vec = so3_log(R_rel)
                w_vec = rot_vec / dt
                w_mag = np.linalg.norm(w_vec)
                # Filter extreme outliers
                if w_mag < 8.0:
                    t_vis.append(0.5 * (prev_t + t))
                    omega_vis.append(w_vec)
                    omega_vis_norm.append(w_mag)
        prev_frame = frame
        prev_t = t

    t_vis = np.array(t_vis)
    omega_vis = np.array(omega_vis)
    omega_vis_norm = np.array(omega_vis_norm)
    print(f"Successfully tracked {len(t_vis)} visual rotation samples.")

    print("\n=== STEP 3: Cross-Correlate Visual & IMU Angular Velocity Magnitudes ===")
    best_lag, peak_corr, lags, corr_vals = cross_correlate_signals(
        t_vis, omega_vis_norm,
        t_imu, gyro_imu_norm,
        dt_resample=0.04, # 25 Hz
        max_lag_sec=15.0
    )
    print(f"Optimal Clock Offset Delta_t: {best_lag:.4f} s (Peak Correlation: {peak_corr:.4f})")

    # Shift IMU timestamps by best_lag to align with video
    t_imu_aligned = t_imu - best_lag

    # Interpolate IMU gyro at visual timestamps
    from scipy.interpolate import interp1d
    valid_overlap = (t_vis >= t_imu_aligned[0]) & (t_vis <= t_imu_aligned[-1])
    t_eval = t_vis[valid_overlap]
    omega_vis_eval = omega_vis[valid_overlap]
    omega_vis_norm_eval = omega_vis_norm[valid_overlap]

    f_gx = interp1d(t_imu_aligned, gyro_imu[:, 0])
    f_gy = interp1d(t_imu_aligned, gyro_imu[:, 1])
    f_gz = interp1d(t_imu_aligned, gyro_imu[:, 2])

    omega_imu_interp = np.column_stack([
        f_gx(t_eval),
        f_gy(t_eval),
        f_gz(t_eval)
    ])

    print("\n=== STEP 4: Solve Camera-IMU Extrinsic Rotation R_CI ===")
    R_ci, rms_err = solve_extrinsic_R_ci(omega_vis_eval, omega_imu_interp, find_nearest_discrete=True)
    print("Optimal R_CI Matrix:")
    print(np.round(R_ci, 4))
    print(f"RMS Angular Velocity Residual: {rms_err:.4f} rad/s")

    # Align IMU gyro into Camera frame
    omega_imu_in_cam = (R_ci @ omega_imu_interp.T).T

    print("\n=== STEP 5: Generate Diagnostic Plots ===")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Cross-correlation curve
    axes[0].plot(lags, corr_vals, 'b-', label='Normalized Cross-Correlation')
    axes[0].axvline(best_lag, color='r', linestyle='--', label=f'Peak Offset: {best_lag:.3f} s')
    axes[0].set_title('Camera-IMU Temporal Cross-Correlation')
    axes[0].set_xlabel('Lag (seconds)')
    axes[0].set_ylabel('Correlation')
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: Magnitude Comparison
    axes[1].plot(t_eval, omega_vis_norm_eval, 'r.-', label='Visual Angular Speed ||omega_vis||')
    axes[1].plot(t_eval, np.linalg.norm(omega_imu_in_cam, axis=1), 'b-', label='Aligned IMU Speed ||omega_imu||')
    axes[1].set_title('Angular Velocity Magnitude Comparison (Aligned)')
    axes[1].set_xlabel('Video Time (seconds)')
    axes[1].set_ylabel('Angular Speed (rad/s)')
    axes[1].grid(True)
    axes[1].legend()

    # Plot 3: 3-Axis Comparison
    axes[2].plot(t_eval, omega_vis_eval[:, 0], 'r--', alpha=0.7, label='Vis X')
    axes[2].plot(t_eval, omega_imu_in_cam[:, 0], 'r-', label='IMU X (in Cam)')
    axes[2].plot(t_eval, omega_vis_eval[:, 1], 'g--', alpha=0.7, label='Vis Y')
    axes[2].plot(t_eval, omega_imu_in_cam[:, 1], 'g-', label='IMU Y (in Cam)')
    axes[2].plot(t_eval, omega_vis_eval[:, 2], 'b--', alpha=0.7, label='Vis Z')
    axes[2].plot(t_eval, omega_imu_in_cam[:, 2], 'b-', label='IMU Z (in Cam)')
    axes[2].set_title('3-Axis Angular Velocity Comparison (in Camera Frame)')
    axes[2].set_xlabel('Video Time (seconds)')
    axes[2].set_ylabel('rad/s')
    axes[2].grid(True)
    axes[2].legend(ncol=3)

    plt.tight_layout()
    plot_path = str(diag_dir / "temporal_sync.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved diagnostic plot to {plot_path}")

    # Save calibration yaml
    import yaml
    calib_data = {
        "temporal_calibration": {
            "clock_offset_sec": float(best_lag),
            "peak_correlation": float(peak_corr)
        },
        "extrinsic_calibration": {
            "R_CI": R_ci.tolist(),
            "rms_residual_rad_s": float(rms_err)
        }
    }
    calib_file = ROOT_DIR / "configs" / "camera_imu_calibration.yaml"
    with open(calib_file, "w") as f:
        yaml.dump(calib_data, f, default_flow_style=False)
    print(f"Saved calibration to {calib_file}")


if __name__ == "__main__":
    main()
