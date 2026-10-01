"""
Master Visual Results Generator.
Orchestrates generation of:
1. Architectural 2D floor plans with wall density maps
2. 3-view orthographic technical projections
3. Motion, elevation, and velocity profile charts
4. Standalone interactive 3D HTML web viewer
5. Annotated 360 keyframe snapshots with re-projected 3D landmarks
6. Annotated replay survey video with HUD and PiP 2D mini-map
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.stella.stella_io import load_trajectory_csv, load_point_cloud_ply
from src.ingestion.video_reader import VideoReader
from src.visuals.floorplan_2d import FloorPlan2DVisualizer
from src.visuals.interactive_3d import Interactive3DVisualizer
from src.visuals.video_annotator import VideoAnnotator

def main():
    print("===================================================================")
    print("      METRIC 360° VISUAL RESULTS & ANNOTATION GENERATOR            ")
    print("===================================================================")

    project_root = ROOT_DIR.parent
    data_dir = project_root / "data" / "2026-08-31"

    traj_path = str(ROOT_DIR / "output" / "metric_trajectory.csv")
    ply_path = str(ROOT_DIR / "output" / "metric_map.ply")
    loc_path = str(data_dir / "20260831_163824_location.json")
    video_path = str(data_dir / "insta360_export_1788174571431.mp4")
    out_dir = str(ROOT_DIR / "output" / "visuals")
    os.makedirs(out_dir, exist_ok=True)


    # 1. Load Metric Trajectory & Landmarks
    print("\n[Step 1/5] Loading reconstructed metric trajectory and point cloud...")
    if not os.path.exists(traj_path) or not os.path.exists(ply_path):
        print("ERROR: Trajectory or point cloud file missing in output/! Run scripts/run_metric_pipeline.py first.")
        sys.exit(1)

    traj = load_trajectory_csv(traj_path)
    landmarks = load_point_cloud_ply(ply_path)
    print(f"-> Loaded {len(traj)} keyframes and {len(landmarks)} metric 3D landmarks.")

    # Load location metadata if present
    start_meta = None
    if os.path.exists(loc_path):
        with open(loc_path, "r") as f:
            start_meta = json.load(f)
        print(f"-> Loaded survey metadata: {start_meta.get('building', {}).get('name')}, {start_meta.get('floor', {}).get('label')}")

    # 2. Render 2D Floor Plan & Multi-View Orthographic Views
    print("\n[Step 2/5] Rendering architectural 2D floor plans & orthographic drawings...")
    vis2d = FloorPlan2DVisualizer(dpi=200)

    fp_path = os.path.join(out_dir, "floorplan_topdown.png")
    vis2d.render_topdown_floorplan(traj, landmarks, fp_path, start_metadata=start_meta)
    print(f"-> Saved 2D Top-Down Floor Plan to: {fp_path}")

    multi_path = os.path.join(out_dir, "multiview_orthographic.png")
    vis2d.render_multiview_orthographic(traj, landmarks, multi_path)
    print(f"-> Saved Multi-View Orthographic to: {multi_path}")

    prof_path = os.path.join(out_dir, "motion_profiles.png")
    vis2d.render_motion_profiles(traj, prof_path)
    print(f"-> Saved Motion & Telemetry Profiles to: {prof_path}")

    # 3. Render Interactive 3D HTML Web Model
    print("\n[Step 3/5] Generating standalone interactive 3D HTML web viewer...")
    html_3d_path = os.path.join(out_dir, "interactive_3d_reconstruction.html")
    Interactive3DVisualizer.render_interactive_html(
        traj, landmarks, html_3d_path,
        title="Tower Alpha (North Block) - Floor 1 Survey"
    )
    print(f"-> Saved Interactive 3D Model to: {html_3d_path}")

    # 4. Generate Annotated Keyframe Snapshots
    print("\n[Step 4/5] Creating annotated keyframe snapshots with re-projected 3D landmarks...")
    annotator = VideoAnnotator(traj, landmarks)
    snap_dir = os.path.join(out_dir, "annotated_keyframes")
    os.makedirs(snap_dir, exist_ok=True)

    reader = VideoReader(video_path, target_width=960)
    # Extract snapshots at key survey moments (e.g. 0s, 10s, 20s, 30s, 40s)
    sample_timestamps = [0.0, 10.0, 20.0, 30.0, 40.0]
    kf_ts = np.array([kf.timestamp for kf in traj.keyframes])

    for target_t in sample_timestamps:
        kf_idx = int(np.argmin(np.abs(kf_ts - target_t)))
        # Read matching frame
        frame_t_target = traj.keyframes[kf_idx].timestamp
        frame_match = None
        for idx, t, f in reader.frames(max_frames=int(frame_t_target * reader.fps) + 5):
            if abs(t - frame_t_target) < 0.05:
                frame_match = f
                break
        if frame_match is not None:
            ann_frame = annotator.annotate_frame(frame_match, kf_idx)
            snap_path = os.path.join(snap_dir, f"keyframe_{kf_idx:03d}_t{frame_t_target:.1f}s.png")
            cv2.imwrite(snap_path, ann_frame)
            print(f"-> Saved annotated snapshot: {snap_path}")

    # 5. Generate Annotated Replay Video
    print("\n[Step 5/5] Generating annotated survey video with telemetry HUD & PiP mini-map...")
    vid_out_path = os.path.join(out_dir, "annotated_survey_preview.mp4")
    # Generate 15-second / 375 frames preview clip for rapid sharing & visualization
    max_video_frames = min(375, reader.frame_count)
    reader_preview = VideoReader(video_path, target_width=800)
    annotator.generate_annotated_video(reader_preview, vid_out_path, max_frames=max_video_frames)
    print(f"-> Saved annotated preview video to: {vid_out_path}")

    print("\n===================================================================")
    print("         ALL 2D & 3D VISUAL RESULTS GENERATED SUCCESSFULLY!        ")
    print("===================================================================")
    print(f"1. Top-Down Floor Plan:        {fp_path}")
    print(f"2. Multi-View Orthographic:    {multi_path}")
    print(f"3. Motion Profiles:            {prof_path}")
    print(f"4. Interactive 3D HTML Viewer: {html_3d_path}")
    print(f"5. Keyframe Snapshots:         {snap_dir}/")
    print(f"6. Annotated Video with HUD:   {vid_out_path}")

if __name__ == "__main__":
    main()
