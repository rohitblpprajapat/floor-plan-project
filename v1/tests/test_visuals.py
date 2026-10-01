import os
import cv2
import numpy as np
import pytest
from src.stella.stella_io import VisualKeyframe, VisualTrajectory
from src.visuals.floorplan_2d import FloorPlan2DVisualizer
from src.visuals.interactive_3d import Interactive3DVisualizer
from src.visuals.video_annotator import VideoAnnotator

@pytest.fixture
def sample_trajectory_and_landmarks():
    # 10 keyframes along a 5-meter path
    keyframes = [
        VisualKeyframe(
            keyframe_id=i,
            timestamp=float(i) * 0.5,
            R_wc=np.eye(3),
            p_wc=np.array([float(i) * 0.5, 0.1 * np.sin(i), 0.05 * np.cos(i)])
        )
        for i in range(10)
    ]
    traj = VisualTrajectory(keyframes)

    # 100 random 3D landmarks around path
    np.random.seed(42)
    landmarks = np.random.uniform(-2.0, 6.0, size=(100, 3))
    landmarks[:, 2] = np.random.uniform(-0.5, 1.5, size=100)

    return traj, landmarks

def test_floorplan_2d_rendering(tmp_path, sample_trajectory_and_landmarks):
    traj, landmarks = sample_trajectory_and_landmarks
    vis = FloorPlan2DVisualizer(dpi=100)

    fp_out = str(tmp_path / "test_floorplan.png")
    vis.render_topdown_floorplan(traj, landmarks, fp_out)
    assert os.path.exists(fp_out)
    assert os.path.getsize(fp_out) > 10000 # Valid non-empty PNG

    ortho_out = str(tmp_path / "test_ortho.png")
    vis.render_multiview_orthographic(traj, landmarks, ortho_out)
    assert os.path.exists(ortho_out)
    assert os.path.getsize(ortho_out) > 10000

    prof_out = str(tmp_path / "test_profiles.png")
    vis.render_motion_profiles(traj, prof_out)
    assert os.path.exists(prof_out)
    assert os.path.getsize(prof_out) > 10000

def test_interactive_3d_html(tmp_path, sample_trajectory_and_landmarks):
    traj, landmarks = sample_trajectory_and_landmarks
    html_out = str(tmp_path / "test_3d.html")

    Interactive3DVisualizer.render_interactive_html(traj, landmarks, html_out)
    assert os.path.exists(html_out)
    assert os.path.getsize(html_out) > 50000 # Standalone HTML with Plotly JS

    with open(html_out, "r", encoding="utf-8") as f:
        content = f.read()
    assert "plotly" in content.lower()
    assert "Camera Trajectory" in content

def test_video_annotator_frame_and_minimap(sample_trajectory_and_landmarks):
    traj, landmarks = sample_trajectory_and_landmarks
    annotator = VideoAnnotator(traj, landmarks)

    # Mini-map rendering
    minimap = annotator.render_minimap(curr_idx=3, map_size=180)
    assert minimap.shape == (180, 180, 3)

    # Blank test frame (equirectangular 800x400)
    dummy_frame = np.zeros((400, 800, 3), dtype=np.uint8)
    ann_frame = annotator.annotate_frame(dummy_frame, keyframe_idx=3)

    assert ann_frame.shape == (400, 800, 3)
    # HUD text and mini-map should make the frame non-zero
    assert np.sum(ann_frame) > 0
