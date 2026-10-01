import tempfile
from pathlib import Path
import cv2
import numpy as np
import pycolmap

from vscan.frames.projection import get_default_pinhole_intrinsics, build_virtual_camera_poses
from vscan.frames.rig import export_rig_config_json

temp_dir = Path(tempfile.mkdtemp())
print("Testing in:", temp_dir)
images_dir = temp_dir / "images"
images_dir.mkdir()
db_path = temp_dir / "test.db"

# Notice how images are named in Prompt 05 pilot extraction:
# frame_000087_face_front.jpg
# Wait! In build_colmap_rig_json:
# "image_prefix": f"{pose.camera_name}_" -> "face_front_"
# But image name was "frame_000087_face_front.jpg"!
# Does image_prefix look at prefix of filename?
# If image_prefix is "face_front_", then "frame_..." does NOT start with "face_front_"!
# OR is image_prefix matched or stripped?
# Let's inspect how pycolmap matches image_prefix in rig.cc!

# Let's check:
intrinsics = get_default_pinhole_intrinsics(480, 480, 90.0)
poses = build_virtual_camera_poses(intrinsics)
for k, v in poses.items():
    print(f"Pose {k}: cam_name={v.camera_name}, face={v.face_name}")

# Let's test image naming convention:
# Option 1: face_front/frame_000000.jpg (subfolder) with prefix "face_front/"
# Option 2: face_front_000000.jpg with prefix "face_front_"
# Option 3: frame_000000_face_front.jpg with prefix ?
