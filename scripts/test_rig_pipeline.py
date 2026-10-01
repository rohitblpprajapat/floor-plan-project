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

# Create 2 stations x 6 faces dummy images with some corners/features
faces = ["front", "right", "back", "left", "up", "down"]
for st in range(2):
    for f in faces:
        img_name = f"frame_{st:06d}_face_{f}.jpg"
        img = np.zeros((480, 480, 3), dtype=np.uint8)
        # Draw some circles/rectangles for features
        for x in range(50, 450, 60):
            for y in range(50, 450, 60):
                cv2.circle(img, (x + st*5, y), 8, (255, 255, 255), -1)
                cv2.rectangle(img, (x-10, y-10), (x+10, y+10), (128, 128, 128), 2)
        cv2.imwrite(str(images_dir / img_name), img)

intrinsics = get_default_pinhole_intrinsics(480, 480, 90.0)
poses = build_virtual_camera_poses(intrinsics)
rig_json = temp_dir / "rig_config.json"
export_rig_config_json(poses, rig_json)
rigs = pycolmap.read_rig_config(str(rig_json))

# Test Option A: extract features first, then apply_rig_config
reader_opts = pycolmap.ImageReaderOptions()
reader_opts.camera_model = "PINHOLE"
reader_opts.camera_params = f"{intrinsics.fx},{intrinsics.fy},{intrinsics.cx},{intrinsics.cy}"

extract_opts = pycolmap.FeatureExtractionOptions()
extract_opts.use_gpu = False # test on cpu first
extract_opts.num_threads = 2

print("Running extract_features...")
pycolmap.extract_features(
    database_path=str(db_path),
    image_path=str(images_dir),
    camera_mode=pycolmap.CameraMode.SINGLE,
    reader_options=reader_opts,
    extraction_options=extract_opts,
    device=pycolmap.Device.cpu,
)

db = pycolmap.Database.open(str(db_path))
print("Num cameras:", db.num_cameras())
print("Num images:", db.num_images())
print("Num keypoints in img 1:", db.num_keypoints_for_image(1))

print("Applying rig config...")
pycolmap.apply_rig_config(rigs, db)
print("Num rigs:", db.num_rigs())
print("Num frames:", db.num_frames())
db.close()

# Test matching
match_opts = pycolmap.FeatureMatchingOptions()
match_opts.use_gpu = False
match_opts.num_threads = 2
match_opts.skip_image_pairs_in_same_frame = True

pair_opts = pycolmap.SequentialPairingOptions()
pair_opts.overlap = 10
pair_opts.expand_rig_images = True

print("Running match_sequential...")
pycolmap.match_sequential(
    database_path=str(db_path),
    matching_options=match_opts,
    pairing_options=pair_opts,
    device=pycolmap.Device.cpu,
)

db = pycolmap.Database.open(str(db_path))
print("Num verified matches:", db.num_verified_image_pairs())
db.close()
print("Pipeline test successful!")
