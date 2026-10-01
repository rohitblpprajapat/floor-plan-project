import tempfile
from pathlib import Path
import cv2
import numpy as np
import pycolmap

from vscan.frames.projection import get_default_pinhole_intrinsics, build_virtual_camera_poses
from vscan.frames.rig import export_rig_config_json
from vscan.frames.synthetic import SyntheticRoom

temp_dir = Path(tempfile.mkdtemp())
print("Testing synthetic rig reconstruction in:", temp_dir)

# Render synthetic room for 3 stations with small translation along Z
room = SyntheticRoom()
w, h = 320, 320
intrinsics = get_default_pinhole_intrinsics(w, h, 90.0)
poses = build_virtual_camera_poses(intrinsics)

rig_json = temp_dir / "rig_config.json"
export_rig_config_json(poses, rig_json)
rigs = pycolmap.read_rig_config(str(rig_json))

images_dir = temp_dir / "images"
images_dir.mkdir()

# Station centers along Z: -0.5, 0.0, 0.5
stations = [-0.5, 0.0, 0.5]
for idx, z in enumerate(stations):
    cam_center = np.array([0.0, 0.0, z], dtype=np.float64)
    # Render all 6 faces
    for cam_name, pose in poses.items():
        R = np.array(pose.R_pano_cam, dtype=np.float64).reshape((3, 3))
        rgb, _ = room.render_analytic_pinhole(cam_center, R, intrinsics)
        # Name matching the rig prefix: face_{face}_frame_{idx:06d}.jpg
        # e.g. face_front_frame_000000.jpg
        img_name = f"{cam_name}_frame_{idx:06d}.jpg"
        cv2.imwrite(str(images_dir / img_name), rgb)

db_path = temp_dir / "synthetic.db"
reader_opts = pycolmap.ImageReaderOptions()
reader_opts.camera_model = "PINHOLE"
reader_opts.camera_params = f"{intrinsics.fx},{intrinsics.fy},{intrinsics.cx},{intrinsics.cy}"

extract_opts = pycolmap.FeatureExtractionOptions()
extract_opts.use_gpu = False
extract_opts.num_threads = 4

print("Extracting features...")
pycolmap.extract_features(
    database_path=str(db_path),
    image_path=str(images_dir),
    camera_mode=pycolmap.CameraMode.PER_FOLDER, # or SINGLE
    reader_options=reader_opts,
    extraction_options=extract_opts,
    device=pycolmap.Device.cpu,
)

db = pycolmap.Database.open(str(db_path))
print(f"DB: {db.num_cameras()} cameras, {db.num_images()} images")

print("Applying rig config...")
pycolmap.apply_rig_config(rigs, db)
print(f"DB with rig: {db.num_rigs()} rigs, {db.num_frames()} frames")
db.close()

# Match features
match_opts = pycolmap.FeatureMatchingOptions()
match_opts.use_gpu = False
match_opts.num_threads = 4
match_opts.skip_image_pairs_in_same_frame = True
match_opts.rig_verification = True

pair_opts = pycolmap.SequentialPairingOptions()
pair_opts.overlap = 5
pair_opts.expand_rig_images = True

print("Matching features...")
pycolmap.match_sequential(
    database_path=str(db_path),
    matching_options=match_opts,
    pairing_options=pair_opts,
    device=pycolmap.Device.cpu,
)

db = pycolmap.Database.open(str(db_path))
print(f"Verified image pairs: {db.num_verified_image_pairs()}")
db.close()

# Run incremental mapping
pipe_opts = pycolmap.IncrementalPipelineOptions()
pipe_opts.num_threads = 4
pipe_opts.constant_cameras = True
pipe_opts.constant_rigs = True
pipe_opts.ba_refine_focal_length = False
pipe_opts.ba_refine_principal_point = False
pipe_opts.ba_refine_extra_params = False
pipe_opts.ba_refine_sensor_from_rig = False

output_dir = temp_dir / "sparse"
output_dir.mkdir()

print("Running incremental mapping...")
recons = pycolmap.incremental_mapping(
    database_path=str(db_path),
    image_path=str(images_dir),
    output_path=str(output_dir),
    options=pipe_opts,
)

print("Reconstructions found:", len(recons))
for idx, r in recons.items():
    print(f"  Model {idx}: {r.num_reg_frames()} reg frames, {r.num_reg_images()} reg images, {r.num_points3D()} 3D points")
    print(f"  Mean reproj error: {r.compute_mean_reprojection_error():.3f}px, mean track len: {r.compute_mean_track_length():.2f}")
