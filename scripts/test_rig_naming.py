import tempfile
from pathlib import Path
import pycolmap

from vscan.frames.projection import get_default_pinhole_intrinsics, build_virtual_camera_poses
from vscan.frames.rig import export_rig_config_json

intrinsics = get_default_pinhole_intrinsics(480, 480, 90.0)
poses = build_virtual_camera_poses(intrinsics)

temp_dir = Path(tempfile.mkdtemp())
rig_json = temp_dir / "rig_config.json"
export_rig_config_json(poses, rig_json)
rigs = pycolmap.read_rig_config(str(rig_json))

print("Rig config cameras:")
for c in rigs[0].cameras:
    print(f"  prefix: {c.image_prefix}, ref: {c.ref_sensor}")

# Test 1: image names starting with image_prefix: e.g. face_front_000000.jpg
db_path1 = temp_dir / "test1.db"
db1 = pycolmap.Database.open(str(db_path1))
# 6 cameras (one per rig camera) or 1 camera?
cam_id1 = db1.write_camera(pycolmap.Camera(model="PINHOLE", width=480, height=480, params=[240, 240, 240, 240]))

# What if all images use cam_id1, but start with the prefix?
for st in range(2):
    for face in ["front", "right", "back", "left", "up", "down"]:
        img_name = f"face_{face}_frame_{st:06d}.jpg"
        db1.write_image(pycolmap.Image(name=img_name, camera_id=cam_id1))

print("\nTrying apply_rig_config with face_{face}_frame_... image names (1 camera):")
try:
    pycolmap.apply_rig_config(rigs, db1)
    print("SUCCESS! num_rigs:", db1.num_rigs(), "num_frames:", db1.num_frames())
except Exception as e:
    print("Failed:", e)
db1.close()

# Test 2: What if each face has its own camera_id in DB?
db_path2 = temp_dir / "test2.db"
db2 = pycolmap.Database.open(str(db_path2))
cam_ids = {}
for face in ["front", "right", "back", "left", "up", "down"]:
    cam_ids[face] = db2.write_camera(pycolmap.Camera(model="PINHOLE", width=480, height=480, params=[240, 240, 240, 240]))

for st in range(2):
    for face in ["front", "right", "back", "left", "up", "down"]:
        img_name = f"face_{face}_frame_{st:06d}.jpg"
        db2.write_image(pycolmap.Image(name=img_name, camera_id=cam_ids[face]))

print("\nTrying apply_rig_config with face_{face}_frame_... image names (6 cameras):")
try:
    pycolmap.apply_rig_config(rigs, db2)
    print("SUCCESS! num_rigs:", db2.num_rigs(), "num_frames:", db2.num_frames())
except Exception as e:
    print("Failed:", e)
db2.close()

# Test 3: What about the current naming: frame_000000_face_front.jpg?
db_path3 = temp_dir / "test3.db"
db3 = pycolmap.Database.open(str(db_path3))
for st in range(2):
    for face in ["front", "right", "back", "left", "up", "down"]:
        img_name = f"frame_{st:06d}_face_{face}.jpg"
        db3.write_image(pycolmap.Image(name=img_name, camera_id=cam_ids[face]))

print("\nTrying apply_rig_config with frame_000000_face_... image names (6 cameras):")
try:
    pycolmap.apply_rig_config(rigs, db3)
    print("SUCCESS! num_rigs:", db3.num_rigs(), "num_frames:", db3.num_frames())
except Exception as e:
    print("Failed:", e)
db3.close()
