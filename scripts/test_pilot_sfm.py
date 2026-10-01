import json
import time
from pathlib import Path
from vscan.sfm.config import SfmConfig
from vscan.sfm.pipeline import SfmPipeline

manifest_path = Path("/home/rohit/projects/insta360_mapper/runs/run_frames_20260917_053006_a4cf20/outputs/pilot_manifest.json")
rig_path = Path("/home/rohit/projects/insta360_mapper/runs/run_frames_20260917_053006_a4cf20/outputs/rig_config.json")
images_dir = Path("/home/rohit/projects/insta360_mapper/runs/run_frames_20260917_053006_a4cf20/outputs/pilot_frames")
masks_dir = Path("/home/rohit/projects/insta360_mapper/runs/run_frames_20260917_053006_a4cf20/outputs/masks")

out_dir = Path("/home/rohit/projects/insta360_mapper/runs/test_pilot_sfm_output")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Manifest exists: {manifest_path.is_file()}")
print(f"Rig exists: {rig_path.is_file()}")
print(f"Images dir exists: {images_dir.is_dir()}")

# Configure for pilot run
# Use CUDA if available (GTX 1650), 6 CPU threads
cfg = SfmConfig(
    session_id="20260910_172939",
    run_id="run_sfm_pilot_test_001",
    output_dir=out_dir,
    input_manifest_path=manifest_path,
    rig_config_path=rig_path,
    images_dir=images_dir,
    masks_dir=masks_dir,
    device_requested="cuda",
    allow_cpu_fallback=False,
    num_threads=6,
    max_num_features=4096,
    temporal_overlap=10,
    quadratic_overlap=True,
    fix_intrinsics=True,
    fix_rig_extrinsics=True,
    is_synthetic=False,
)

pipeline = SfmPipeline(cfg)
t0 = time.time()
result = pipeline.run()
elapsed = time.time() - t0

print("==================================================")
print(f"Pilot SfM executed in {elapsed:.2f} seconds")
print(f"Software status: {result['manifest']['status']}")
print(f"Actual data status: {result['manifest']['actual_data_status']}")
print(f"Scale status: {result['manifest']['scale_status']}")
print(f"Total components: {result['components_count']}")
print(f"Primary component ID: {result['primary_component_id']}")
for comp in result['manifest']['components']:
    print(f"Component {comp['component_id']} (primary={comp['is_primary']}):")
    print(f"  Registered stations: {comp['num_registered_stations']} / {result['manifest']['total_panoramas_input']}")
    print(f"  Registered views: {comp['num_registered_views']} / {result['manifest']['total_views_input']}")
    print(f"  Landmarks: {comp['num_landmarks']}")
    print(f"  Mean reproj error: {comp['mean_reprojection_error_px']:.4f} px (RMSE: {comp['rmse_reprojection_error_px']:.4f} px)")
    print(f"  Mean track length: {comp['mean_track_length']:.2f}")
    print(f"  Mean triangulation angle: {comp['mean_triangulation_angle_deg']:.2f} deg")
print("==================================================")
