import tempfile
import time
from pathlib import Path
from vscan.sfm.config import SfmConfig
from vscan.sfm.pipeline import SfmPipeline
from vscan.sfm.synthetic import align_sim3_umeyama
import numpy as np

with tempfile.TemporaryDirectory() as td:
    out_dir = Path(td) / "sfm_out"
    cfg = SfmConfig(
        session_id="synth_test_session",
        run_id="run_sfm_test_001",
        output_dir=out_dir,
        is_synthetic=True,
        synthetic_stations_count=5,
        device_requested="cpu",
        num_threads=4,
        temporal_overlap=12,
        quadratic_overlap=True,
    )
    pipeline = SfmPipeline(cfg)
    t0 = time.time()
    result = pipeline.run()
    elapsed = time.time() - t0
    print(f"Pipeline executed in {elapsed:.2f}s")
    print(f"Status: {result['manifest']['status']}")
    print(f"Components count: {result['components_count']}")
    print(f"Primary component ID: {result['primary_component_id']}")
    for comp in result['manifest']['components']:
        print(f"Comp {comp['component_id']}: reg_stations={comp['num_registered_stations']}, reg_views={comp['num_registered_views']}, lm={comp['num_landmarks']}, reproj_rmse={comp['rmse_reprojection_error_px']:.4f}px, mean_angle={comp['mean_triangulation_angle_deg']:.2f} deg")

    # Check Sim(3) alignment on ground truth
    gt_stations = result["gt_stations"]
    primary_id = result["primary_component_id"]
    if primary_id is not None and primary_id in result["reconstructions"]:
        recon = result["reconstructions"][primary_id]
        recovered_centers = []
        gt_centers = []

        # Build name to image mapping
        name_to_im = {im.name: im for im in recon.images.values() if im.has_pose}

        for st in gt_stations:
            idx = st["station_index"]
            img_name = f"face_front_frame_{idx:06d}.jpg"
            if img_name in name_to_im:
                im = name_to_im[img_name]
                C = im.cam_from_world().inverse().translation
                recovered_centers.append(list(C))
                gt_centers.append(st["camera_center"])

        print(f"Registered stations with front pose: {len(recovered_centers)} / {len(gt_stations)}")
        if len(recovered_centers) >= 3:
            s, R, t, rmse = align_sim3_umeyama(np.array(recovered_centers), np.array(gt_centers))
            print(f"Sim(3) alignment scale={s:.4f}, camera-center RMSE={rmse:.6f} m")
            room_diag = float(np.sqrt(4.0**2 + 3.0**2 + 6.0**2))  # 7.81025m
            rmse_pct = (rmse / room_diag) * 100.0
            print(f"Camera RMSE % of room diagonal ({room_diag:.2f}m): {rmse_pct:.4f}% (Threshold: < 2.0%)")
            assert rmse_pct < 2.0, f"RMSE {rmse_pct:.2f}% exceeds 2% threshold"
            print(">>> VERIFICATION SUCCESS: Camera RMSE below 2% of room diagonal! <<<")
