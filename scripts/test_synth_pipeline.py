import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pycolmap

from vscan.frames.synthetic import SyntheticRoom
from vscan.frames.projection import get_default_pinhole_intrinsics, CUBEMAP_ROTATIONS, CUBEMAP_FACES

def test_synth_pipeline():
    room = SyntheticRoom()
    intrinsics = get_default_pinhole_intrinsics(320, 320, 90.0)

    # 5 stations along Z
    stations = [
        np.array([ 0.0, 0.0, -0.8]),
        np.array([ 0.0, 0.0, -0.4]),
        np.array([ 0.0, 0.0,  0.0]),
        np.array([ 0.0, 0.0,  0.4]),
        np.array([ 0.0, 0.0,  0.8]),
    ]

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        img_dir = p / "images"
        img_dir.mkdir()

        # All 6 cubemap faces
        all_faces = CUBEMAP_FACES  # front, right, back, left, up, down
        print(f"Rendering {len(stations)} stations x {len(all_faces)} faces...", flush=True)

        image_names = []
        for s_idx, s_pos in enumerate(stations):
            st_suffix = f"station_{s_idx:03d}.jpg"
            for face in all_faces:
                R_face = CUBEMAP_ROTATIONS[face]
                rgb, _ = room.render_analytic_pinhole(s_pos, R_face, intrinsics)
                name = f"{face.value}_{st_suffix}"
                Image.fromarray(rgb).save(img_dir / name)
                image_names.append(name)

        db_path = p / "database.db"
        output_dir = p / "output"
        output_dir.mkdir()

        # Create rig config JSON
        from vscan.frames.rig import export_rig_config_json
        from vscan.frames.projection import build_virtual_camera_poses
        poses = build_virtual_camera_poses(intrinsics)
        rig_path = p / "rig_config.json"
        export_rig_config_json(poses, rig_path)
        rig_configs = pycolmap.read_rig_config(str(rig_path))

        print("Importing images with fixed camera intrinsics...", flush=True)
        with pycolmap.Database.open(str(db_path)) as db:
            for face in all_faces:
                prefix = f"{face.value}_"
                cam = pycolmap.Camera(
                    model="PINHOLE",
                    width=320,
                    height=320,
                    params=[160.0, 160.0, 160.0, 160.0],
                    has_prior_focal_length=True,
                )
                cid = db.write_camera(cam)
                reader_opts = pycolmap.ImageReaderOptions()
                reader_opts.existing_camera_id = cid
                reader_opts.camera_model = "PINHOLE"
                imgs_for_face = [name for name in image_names if name.startswith(prefix)]
                pycolmap.import_images(str(db_path), str(img_dir), image_names=imgs_for_face, options=reader_opts)

            print("Applying rig configuration...", flush=True)
            pycolmap.apply_rig_config(rig_configs, db)
            print(f"DB configured: {db.num_rigs()} rigs, {db.num_frames()} frames.", flush=True)

        print("Extracting features...", flush=True)
        extract_opts = pycolmap.FeatureExtractionOptions()
        extract_opts.use_gpu = False
        extract_opts.num_threads = 4
        extract_opts.sift.max_num_features = 2048

        pycolmap.extract_features(
            str(db_path),
            str(img_dir),
            extraction_options=extract_opts,
        )

        print("Matching features...", flush=True)
        match_opts = pycolmap.FeatureMatchingOptions()
        match_opts.use_gpu = False
        match_opts.num_threads = 4

        seq_opts = pycolmap.SequentialPairingOptions()
        seq_opts.overlap = 12
        seq_opts.quadratic_overlap = True

        pycolmap.match_sequential(
            str(db_path),
            matching_options=match_opts,
            pairing_options=seq_opts,
        )

        with pycolmap.Database.open(str(db_path)) as db:
            print(f"Verified image pairs in DB: {db.num_verified_image_pairs()}", flush=True)
            print(f"Total matches: {db.num_matches()}", flush=True)

        print("Running incremental mapping...", flush=True)
        pipe_opts = pycolmap.IncrementalPipelineOptions()
        pipe_opts.num_threads = 4
        pipe_opts.ba_refine_focal_length = False
        pipe_opts.ba_refine_principal_point = False
        pipe_opts.ba_refine_extra_params = False
        pipe_opts.ba_refine_sensor_from_rig = False
        pipe_opts.mapper.abs_pose_refine_focal_length = False
        pipe_opts.mapper.abs_pose_refine_extra_params = False
        pipe_opts.mapper.init_min_tri_angle = 4.0
        pipe_opts.mapper.init_min_num_inliers = 15
        pipe_opts.mapper.abs_pose_min_num_inliers = 10
        pipe_opts.min_model_size = 5

        maps = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(img_dir),
            output_path=str(output_dir),
            options=pipe_opts,
        )

        print(f"Reconstruction finished! Produced {len(maps)} component(s).", flush=True)
        for cid, recon in maps.items():
            print(f"Component {cid}: num_reg_frames={recon.num_reg_frames()}, num_reg_images={recon.num_reg_images()}, num_points3D={recon.num_points3D()}", flush=True)
            print(f"  Mean reproj error: {recon.compute_mean_reprojection_error():.4f} px", flush=True)
            print(f"  Mean track length: {recon.compute_mean_track_length():.2f}", flush=True)
            for fid in recon.reg_frame_ids:
                fr = recon.frame(fid)
                print(f"    Registered Frame {fid}: has_pose={fr.has_pose}", flush=True)

if __name__ == "__main__":
    test_synth_pipeline()
