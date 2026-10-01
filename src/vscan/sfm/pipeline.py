"""Core Structure from Motion (SfM) pipeline implementation using PyCOLMAP 4.2.0."""

from __future__ import annotations

import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pycolmap

from vscan.frames.projection import (
    CUBEMAP_FACES,
    PinholeIntrinsics,
    ReprojectionFace,
    build_virtual_camera_poses,
    get_default_pinhole_intrinsics,
)
from vscan.frames.rig import export_rig_config_json, verify_rig_config_with_pycolmap
from vscan.schemas.jobs import CompletionMarkerSchema
from vscan.schemas.sfm import RelativeMapManifestSchema, SfmComponentSchema
from vscan.sfm.config import SfmConfig
from vscan.sfm.export import export_reconstruction_component
from vscan.sfm.matching import run_matching
from vscan.sfm.options import (
    build_feature_extraction_options,
    build_incremental_pipeline_options,
    log_effective_options,
    resolve_device,
)
from vscan.sfm.synthetic import generate_synthetic_sfm_dataset
from vscan.storage.artifacts import (
    compute_sha256_dict,
    compute_sha256_file,
    write_atomic_completion_marker,
)


class SfmPipeline:
    """Orchestrates feature extraction, rig setup, matching, and incremental reconstruction."""

    def __init__(self, config: SfmConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Execute the SfM pipeline stages adhering to all build contracts."""

        def _report(pct: float, step: str) -> None:
            if progress_callback:
                progress_callback(pct, step)
            print(f"[SfmPipeline] [{pct:.1f}%] {step}", flush=True)

        def _check_cancel(stage_name: str) -> None:
            if cancel_checker and cancel_checker():
                raise InterruptedError(f"Job cancellation requested during {stage_name}")

        _report(5.0, "Resolving compute device and logging effective options")
        device, gpu_info = resolve_device(self.config)
        effective_opts = log_effective_options(self.config, device)

        # 1. Dataset & Staging Setup
        _report(10.0, "Preparing image dataset and PyCOLMAP rig configuration")
        staged_dir = self.output_dir / "staged"
        staged_images_dir = staged_dir / "images"
        staged_masks_dir = staged_dir / "masks"
        staged_images_dir.mkdir(parents=True, exist_ok=True)
        staged_masks_dir.mkdir(parents=True, exist_ok=True)

        gt_stations: Optional[List[Dict[str, Any]]] = None
        source_manifest_hash: Optional[str] = None
        rig_config_hash: Optional[str] = None
        total_panos = 0
        total_views = 0
        actual_data_status = "UNKNOWN"

        if self.config.is_synthetic:
            # Generate synthetic room fixture
            synth_data = generate_synthetic_sfm_dataset(
                target_dir=staged_dir,
                num_stations=self.config.synthetic_stations_count,
            )
            rig_config_path = synth_data["rig_config_path"]
            manifest_path = synth_data["manifest_path"]
            gt_stations = synth_data["gt_stations"]
            images_source_dir = synth_data["images_dir"]
            masks_source_dir = synth_data["masks_dir"]

            with open(manifest_path, "r", encoding="utf-8") as f:
                man_json = json.load(f)
            total_panos = man_json.get("selected_pilot_frames_count", len(gt_stations))
            total_views = total_panos * 6
            actual_data_status = "SYNTHETIC_EASY_FIXTURE"
            source_manifest_hash = compute_sha256_file(manifest_path)
            rig_config_hash = compute_sha256_file(rig_config_path)
        else:
            # Load actual pilot manifest
            if not self.config.input_manifest_path or not self.config.input_manifest_path.is_file():
                raise FileNotFoundError(f"Pilot manifest not found: {self.config.input_manifest_path}")
            if not self.config.rig_config_path or not self.config.rig_config_path.is_file():
                raise FileNotFoundError(f"Rig config not found: {self.config.rig_config_path}")

            manifest_path = self.config.input_manifest_path
            rig_config_path = self.config.rig_config_path
            source_manifest_hash = compute_sha256_file(manifest_path)
            rig_config_hash = compute_sha256_file(rig_config_path)

            with open(manifest_path, "r", encoding="utf-8") as f:
                man_json = json.load(f)

            readiness = man_json.get("actual_data_readiness", "UNKNOWN")
            if readiness in ("UNRESOLVED_UNKNOWN_PROJECTION", "INVALID"):
                raise ValueError(
                    f"Contract START violation: Input imagery status is '{readiness}'. "
                    "Actual-data SfM requires passing projection/rig tests and ready actual imagery."
                )

            panos = man_json.get("pilot_panoramas", [])
            total_panos = len(panos)

            # Stage actual pilot images and masks
            base_dir = manifest_path.parent
            for p_item in panos:
                st_idx = p_item["panorama_index"]
                views = p_item.get("views", {})
                total_views += len(views)
                for cam_name, v_info in views.items():
                    src_img_rel = v_info["image_rel_path"]
                    src_img = base_dir / src_img_rel
                    if not src_img.is_file() and self.config.images_dir:
                        cand1 = self.config.images_dir / Path(src_img_rel).name
                        cand2 = self.config.images_dir / src_img_rel
                        if cand1.is_file():
                            src_img = cand1
                        elif cand2.is_file():
                            src_img = cand2

                    # Standard PyCOLMAP rig prefix format: face_{face}_frame_{station:06d}.jpg
                    prefix = cam_name if cam_name.startswith("face_") else f"face_{cam_name}"
                    staged_name = f"{prefix}_frame_{st_idx:06d}.jpg"
                    staged_dst = staged_images_dir / staged_name
                    if not staged_dst.exists() and src_img.is_file():
                        try:
                            os.symlink(src_img.resolve(), staged_dst)
                        except (OSError, NotImplementedError):
                            shutil.copy2(src_img, staged_dst)

                    # Stage mask
                    mask_rel = v_info.get("mask_rel_path")
                    if mask_rel:
                        src_mask = base_dir / mask_rel
                        if not src_mask.is_file() and self.config.masks_dir:
                            cand1 = self.config.masks_dir / Path(mask_rel).name
                            cand2 = self.config.masks_dir / mask_rel
                            if cand1.is_file():
                                src_mask = cand1
                            elif cand2.is_file():
                                src_mask = cand2
                        if src_mask.is_file():
                            # COLMAP expects mask matching image filename + .png
                            mask_dst = staged_masks_dir / f"{staged_name}.png"
                            if not mask_dst.exists():
                                try:
                                    os.symlink(src_mask.resolve(), mask_dst)
                                except (OSError, NotImplementedError):
                                    shutil.copy2(src_mask, mask_dst)

            images_source_dir = staged_images_dir
            masks_source_dir = staged_masks_dir
            actual_data_status = f"ACTUAL_DATA_PILOT_{total_panos}_STATIONS"

        # 2. Database & Camera Initialization
        _report(20.0, "Initializing SQLite database with fixed camera intrinsics")
        db_path = self.output_dir / "database.db"
        staged_files = sorted([f.name for f in images_source_dir.glob("*.jpg")])
        rig_configs = pycolmap.read_rig_config(str(rig_config_path))

        db_already_initialized = False
        if db_path.exists():
            try:
                with pycolmap.Database.open(str(db_path)) as test_db:
                    n_imgs = test_db.num_images()
                    n_cams = test_db.num_cameras()
                    if n_imgs == len(staged_files) and n_cams >= 1:
                        db_already_initialized = True
            except Exception:
                db_already_initialized = False

            if not db_already_initialized:
                # Partial or corrupted database must not be treated as success or reused
                print(f"[SfmPipeline] Removing unverified/partial database at {db_path}", flush=True)
                db_path.unlink(missing_ok=True)

        if not db_already_initialized:
            # Open db and create camera models
            with pycolmap.Database.open(str(db_path)) as db:
                # We assign one fixed camera per face
                for face in CUBEMAP_FACES:
                    face_prefix = f"face_{face.value}_"
                    matching_imgs = [name for name in staged_files if name.startswith(face_prefix)]
                    if not matching_imgs:
                        continue

                    # Pinhole intrinsics (fx=fy=W/2, cx=cy=W/2 for 90 deg FOV)
                    # Read sample image to get exact resolution
                    sample_img_path = images_source_dir / matching_imgs[0]
                    from PIL import Image
                    with Image.open(sample_img_path) as s_im:
                        img_w, img_h = s_im.size

                    fx = float(img_w) / 2.0
                    fy = float(img_h) / 2.0
                    cx = float(img_w) / 2.0
                    cy = float(img_h) / 2.0

                    cam = pycolmap.Camera(
                        model="PINHOLE",
                        width=img_w,
                        height=img_h,
                        params=[fx, fy, cx, cy],
                        has_prior_focal_length=True,
                    )
                    cam_id = db.write_camera(cam)

                    reader_opts = pycolmap.ImageReaderOptions()
                    reader_opts.existing_camera_id = cam_id
                    reader_opts.camera_model = "PINHOLE"
                    if self.config.use_masks and masks_source_dir.exists():
                        reader_opts.mask_path = str(masks_source_dir)

                    pycolmap.import_images(
                        database_path=str(db_path),
                        image_path=str(images_source_dir),
                        image_names=matching_imgs,
                        options=reader_opts,
                    )

                # Apply rig configuration
                _report(30.0, "Applying pinned PyCOLMAP rig configuration")
                pycolmap.apply_rig_config(rig_configs, db)

        _check_cancel("pre-extraction")

        # 3. Feature Extraction
        _report(40.0, f"Extracting SIFT features on {device.name.upper()} device with masks")
        extract_opts = build_feature_extraction_options(self.config, device)
        reader_opts = pycolmap.ImageReaderOptions()
        if self.config.use_masks and masks_source_dir.exists():
            reader_opts.mask_path = str(masks_source_dir)

        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(images_source_dir),
            reader_options=reader_opts,
            extraction_options=extract_opts,
            device=device,
        )

        _check_cancel("post-extraction")

        # 4. Feature Matching & Geometric Verification
        _report(60.0, f"Matching temporal neighbors and verifying geometry on {device.name.upper()}")
        loop_evidence = run_matching(
            database_path=db_path,
            config=self.config,
            device=device,
        )

        _check_cancel("post-matching")

        # 5. Incremental Reconstruction
        _report(75.0, "Running incremental mapping and bundle adjustment (fixed intrinsics/rig)")
        recon_dir = self.output_dir / "reconstruction"
        recon_dir.mkdir(exist_ok=True)
        pipe_opts = build_incremental_pipeline_options(self.config)

        reconstructions_dict = pycolmap.incremental_mapping(
            database_path=str(db_path),
            image_path=str(images_source_dir),
            output_path=str(recon_dir),
            options=pipe_opts,
        )

        _check_cancel("post-reconstruction")

        # 6. Multi-Component Analysis and Export
        _report(90.0, f"Exporting {len(reconstructions_dict)} component(s), poses, and tracks")
        component_summaries: List[SfmComponentSchema] = []
        accepted_reconstructions: Dict[int, Any] = {}
        primary_comp_id: Optional[int] = None
        max_reg_stations = -1

        weak_regions: List[Dict[str, Any]] = []

        # Retain all disconnected components explicitly, but reject degenerate zero-baseline components
        for cid, recon in reconstructions_dict.items():
            comp_summary, details = export_reconstruction_component(
                recon=recon,
                component_id=cid,
                output_dir=self.output_dir,
                is_primary=False,  # Assigned after finding max
            )

            # Contract: Pure rotation or all views from one station must not produce an accepted triangulated metric map.
            # Virtual faces do not count as baselines.
            # Reject any component where registered physical stations have zero translation baseline:
            # e.g., fewer than 2 registered physical stations, or max baseline <= 0.02m, or mean triangulation angle < 1.0 deg.
            station_centers = [st["nodal_center_world"] for st in details["station_poses"]]
            max_station_baseline = 0.0
            if len(station_centers) >= 2:
                for i in range(len(station_centers)):
                    for j in range(i + 1, len(station_centers)):
                        d = float(np.linalg.norm(np.array(station_centers[i]) - np.array(station_centers[j])))
                        if d > max_station_baseline:
                            max_station_baseline = d

            is_degenerate = (
                len(station_centers) < 2
                or max_station_baseline <= 0.02
                or comp_summary.mean_triangulation_angle_deg < 1.0
                or comp_summary.num_landmarks == 0
            )

            if is_degenerate:
                print(
                    f"[SfmPipeline] Rejecting degenerate component {cid}: "
                    f"registered_stations={len(station_centers)}, "
                    f"max_station_baseline={max_station_baseline:.4f}m, "
                    f"mean_triangulation_angle={comp_summary.mean_triangulation_angle_deg:.2f} deg, "
                    f"num_landmarks={comp_summary.num_landmarks}. "
                    "Rule: pure rotation or zero physical baseline cannot produce an accepted metric map.",
                    flush=True,
                )
                comp_dir = self.output_dir / f"component_{cid:02d}"
                shutil.rmtree(comp_dir, ignore_errors=True)
                continue

            accepted_reconstructions[cid] = recon
            component_summaries.append(comp_summary)
            if comp_summary.num_registered_stations > max_reg_stations:
                max_reg_stations = comp_summary.num_registered_stations
                primary_comp_id = cid

            # Identify weak regions: points with small triangulation angles or low track lengths
            for lm in details["landmarks"]:
                if lm["track_length"] < 3 or lm["reprojection_error_px"] > 3.0:
                    weak_regions.append({
                        "component_id": cid,
                        "point3d_id": lm["point3d_id"],
                        "track_length": lm["track_length"],
                        "reprojection_error_px": lm["reprojection_error_px"],
                    })

        # Set primary flag on primary component
        if primary_comp_id is not None:
            for cs in component_summaries:
                if cs.component_id == primary_comp_id:
                    cs.is_primary = True

        # Classify actual data result
        if not component_summaries:
            actual_data_status = "RECONSTRUCTION_EMPTY_NO_MODELS"
        elif len(component_summaries) == 1 and max_reg_stations >= total_panos * 0.8:
            actual_data_status = "RECONSTRUCTION_CONNECTED_ACCEPTABLE"
        elif len(component_summaries) > 1:
            actual_data_status = f"RECONSTRUCTION_DISCONNECTED_{len(component_summaries)}_COMPONENTS"
        else:
            actual_data_status = f"RECONSTRUCTION_PARTIAL_{max_reg_stations}_OF_{total_panos}_STATIONS"

        # 7. Immutability, Artifact Hashes & Manifest
        _report(95.0, "Calculating artifact SHA-256 hashes and saving relative map manifest")
        artifact_files = [
            db_path,
            rig_config_path,
        ]
        # Include component export files
        for cs in component_summaries:
            cid = cs.component_id
            comp_d = self.output_dir / f"component_{cid:02d}"
            for name in ["camera_poses.json", "station_poses.json", "tracks.json", "component_summary.json"]:
                p_file = comp_d / name
                if p_file.is_file():
                    artifact_files.append(p_file)

        artifact_hashes: Dict[str, str] = {}
        for p_file in artifact_files:
            p_file = Path(p_file)
            if p_file.is_file():
                try:
                    rel_name = str(p_file.relative_to(self.output_dir))
                except ValueError:
                    rel_name = p_file.name
                artifact_hashes[rel_name] = compute_sha256_file(p_file)

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        map_id = f"map_{self.config.session_id}_{self.config.run_id[:8]}_rel"

        manifest = RelativeMapManifestSchema(
            schema_version="1.0.0",
            session_id=self.config.session_id,
            map_id=map_id,
            run_id=self.config.run_id,
            created_at_utc=now_utc,
            status="STAGE_06_COMPLETE",
            actual_data_status=actual_data_status,
            scale_status="NOT_ESTIMATED",
            device_requested=self.config.device_requested,
            device_used=device.name,
            gpu_info=gpu_info,
            effective_options=effective_opts,
            source_manifest_hash=source_manifest_hash,
            rig_config_hash=rig_config_hash,
            total_panoramas_input=total_panos,
            total_views_input=total_views,
            primary_component_id=primary_comp_id,
            components=component_summaries,
            loop_evidence=loop_evidence,
            weak_regions=weak_regions[:50],  # Bounded top weak regions
            artifact_hashes=artifact_hashes,
        )

        manifest_file = self.output_dir / "relative_map_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        # Write atomic completion marker adhering strictly to CompletionMarkerSchema
        marker_outputs: Dict[str, str] = {
            "relative_map_manifest.json": compute_sha256_file(manifest_file),
        }
        marker_sizes: Dict[str, int] = {
            "relative_map_manifest.json": manifest_file.stat().st_size,
        }
        if db_path.is_file():
            marker_outputs["database.db"] = compute_sha256_file(db_path)
            marker_sizes["database.db"] = db_path.stat().st_size

        for cs in component_summaries:
            cid = cs.component_id
            comp_d = self.output_dir / f"component_{cid:02d}"
            for name in ["camera_poses.json", "station_poses.json", "tracks.json", "component_summary.json"]:
                p_file = comp_d / name
                if p_file.is_file():
                    rel_p = f"component_{cid:02d}/{name}"
                    marker_outputs[rel_p] = compute_sha256_file(p_file)
                    marker_sizes[rel_p] = p_file.stat().st_size

        completion_cache_key = compute_sha256_dict({
            "stage": "sfm",
            "session_id": self.config.session_id,
            "run_id": self.config.run_id,
            "manifest_hash": compute_sha256_file(manifest_file),
        })

        marker = CompletionMarkerSchema(
            schema_version="1.0.0",
            run_id=self.config.run_id,
            stage="sfm",
            session_id=self.config.session_id,
            completed_at_utc=now_utc,
            cache_key=completion_cache_key,
            output_checksums=marker_outputs,
            output_sizes_bytes=marker_sizes,
        )
        write_atomic_completion_marker(marker, self.output_dir)

        _report(100.0, f"Stage 06 SfM relative mapping complete: {len(component_summaries)} component(s)")

        return {
            "map_id": map_id,
            "manifest_path": manifest_file,
            "manifest": manifest.model_dump(mode="json"),
            "primary_component_id": primary_comp_id,
            "components_count": len(component_summaries),
            "reconstructions": accepted_reconstructions,
            "gt_stations": gt_stations,
        }
