"""Comprehensive unit tests for vscan versioned schemas and coordinate conventions."""

import pytest
import numpy as np
from vscan.schemas import (
    CoordinateFrame,
    PoseMatrix,
    Units,
    InputAssetSchema,
    SourceFrameSchema,
    ClockMappingSchema,
    CameraModel,
    CameraRigSchema,
    CalibrationSchema,
    PoseSchema,
    LandmarkSchema,
    MapVersionSchema,
    SurfaceSchema,
    DepthMapSchema,
    MeshSchema,
    AnnotationSchema,
    MeasurementSchema,
    JobRecordSchema,
    CompletionMarkerSchema,
    JobStatus,
    ErrorClassification,
    SCHEMA_VERSION,
    InputManifestSchema,
    SensorSummarySchema,
)


def test_pose_matrix_so3_validation():
    """Verify SO(3) constraint enforcement and row-major serialization."""
    # Identity is valid SO(3)
    p_ident = PoseMatrix.identity("body", "world")
    assert len(p_ident.rotation_row_major) == 9
    assert p_ident.transform_point([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    # Valid 90 deg rotation around Z
    # [ 0, -1, 0]
    # [ 1,  0, 0]
    # [ 0,  0, 1]
    R_z90 = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    t = [10.0, 0.0, 0.0]
    p_rot = PoseMatrix(
        from_frame="body",
        to_frame="world",
        rotation_row_major=R_z90,
        translation=t,
    )
    # Transforming [1, 0, 0] in body:
    # R * [1, 0, 0]^T + t = [0, 1, 0]^T + [10, 0, 0]^T = [10, 1, 0]
    p_world = p_rot.transform_point([1.0, 0.0, 0.0])
    assert pytest.approx(p_world[0]) == 10.0
    assert pytest.approx(p_world[1]) == 1.0
    assert pytest.approx(p_world[2]) == 0.0

    # Invalid rotation: determinant not 1 (reflection matrix)
    R_reflect = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="not proper SO\\(3\\)"):
        PoseMatrix(
            from_frame="b",
            to_frame="w",
            rotation_row_major=R_reflect,
            translation=[0.0, 0.0, 0.0],
        )

    # Invalid rotation: non-orthogonal
    R_non_ortho = [1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="not proper SO\\(3\\)|not orthogonal"):
        PoseMatrix(
            from_frame="b",
            to_frame="w",
            rotation_row_major=R_non_ortho,
            translation=[0.0, 0.0, 0.0],
        )


def test_pose_matrix_inverse_and_composition():
    """Verify T_A_B transformation conventions and inverse/compose."""
    # T_A_B: B -> A
    # T_B_C: C -> B
    # T_A_C = T_A_B * T_B_C
    T_A_B = PoseMatrix(
        from_frame="B",
        to_frame="A",
        rotation_row_major=[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        translation=[1.0, 2.0, 3.0],
    )
    T_B_C = PoseMatrix(
        from_frame="C",
        to_frame="B",
        rotation_row_major=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation=[0.0, 5.0, 0.0],
    )

    T_A_C = T_A_B.compose(T_B_C)
    assert T_A_C.from_frame == "C"
    assert T_A_C.to_frame == "A"

    point_in_C = [2.0, 1.0, 4.0]
    point_in_B = T_B_C.transform_point(point_in_C)
    point_in_A_step = T_A_B.transform_point(point_in_B)
    point_in_A_composed = T_A_C.transform_point(point_in_C)

    assert pytest.approx(point_in_A_step) == point_in_A_composed

    # Inverse test: T_B_A transforms point in A back to B
    T_B_A = T_A_B.inverse()
    p_roundtrip = T_B_A.transform_point(point_in_A_step)
    assert pytest.approx(p_roundtrip) == point_in_B


def test_clock_mapping_integer_timestamps():
    """Verify integer large raw timestamps and floating point local relative seconds."""
    mapping = ClockMappingSchema(
        session_id="synth_session_01",
        clock_source="device_ticks_ns",
        t0_raw_ns=1726480000000000000,  # 1.726 billion seconds in ns
        time_scale_factor=1e-9,
    )
    assert isinstance(mapping.t0_raw_ns, int)

    raw_stamp = 1726480005500000000  # +5.5s
    rel_s = mapping.raw_to_relative_s(raw_stamp)
    assert pytest.approx(rel_s) == 5.5
    assert isinstance(rel_s, float)

    reconstructed_raw = mapping.relative_to_raw_ns(rel_s)
    assert isinstance(reconstructed_raw, int)
    assert reconstructed_raw == raw_stamp


def test_camera_rig_and_provenance():
    """Verify camera axes convention and multi-camera rig validation."""
    cam0 = CameraModel(
        camera_id="cam0",
        model_name="OPENCV_FISHEYE",
        width=1920,
        height=1920,
        params=[960.0, 960.0, 0.0, 0.0],
        T_rig_cam=PoseMatrix.identity("cam0", "rig"),
        camera_axes_convention="OPENCV (+X right, +Y down, +Z forward)",
    )
    rig = CameraRigSchema(
        rig_id="insta360_x3_rig",
        session_id="synth_session_01",
        cameras=[cam0],
        calibration_provenance="factory_eeprom",
    )
    assert rig.schema_version == SCHEMA_VERSION
    assert len(rig.cameras) == 1
    assert rig.cameras[0].camera_axes_convention == "OPENCV (+X right, +Y down, +Z forward)"


def test_map_version_conventions():
    """Verify relative arbitrary maps vs metric gravity-aligned right-handed Z-up maps."""
    # Relative unscaled map
    rel_map = MapVersionSchema(
        map_id="map_session_v1_rel",
        session_id="synth_session_01",
        map_type="relative_unscaled",
        gravity_aligned=False,
        coordinate_frame=CoordinateFrame.WORLD_RELATIVE_UNSCALED,
        units=Units.ARBITRARY,
        provenance="pycolmap_sfm",
        is_immutable=True,
        exported_at_utc="2026-09-16T12:00:00Z",
    )
    assert rel_map.units == Units.ARBITRARY
    assert not rel_map.gravity_aligned

    # Metric gravity-aligned map
    metric_map = MapVersionSchema(
        map_id="map_session_v2_metric",
        session_id="synth_session_01",
        parent_map_id="map_session_v1_rel",
        map_type="metric_gravity_aligned",
        gravity_aligned=True,
        coordinate_frame=CoordinateFrame.WORLD_METRIC_GRAVITY_ALIGNED,
        units=Units.METERS,
        scale_factor_meters_per_unit=1.0,
        provenance="gtsam_vio_alignment",
        is_immutable=True,
        exported_at_utc="2026-09-16T12:30:00Z",
    )
    assert metric_map.units == Units.METERS
    assert metric_map.gravity_aligned
    assert metric_map.coordinate_frame == CoordinateFrame.WORLD_METRIC_GRAVITY_ALIGNED


def test_pose_schema_carries_map_id_and_provenance():
    """Verify every pose carries map ID, units, and calibration provenance."""
    pose = PoseSchema(
        pose_id="pose_001",
        from_frame="cam0",
        to_frame="map",
        map_id="map_session_v2_metric",
        units=Units.METERS,
        calibration_provenance="factory_rig_calib_v1",
        timestamp_raw_ns=1726480001000000000,
        timestamp_relative_s=1.0,
        transform=PoseMatrix.identity("cam0", "map"),
    )
    assert pose.map_id == "map_session_v2_metric"
    assert pose.units == Units.METERS
    assert pose.calibration_provenance == "factory_rig_calib_v1"


def test_depth_map_schema_carries_map_id_and_provenance():
    """Verify depth map schema carries mandatory map ID, units, and calibration provenance."""
    depth = DepthMapSchema(
        depth_id="depth_001",
        map_id="map_session_v2_metric",
        units=Units.METERS,
        calibration_provenance="stereo_dense_matcher_v1",
        frame_id="frame_001",
        camera_id="cam0",
        width=1920,
        height=1920,
        file_path="depth/cam0_001.npy",
        min_depth=0.2,
        max_depth=15.0,
        scale_factor=1.0,
    )
    assert depth.map_id == "map_session_v2_metric"
    assert depth.units == Units.METERS
    assert depth.calibration_provenance == "stereo_dense_matcher_v1"
    assert depth.width == 1920


def test_mesh_schema_carries_map_id_and_provenance():
    """Verify mesh schema carries mandatory map ID, units, and calibration provenance."""
    mesh = MeshSchema(
        mesh_id="mesh_room_01",
        map_id="map_session_v2_metric",
        units=Units.METERS,
        calibration_provenance="poisson_recon_v1",
        mesh_type="triangular_mesh",
        file_path="meshes/room.ply",
        vertex_count=50000,
        face_count=98000,
        format="ply",
        bounding_box_min=[-5.0, -4.0, 0.0],
        bounding_box_max=[5.0, 4.0, 3.0],
        exported_at_utc="2026-09-16T15:00:00Z",
    )
    assert mesh.map_id == "map_session_v2_metric"
    assert mesh.units == Units.METERS
    assert mesh.calibration_provenance == "poisson_recon_v1"
    assert mesh.vertex_count == 50000


def test_database_manager_asset_and_map_crud(tmp_path):
    """Verify DatabaseManager CRUD operations for InputAssetSchema and MapVersionSchema."""
    from vscan.storage.database import DatabaseManager

    db_path = tmp_path / "test_app.sqlite3"
    db = DatabaseManager(db_path)

    # 1. Input asset CRUD
    asset = InputAssetSchema(
        asset_id="asset_test_01",
        session_id="synth_session_01",
        file_path="data/raw_test.mp4",
        file_size_bytes=1024000,
        file_sha256="abc123def456",
        asset_type="raw_video_insv",
        metadata={"fps": 30.0},
        created_at_utc="2026-09-16T12:00:00Z",
    )
    db.insert_input_asset(asset)
    fetched_asset = db.get_input_asset("asset_test_01")
    assert fetched_asset is not None
    assert fetched_asset.asset_id == "asset_test_01"
    assert fetched_asset.file_sha256 == "abc123def456"

    assets_list = db.list_input_assets("synth_session_01")
    assert len(assets_list) == 1
    assert assets_list[0].asset_id == "asset_test_01"

    # 2. Map CRUD
    map_rec = MapVersionSchema(
        map_id="map_synth_v1",
        session_id="synth_session_01",
        map_version=1,
        map_type="metric_gravity_aligned",
        gravity_aligned=True,
        coordinate_frame=CoordinateFrame.WORLD_METRIC_GRAVITY_ALIGNED,
        units=Units.METERS,
        scale_factor_meters_per_unit=1.0,
        provenance="gtsam_vio",
        exported_at_utc="2026-09-16T12:00:00Z",
    )
    db.insert_map(map_rec)
    fetched_map = db.get_map("map_synth_v1")
    assert fetched_map is not None
    assert fetched_map.map_id == "map_synth_v1"
    assert fetched_map.units == Units.METERS
    assert fetched_map.gravity_aligned is True

    maps_list = db.list_maps("synth_session_01")
    assert len(maps_list) == 1
    assert maps_list[0].map_id == "map_synth_v1"


def test_input_manifest_schema_validation():
    """Verify InputManifestSchema enforces scale_status == NOT_ESTIMATED and schema version."""
    manifest = InputManifestSchema(
        schema_version=SCHEMA_VERSION,
        session_id="20260910_172939",
        created_at_utc="2026-09-16T12:00:00Z",
        input_directory="/data/2026-09-10",
        storage_mode="reference",
        total_input_bytes=1000000,
        scale_status="NOT_ESTIMATED",
        status="COMPLETE",
    )
    assert manifest.scale_status == "NOT_ESTIMATED"
    assert manifest.schema_version == "1.0.0"
    dumped = manifest.model_dump(mode="json")
    assert dumped["scale_status"] == "NOT_ESTIMATED"
