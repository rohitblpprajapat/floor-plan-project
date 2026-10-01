# Data Contracts & Versioned Schemas Specification (v2)

**Specification Version**: `1.0.0`  
**Application**: `vscan` (Local Insta360 Reconstruction & Annotation Application)  
**Status**: Immutable Operational Baseline  

---

## 1. Core Principles & Geometric Conventions

### 1.1 Transformation Direction ($T_{A\_B}$)
All rigid-body spatial transformations adhere to the forward mapping convention:
$$p_A = T_{A\_B} \cdot p_B = R_{A\_B} \cdot p_B + t_{A\_B}$$
Where:
- $p_B$ is a point vector expressed in Source Frame $B$.
- $p_A$ is the same point vector transformed into Target Frame $A$.
- $T_{A\_B}$ contains rotation $R_{A\_B} \in \mathrm{SO}(3)$ and translation $t_{A\_B} \in \mathbb{R}^3$.
- Inversion yields $T_{B\_A} = (T_{A\_B})^{-1}$:
  $$R_{B\_A} = R_{A\_B}^T, \quad t_{B\_A} = -R_{A\_B}^T \cdot t_{A\_B}$$
- Composition strictly checks frame continuity: $T_{A\_C} = T_{A\_B} \cdot T_{B\_C}$.

### 1.2 Matrix Serialization
- Matrices are serialized in **explicit row-major order** (`order="C"`).
- A $3 \times 3$ rotation matrix is represented as a flat 9-float sequence:
  $$[R_{00}, R_{01}, R_{02}, R_{10}, R_{11}, R_{12}, R_{20}, R_{21}, R_{22}]$$
- A $4 \times 4$ homogeneous matrix is serialized as a flat 16-float sequence:
  $$\begin{bmatrix} R_{00} & R_{01} & R_{02} & t_x \\ R_{10} & R_{11} & R_{12} & t_y \\ R_{20} & R_{21} & R_{22} & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 1.3 SO(3) Rotation Validity
Rotations must belong to the Special Orthogonal Group $\mathrm{SO}(3)$:
$$\det(R) = +1.0 \pm 10^{-3}, \quad \|R^T R - I\|_\infty < 10^{-3}$$
Reflection matrices ($\det(R) = -1$) and non-orthogonal matrices are rejected during validation.

### 1.4 Camera Axes Convention (OpenCV Standard)
Camera optical coordinate systems strictly follow the OpenCV coordinate convention:
- **+X**: Points to the **right** in the image plane.
- **+Y**: Points **down** in the image plane.
- **+Z**: Points **forward** along the optical axis into the scene (positive depth).

### 1.5 World Reference Frames & Map Types
1. **Relative Reconstruction Maps (`world_relative_unscaled`)**:
   - Units: `arbitrary`.
   - Gravity: `gravity_aligned = False`. No assumed vertical or world orientation.
   - Scale: Unscaled relative units from initial bundle adjustment.
2. **Metric Gravity-Aligned Maps (`world_metric_gravity_aligned`)**:
   - Units: `meters`.
   - Gravity: `gravity_aligned = True`.
   - Coordinate Frame: Right-handed Z-up:
     - **+X**: East / local right horizontal axis.
     - **+Y**: North / local forward horizontal axis.
     - **+Z**: Up (anti-parallel to verified local gravity vector).

### 1.6 Provenance and Identification
- Every pose, surface, landmark, depth, and mesh **must** explicitly declare:
  1. `map_id`: Identifier of the reconstruction map version it belongs to.
  2. `units`: Either `meters` or `arbitrary`.
  3. `calibration_provenance`: Provenance descriptor of the sensor calibration.

### 1.7 Timestamp Representation
- **Raw Large Timestamps**: Stored strictly as **64-bit integers** (`timestamp_raw_ns: int`) in device or clock domains (nanoseconds) to prevent floating-point precision loss.
- **Numerical Work**: Numerical optimization, spline interpolation, and SfM/VIO solvers use **local relative seconds** (`timestamp_relative_s: float`) defined as $\Delta t = (t_{\mathrm{raw}} - t_0) \times 10^{-9}$.

---

## 2. Versioned Schema Definitions (`schema_version = "1.0.0"`)

### 2.1 Input Assets (`InputAssetSchema`)
Tracks immutable raw captures (video, LRV, IMU, calibration files):
```json
{
  "schema_version": "1.0.0",
  "asset_id": "asset_video_front_01",
  "session_id": "2026-08-31",
  "file_path": "data/2026-08-31/LRV_20260831_163825_01_081.mp4",
  "file_size_bytes": 104857600,
  "file_sha256": "3a7b9c...",
  "asset_type": "raw_video_lrv",
  "metadata": {"fps": 29.97, "duration_s": 42.5},
  "created_at_utc": "2026-09-16T10:00:00Z"
}
```

### 2.2 Source Frames (`SourceFrameSchema`)
Tracks extracted image frames from video streams:
```json
{
  "schema_version": "1.0.0",
  "frame_id": "frame_20260831_000142",
  "session_id": "2026-08-31",
  "index": 142,
  "camera_id": "cam0",
  "timestamp_raw_ns": 1726480005000000000,
  "timestamp_relative_s": 5.0,
  "image_path": "data/sessions/2026-08-31/frames/cam0_000142.png",
  "width": 1920,
  "height": 1920,
  "asset_id": "asset_video_front_01",
  "exposure_time_s": 0.0033
}
```

### 2.3 Clock Mapping (`ClockMappingSchema`)
Maps raw large device clocks to continuous local relative seconds:
```json
{
  "schema_version": "1.0.0",
  "session_id": "2026-08-31",
  "clock_source": "insta360_sensor_ticks_ns",
  "t0_raw_ns": 1726480000000000000,
  "time_scale_factor": 1e-9,
  "sync_anchors": []
}
```

### 2.4 Camera Rig & Calibration (`CameraRigSchema`, `CalibrationSchema`)
Models fixed multi-camera rigs (e.g. dual fisheye) and IMU extrinsics:
```json
{
  "schema_version": "1.0.0",
  "calibration_id": "calib_insta360_x3_factory",
  "session_id": "2026-08-31",
  "provenance": "factory_eeprom_dual_fisheye",
  "rig": {
    "schema_version": "1.0.0",
    "rig_id": "insta360_rig",
    "session_id": "2026-08-31",
    "cameras": [
      {
        "camera_id": "cam0",
        "model_name": "OPENCV_FISHEYE",
        "width": 1920,
        "height": 1920,
        "params": [960.0, 960.0, 960.0, 960.0, 0.0, 0.0, 0.0, 0.0],
        "T_rig_cam": {
          "from_frame": "cam0",
          "to_frame": "rig",
          "rotation_row_major": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
          "translation": [0.0, 0.0, 0.0]
        },
        "camera_axes_convention": "OPENCV (+X right, +Y down, +Z forward)"
      }
    ],
    "calibration_provenance": "factory_eeprom"
  },
  "T_rig_imu": null,
  "gravity_magnitude_m_s2": 9.80665
}
```

### 2.5 6-DoF Poses (`PoseSchema`)
Defines estimated poses of camera/rig relative to map:
```json
{
  "schema_version": "1.0.0",
  "pose_id": "pose_20260831_000142",
  "from_frame": "rig",
  "to_frame": "map",
  "map_id": "map_20260831_v2_metric",
  "units": "meters",
  "calibration_provenance": "factory_eeprom",
  "timestamp_raw_ns": 1726480005000000000,
  "timestamp_relative_s": 5.0,
  "transform": {
    "from_frame": "rig",
    "to_frame": "map",
    "rotation_row_major": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "translation": [1.5, 3.2, 1.1]
  }
}
```

### 2.6 Map Versions (`MapVersionSchema`)
Immutable exported reconstruction maps:
```json
{
  "schema_version": "1.0.0",
  "map_id": "map_20260831_v2_metric",
  "session_id": "2026-08-31",
  "parent_map_id": "map_20260831_v1_rel",
  "map_version": 2,
  "map_type": "metric_gravity_aligned",
  "gravity_aligned": true,
  "coordinate_frame": "world_metric_gravity_aligned",
  "units": "meters",
  "scale_factor_meters_per_unit": 1.0,
  "provenance": "pycolmap_sfm_plus_gtsam_vio_alignment",
  "is_immutable": true,
  "num_poses": 450,
  "num_landmarks": 12840,
  "exported_at_utc": "2026-09-16T12:00:00Z"
}
```

### 2.7 Surfaces (`SurfaceSchema`)
Fitted planes, floors, ceilings, walls, and bounding meshes:
```json
{
  "schema_version": "1.0.0",
  "surface_id": "surface_floor_main",
  "map_id": "map_20260831_v2_metric",
  "units": "meters",
  "calibration_provenance": "factory_eeprom",
  "surface_type": "floor",
  "plane_equation_abcd": [0.0, 0.0, 1.0, 0.0],
  "polygon_vertices_xyz": [
    [0.0, 0.0, 0.0],
    [5.0, 0.0, 0.0],
    [5.0, 4.0, 0.0],
    [0.0, 4.0, 0.0]
  ],
  "confidence": 0.98
}
```

### 2.8 Annotations & Observations (`AnnotationSchema`)
Anchors 3D annotations while maintaining persistent 2D source-image observations:
```json
{
  "schema_version": "1.0.0",
  "annotation_id": "anno_door_entrance",
  "map_id": "map_20260831_v2_metric",
  "session_id": "2026-08-31",
  "label": "Front Entrance Door",
  "category": "door",
  "source_observations": [
    {
      "frame_id": "frame_20260831_000142",
      "camera_id": "cam0",
      "pixel_coords": [1024.5, 840.2],
      "normalized_ray": [0.05, -0.12, 0.99]
    }
  ],
  "anchor_xyz_3d": [2.4, 4.1, 1.0],
  "units": "meters",
  "metadata": {"width_m": 0.9, "height_m": 2.1},
  "created_at_utc": "2026-09-16T12:15:00Z"
}
```

### 2.9 Measurements (`MeasurementSchema`)
Physical metric measurements derived between annotations or surfaces:
```json
{
  "schema_version": "1.0.0",
  "measurement_id": "meas_room_width",
  "map_id": "map_20260831_v2_metric",
  "session_id": "2026-08-31",
  "measurement_type": "point_to_point_distance",
  "source_annotation_ids": ["anno_wall_west", "anno_wall_east"],
  "start_xyz": [0.0, 2.0, 1.0],
  "end_xyz": [5.0, 2.0, 1.0],
  "value": 5.0,
  "units": "meters",
  "confidence": 0.99,
  "created_at_utc": "2026-09-16T12:20:00Z"
}
```

### 2.10 Depth Maps (`DepthMapSchema`)
Dense or sparse depth map representations carrying mandatory map ID, units, and calibration provenance:
```json
{
  "schema_version": "1.0.0",
  "depth_id": "depth_20260831_000142_cam0",
  "map_id": "map_20260831_v2_metric",
  "units": "meters",
  "calibration_provenance": "factory_eeprom",
  "frame_id": "frame_20260831_000142",
  "camera_id": "cam0",
  "width": 1920,
  "height": 1920,
  "file_path": "depth/cam0_000142.npy",
  "min_depth": 0.3,
  "max_depth": 20.0,
  "scale_factor": 1.0,
  "timestamp_raw_ns": 1726480005000000000,
  "timestamp_relative_s": 5.0
}
```

### 2.11 3D Meshes (`MeshSchema`)
Reconstructed surface meshes, point clouds, or wireframes carrying mandatory map ID, units, and calibration provenance:
```json
{
  "schema_version": "1.0.0",
  "mesh_id": "mesh_20260831_room_dense",
  "map_id": "map_20260831_v2_metric",
  "units": "meters",
  "calibration_provenance": "pycolmap_stereo_dense",
  "mesh_type": "triangular_mesh",
  "file_path": "meshes/dense_room.ply",
  "vertex_count": 125000,
  "face_count": 240000,
  "format": "ply",
  "bounding_box_min": [-4.5, -3.2, 0.0],
  "bounding_box_max": [4.5, 3.2, 2.8],
  "exported_at_utc": "2026-09-16T12:25:00Z"
}
```

---

## 3. Project Storage Layout

```text
floor plan project/
├── configs/
│   └── laptop.yaml                # Hardware profile, stage timeouts, and concurrency limits
├── data/
│   ├── app.sqlite3                # Persistent durable metadata database (WAL mode)
│   └── sessions/
│       ├── 2026-08-31/            # Real capture session (isolated, read-only)
│       └── synth_*/               # Isolated synthetic test fixtures
├── docs/
│   ├── build_contract.md
│   ├── data_contracts.md          # This document
│   └── job_lifecycle.md
├── reports/
│   ├── 01_preflight.md
│   ├── 02_environment.md
│   └── 03_jobs.md
├── runs/
│   └── <run-id>/
│       ├── run.json               # Job metadata record
│       ├── effective_config.json  # Frozen effective config at launch
│       ├── job.log                # Subprocess stdout & stderr
│       ├── progress.json          # Live heartbeat & telemetry
│       └── outputs/
│           ├── .complete          # Atomic completion marker (SHA256 manifest)
│           └── <artifacts>        # Stage output files
└── src/vscan/
    ├── schemas/                   # Pydantic v2 schemas enforcing contracts
    ├── storage/                   # Database & artifact integrity managers
    ├── engine/                    # Supervisor, telemetry, process lifecycle
    └── workers/                   # Stage worker implementations
```

*Note: Raw videos, sqlite databases, caches, and run directories are strictly excluded from version control via `.gitignore`.*
