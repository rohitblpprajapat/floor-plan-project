# Stage 05 Frames Extraction & Reprojection Report: Session `20260910_172939`

- **Execution Timestamp**: `2026-09-17T05:30:40Z`
- **Session ID**: `20260910_172939`
- **Run ID**: `run_frames_20260917_053006_a4cf20`
- **Status**: `STAGE_05_COMPLETE`
- **Actual Data Readiness**: `PROVISIONAL_PANORAMIC_BASELINE`
- **Metric Scale Status**: `NOT_ESTIMATED` (Authoritative: Immutable baseline)

---

## 1. Status Partitioning

| Domain | Status | Operational Note |
| :--- | :--- | :--- |
| **Software Status** | `STAGE_05_COMPLETE` | Media decoding, perspective cubemap reprojection, PyCOLMAP rig configuration, masking, and quality scoring verified |
| **Actual Data Status** | `PROVISIONAL_PANORAMIC_BASELINE` | Verified supported panoramic export used as visual baseline; raw dual-fisheye uncalibrated |
| **Metric Scale Status** | `NOT_ESTIMATED` | Strictly unscaled; SfM or sensor scale estimation forbidden in this stage |

---

## 2. Projection Adapter & Geometric Specification

### 2.1 Coordinate Conventions & Optical Reference Frame
- **Camera Frame**: Standard OpenCV convention:
  - $+X$: Camera right
  - $+Y$: Camera down
  - $+Z$: Optical axis forward (principal ray)
- **Equirectangular Panorama Frame**:
  - Azimuth $\theta \in [-\pi, \pi]$: $\theta=0$ forward ($+Z$), $+\pi/2$ right ($+X$), $-\pi/2$ left ($-X$), $\pm\pi$ backward ($-Z$).
  - Elevation $\phi \in [-\pi/2, \pi/2]$: $\phi=+\pi/2$ zenith ($-Y$), $\phi=0$ equator, $\phi=-\pi/2$ nadir ($+Y$).
  - Pixel center convention: Half-integer $(u + 0.5, v + 0.5)$.
- **Synthetic Zero Baseline**:
  - All six perspective views of a panorama share the identical nodal optical center $[0, 0, 0]^T$.
  - Relative translation vectors are strictly $\mathbf{t} = [0.0, 0.0, 0.0]^T$.

### 2.2 SO(3) Rotation Formulations
All rotations are proper orthonormal elements of $\mathrm{SO}(3)$ satisfying $\det(\mathbf{R}) = +1.0$ and $\mathbf{R}^T \mathbf{R} = \mathbf{I}$:

| Face | Direction Vector | Rotation Matrix $\mathbf{R}_{\text{pano\_cam}}$ | COLMAP Quaternion $[w, x, y, z]$ |
| :--- | :---: | :---: | :---: |
| `face_front` | $[0, 0, 1]$ | $\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ | $[1.0, 0.0, 0.0, 0.0]$ |
| `face_right` | $[1, 0, 0]$ | $\begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ -1 & 0 & 0 \end{bmatrix}$ | $[\frac{\sqrt{2}}{2}, 0, -\frac{\sqrt{2}}{2}, 0]$ |
| `face_back` | $[0, 0, -1]$ | $\begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$ | $[0.0, 0.0, 1.0, 0.0]$ |
| `face_left` | $[-1, 0, 0]$ | $\begin{bmatrix} 0 & 0 & -1 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \end{bmatrix}$ | $[\frac{\sqrt{2}}{2}, 0, \frac{\sqrt{2}}{2}, 0]$ |
| `face_up` | $[0, -1, 0]$ | $\begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{bmatrix}$ | $[\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}, 0, 0]$ |
| `face_down` | $[0, 1, 0]$ | $\begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{bmatrix}$ | $[\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}, 0, 0]$ |

### 2.3 Pinhole Intrinsics (960x960 @ 90° FOV)
- **Width $\times$ Height**: $960 \times 960$ pixels
- **Field of View**: $90.0^\circ$ symmetric horizontal and vertical
- **Focal Lengths**:
  $$f_x = f_y = \frac{W}{2 \tan(\mathrm{FOV}/2)} = \frac{960}{2 \tan(45^\circ)} = 480.0\text{ px}$$
- **Principal Point**:
  $$c_x = \frac{W}{2} = 480.0\text{ px}, \quad c_y = \frac{H}{2} = 480.0\text{ px}$$
- **Skew**: $s = 0.0$, Distortion: none (ideal rectilinear pinhole).

### 2.4 Boundary Seam Wrapping & Polar Bounding
- **Azimuth Wrap**: Azimuth coordinate in equirectangular image is wrapped modulo width:
  $$u_{\text{eq}} = \mathrm{mod}\left( \frac{\theta + \pi}{2\pi} W_{\text{eq}} - 0.5, \; W_{\text{eq}} \right)$$
  ensuring seamless $C^0$ continuity across the $\pm 180^\circ$ seam boundary.
- **Polar Clamping**: Elevation coordinates are strictly clamped to $[0, H_{\text{eq}} - 1]$ to eliminate coordinate divergence at the zenith and nadir singularities.

---

## 3. PyCOLMAP 4.2.0 Pinned Rig Configuration

The generated `rig_config.json` complies strictly with PyCOLMAP 4.2.0 schema requirements:
- Structure: JSON list containing camera rig configuration dictionaries.
- Reference Sensor: `face_front_` designated as `ref_sensor: true`.
- Relatives Sensors: Non-reference cameras specify `cam_from_rig_rotation` ($[w, x, y, z]$ order) and `cam_from_rig_translation` ($[0.0, 0.0, 0.0]$).
- API Verification: Verified using `pycolmap.read_rig_config(rig_path)`.
- Ray Reproduction: Verified identical 3D direction vectors between analytic rotation and PyCOLMAP-parsed rig ($< 10^{-7}$ error, machine precision $\sim 2.22 \times 10^{-16}$).

---

## 4. Masking Subsystem & Provenance Tracking

Explicit masks are maintained with complete provenance and propagated from equirectangular space to perspective cubemap faces:

| Mask ID | Type | Source / Provenance | Affected Perspectives | Description |
| :--- | :--- | :--- | :--- | :--- |
| `mask_hardware_nadir` | `support_hardware` | `nadir_elevation_deg <= -78.0` | `face_down` | Exclusion of camera selfie stick, tripod legs, and mount bracket |
| `mask_seam_operator` | `operator` | `inspected_content_seam_boundary` | `face_back` | Exclusion of operator body/hands at azimuth $\pm 180^\circ$ seam |

### Sutherland-Hodgman Seam Splitting
Polygons spanning across the azimuth seam ($x = 0.0 / 1.0$) are clipped using Sutherland-Hodgman polygon clipping into left $[0.0, 1.0]$ and wrapped $[-1.0, 0.0] \to [0.0, 1.0]$ segments. Collinear vertices and duplicate boundary points are purged, guaranteeing that reprojected masks form a single contiguous connected component on `face_back` (verified via `cv2.connectedComponentsWithStats`).

---

## 5. Actual Data Pilot Extraction: Session `20260910_172939`

### 5.1 Source Video Properties
- **Source Video**: `data/2026-09-10/insta360_export_1789041643971.mp4`
- **Source Asset ID**: `asset_20260910_172939_insta360_export_1789041643971_mp4`
- **Source Resolution**: $3840 \times 1920$ equirectangular
- **Source Frame Count Inspected**: 121 frames (4.03 seconds)
- **Decoder**: OpenCV VideoCapture, strictly preserving original PTS and presentation timeline.

### 5.2 Pilot Selection & Quality Metrics
- **Selection Policy**: Contiguous overlapping window maximizing composite quality while maintaining temporal continuity.
- **Pilot Panoramas Retained**: 40 panoramas (frames 66 to 105, PTS 2.20s to 3.50s).
- **Total Perspective Views Generated**: $40 \times 6 = 240$ images ($960 \times 960$ JPEG).
- **Quality Scoring Results**:
  - Average Sharpness (Laplacian variance + Tenengrad): 1125.0
  - Average Exposure (Luminance mean): 105.6 / 255
  - Dark Clipping Fraction: $< 0.82\%$
  - Bright Clipping Fraction: $< 0.15\%$
  - Motion Difference: Mean $0.108$ inter-frame photometric shift (smooth motion without blur).

### 5.3 Generated Artifacts
- Manifest: `runs/run_frames_20260917_053006_a4cf20/outputs/pilot_manifest.json` (also mirrored to `data/sessions/20260910_172939/pilot_manifest.json`)
- Rig Config: `runs/run_frames_20260917_053006_a4cf20/outputs/rig_config.json` (also mirrored to `data/sessions/20260910_172939/rig_config.json`)
- Contact Sheets:
  - Pilot panoramas: `runs/run_frames_20260917_053006_a4cf20/outputs/contact_sheets/contact_sheet_pilot_panoramas.jpg`
  - Perspective faces: `runs/run_frames_20260917_053006_a4cf20/outputs/contact_sheets/contact_sheet_perspective_faces.jpg`
- Completion Marker: `runs/run_frames_20260917_053006_a4cf20/outputs/.complete` (verified with SHA-256 hash).

---

## 6. Synthetic Room Fixture & Analytic Ground Truth

### 6.1 Synthetic 3D Textured Room Geometry
A deterministic synthetic room fixture was implemented in `vscan.frames.synthetic`:
- Bounds: $X \in [-2.0, 2.0]$, $Y \in [-1.5, 1.5]$, $Z \in [-3.0, 3.0]$ meters.
- Interior Obstacle: A $0.8\text{m} \times 1.5\text{m} \times 0.8\text{m}$ rectangular pillar positioned at $X \in [0.4, 1.2]$, $Y \in [0.0, 1.5]$, $Z \in [1.0, 1.8]$.
- Texture: Procedural high-frequency 2D grid textures on all 6 walls and obstacle surfaces.
- Ray Tracing: Exact closed-form ray-plane intersection yielding analytic subpixel 3D coordinates and depths.
- Projection Agreement: Equirectangular reprojected perspective views match direct analytic pinhole renders within mean color difference $< 15.0$ intensity levels (due solely to bilinear resampling on the discrete equirectangular grid).

### 6.2 Negative Fixture: Rotation-Only Zero Baseline
A synthetic rotation-only fixture sequence was generated:
- Camera Center: Strictly fixed at $[0.0, 0.0, 0.0]$.
- Translation Baseline Norm: Strictly $0.000000$ meters across all frames.
- Rotations: Smooth SO(3) yaw/pitch rotations ($\det(R) = 1.0$).
- Purpose: Negative test fixture for SfM and visual odometry pipelines to verify that zero-baseline sequences are detected and do not corrupt translation estimation.

---

## 7. Job Lifecycle, Caching & CLI Verification

### 7.1 Operational Lifecycle Commands
```bash
# Start synchronous frames extraction on session
vscan jobs start frames --session 20260910_172939 --config configs/session.yaml --wait

# Start asynchronous frames extraction
vscan jobs start frames --session 20260910_172939 --config configs/session.yaml

# Check job status
vscan jobs status <run_id>

# Cancel running frames job
vscan jobs cancel <run_id>

# Resume cancelled frames job
vscan jobs resume <run_id> --wait
```

### 7.2 Cache Key Computation & Invalidation
- Cache key is calculated from: `stage_name`, `config_hash`, and `input_hashes` (source video SHA-256 and mask definitions).
- Verified that modifying resolution, field of view, or mask parameters produces distinct cache keys and invalidates cache.
- Cooperative cancellation verified with immediate graceful termination and clean resumption.

---

## 8. Summary of Completed Tests

All 10 unit and integration tests in `tests/test_frames.py` and all 39 prior tests passed cleanly (49 passed in total):
- `test_01_equirectangular_pixel_ray_roundtrips`: PASSED
- `test_02_crop_and_resize_intrinsic_transformations`: PASSED
- `test_03_proper_so3_rotations_and_zero_baseline`: PASSED
- `test_04_pycolmap_rig_serialization_and_ray_reproduction`: PASSED
- `test_05_seam_crossing_polygon_mask_alignment`: PASSED
- `test_06_synthetic_room_projection_vs_analytic_ground_truth`: PASSED
- `test_07_negative_fixture_rotation_only_zero_baseline`: PASSED
- `test_08_frames_job_lifecycle_cancel_and_resume`: PASSED
- `test_09_cache_invalidation_on_config_change`: PASSED
- `test_10_real_session_actual_data_extraction_and_contact_sheet`: PASSED
