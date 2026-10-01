# Metric 360° Reconstruction Pipeline (StellaVSLAM + IMU)

Production/Research-grade offline pipeline for metric 3D reconstruction from 360° panoramic video and IMU data.

## Architecture

- **Visual SLAM Frontend**: StellaVSLAM / Equirectangular VO (equirectangular spherical camera model) for robust visual tracking, keyframe poses, and sparse landmark map.
- **Inertial Preintegration**: Bias-aware IMU preintegration (Forster et al. on-manifold formulation).
- **Temporal Alignment**: Camera-IMU clock offset estimation via cross-correlation of angular velocities.
- **Rotation Consistency**: Gating check validating relative visual vs IMU rotations before scale optimization.
- **Inertial Initialization**: Linear least-squares estimation of metric scale, keyframe velocities, and gravity vector.
- **Gravity & Bias Refinement**: Tangent-space gravity optimization (constrained to 9.81 m/s²) and accelerometer bias refinement.
- **Maximum A Posteriori (MAP) Optimization**: Joint visual-inertial optimization with normalized covariance weighting.
- **Metric Transform**: Converting Stella trajectory and landmarks to physical SI units (meters).
- **CAD Floor Plan**: Vectorized structural wall extraction with Manhattan 90° corner snapping.

## Project Structure
```
├── data/                                    # Datasets partitioned by survey date
│   ├── 2026-08-31/                          # August 31, 2026 survey session
│   │   ├── 20260831_163824_location.json
│   │   ├── 20260831_163824_phone_sensors.csv
│   │   ├── LRV_20260831_163825_01_081.mp4
│   │   ├── insta360_export_1788174571431.csv
│   │   └── insta360_export_1788174571431.mp4
│   └── 2026-09-10/                          # September 10, 2026 survey session
│       ├── 20260910_172939_location.json
│       ├── 20260910_172939_phone_sensors.csv
│       ├── LRV_20260910_172941_01_083.mp4
│       ├── VID_20260910_172941_00_083.insv
│       ├── insta360_export_1789041643971.csv
│       └── insta360_export_1789041643971.mp4
├── v1/                                      # Pipeline Version 1 code and artifacts
│   ├── configs/                             # Calibration YAML files
│   ├── src/                                 # Core modules
│   │   ├── alignment/                       # Temporal sync, gyro bias, linear solver
│   │   ├── diagnostics/                     # Observability checks & residual plotting
│   │   ├── inertial/                        # Preintegration & IMU bias models
│   │   ├── ingestion/                       # Video reader & IMU CSV parsers
│   │   ├── metric/                          # Metric converter & trajectory scaling
│   │   ├── optimizer/                       # Normalized MAP Levenberg-Marquardt optimizer
│   │   ├── stella/                          # Spherical VO, Stella I/O, dense stereo
│   │   └── visuals/                         # 2D floor plans, 3D Plotly, video HUD, CAD walls
│   ├── scripts/                             # Orchestration runners
│   │   ├── run_high_resolution_pipeline.py  # Master pipeline runner
│   │   ├── run_metric_pipeline.py           # Standard metric reconstruction CLI
│   │   ├── run_temporal_alignment.py        # Camera-IMU synchronization & calibration
│   │   └── generate_visual_results.py       # Visual asset generator
│   ├── tests/                               # 23 automated unit tests
│   └── output/                              # Exported deliverables (CSV, PLY, SVG, PNG, HTML, MP4)
├── CURRENT_APPROACH_DOCUMENTATION.md        # Exhaustive technical documentation
└── stella_vslam_metric_360_imu_approach.md  # Core specification
```

## Quick Start Commands

```powershell
# Run all automated tests (23 tests)
pytest v1/tests/ -v

# Run temporal sync and camera-IMU calibration
python v1/scripts/run_temporal_alignment.py

# Run full high-resolution metric reconstruction pipeline
python v1/scripts/run_high_resolution_pipeline.py

# Generate 2D floor plans, 3D web viewer, and annotated preview video
python v1/scripts/generate_visual_results.py
```

