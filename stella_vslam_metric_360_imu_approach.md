# Metric 360° Reconstruction Using StellaVSLAM + IMU

## 1. Objective

Build a production-quality offline pipeline that takes:

- 360° panoramic / equirectangular video
- IMU data from the 360° camera
- Optional phone IMU data
- Camera and IMU timestamps / metadata

and produces:

1. A robust visual trajectory using **StellaVSLAM**.
2. A sparse visual map / point cloud from StellaVSLAM.
3. A reliable estimate of **metric scale** using inertial measurements.
4. Gravity-aligned camera poses.
5. Metric 3D landmarks / point cloud coordinates in metres.
6. A trajectory whose translational distances are physically meaningful.
7. Diagnostics that detect invalid calibration, synchronization, stabilization, or observability before accepting a result.

The intended system is **not** to modify StellaVSLAM into a visual-inertial SLAM system.

Instead:

> Use StellaVSLAM as the mature 360° visual SLAM frontend/backend, then run an external visual-inertial alignment and metric optimization stage.

---

# 2. Core Design Decision

## Do not integrate IMU directly into StellaVSLAM

Current StellaVSLAM should be treated as a **vision-only SLAM system**.

Its equirectangular visual support is useful for this problem, but its IMU integration is not mature enough to make it the production path.

Therefore the architecture will be:

```text
                     360° panorama video
                              |
                              v
                       StellaVSLAM
                              |
                  up-to-scale trajectory
                  + visual landmarks/map
                              |
                              |
360 camera IMU --------------+
                              |
                              v
                  IMU preintegration
                              |
                              v
              Visual-Inertial Alignment
                              |
          +-------------------+------------------+
          |                   |                  |
          v                   v                  v
       Scale              Gravity           IMU Biases
          |                   |                  |
          +-------------------+------------------+
                              |
                              v
                 Global VI optimization
                              |
                              v
               Metric camera trajectory
                              |
                              v
               Metric Stella landmarks
```

---

# 3. Why This Architecture

A monocular visual SLAM system reconstructs geometry only up to an unknown scale.

If Stella estimates:

```text
camera position = [1.2, 3.7, 0.4]
```

those numbers do not initially correspond to metres.

The true metric position can initially be represented approximately as:

\[
p_i^{metric} = s \hat{p}_i
\]

where:

- \(\hat{p}_i\) = Stella position
- \(s\) = unknown metric scale

The IMU provides metric acceleration information and therefore makes the scale observable when sufficient translational excitation exists.

However, scale cannot be robustly obtained by simply double-integrating accelerometer measurements.

The correct problem is to jointly estimate:

- metric scale
- gravity direction
- camera/body velocity
- gyroscope bias
- accelerometer bias

and optionally refine:

- visual poses
- camera-IMU time offset
- camera-IMU extrinsics

---

# 4. Inputs Required

## 4.1 360° Video

Preferred:

- raw or minimally processed panorama
- equirectangular projection
- native frame timestamps / presentation timestamps
- no dynamic stabilization, if possible
- known resolution
- known capture frame rate
- access to original camera metadata

Avoid relying on:

```text
timestamp = frame_index / nominal_fps
```

unless it is verified that the video uses a perfectly constant frame interval.

Use actual video timestamps whenever available.

---

## 4.2 360 Camera IMU

Required fields:

```text
timestamp
gyro_x
gyro_y
gyro_z
accel_x
accel_y
accel_z
```

Required metadata:

```text
gyro units
accelerometer units
sample frequency
sensor coordinate convention
timestamp clock source
```

Preferred:

- raw sensor samples
- no vendor motion filtering
- no smoothing
- no transformed "orientation only" stream
- exact sensor timestamps

---

## 4.3 Phone IMU

The phone IMU is **optional**.

Do not use it during the first implementation.

It should only be introduced if the phone and 360 camera are mounted rigidly to the same physical body.

If the phone moves independently relative to the camera, it is not valid as a second IMU for the visual-inertial estimator.

If rigidly mounted:

```text
Rig
 |
 +-- 360 Camera
 |
 +-- 360 Camera IMU
 |
 +-- Phone IMU
```

then the two IMUs require:

- rigid extrinsic calibration
- independent clock-offset estimation
- separate noise models
- separate bias states if fused directly

The camera IMU should remain the primary sensor.

---

# 5. Major Risk: Video Stabilization

This is one of the first things that must be determined.

A 360 camera may record the physical camera orientation:

\[
R_{WB}(t)
\]

while exporting a stabilized panorama corresponding to a virtual camera:

\[
R_{WC_v}(t)
\]

If the camera software applies a time-varying stabilization transform:

\[
R_{stab}(t)
\]

then:

\[
R_{WC_v}(t) = R_{WB}(t)R_{stab}(t)
\]

The IMU measures the physical body motion.

Stella observes the virtual stabilized camera motion.

Therefore:

```text
raw IMU rotation != visual panorama rotation
```

unless the stabilization transformation is compensated.

A fixed camera-IMU extrinsic cannot correct a time-varying stabilization transform.

## Preferred solution

Use:

```text
unstabilized panorama
+
raw camera IMU
```

## Alternative

If only stabilized panorama is available:

extract the camera/vendor stabilization quaternion or orientation metadata and reconstruct the transformation between:

```text
physical camera frame
and
virtual stabilized panorama frame
```

for every video frame.

This must be validated before metric scale estimation.

---

# 6. Coordinate Frames

Define frames explicitly before implementing anything.

Recommended notation:

```text
W       world frame
C       Stella / virtual camera frame
B       physical camera body frame
I       360 camera IMU frame
P       phone IMU frame
```

Important transforms:

```text
T_CI    camera <- IMU
T_BI    body <- IMU
T_CB    visual camera <- physical camera body
T_IP    camera IMU <- phone IMU
```

depending on the exact physical setup.

Every transform convention must specify whether the matrix maps:

```text
source -> destination
```

or:

```text
destination -> source
```

Do not mix pose and extrinsic conventions.

---

# 7. Calibration Requirements

## 7.1 Camera / Equirectangular Model

Stella must use its native equirectangular camera model.

Do not convert the panorama to a conventional pinhole model unless intentionally building a multi-view cubemap pipeline.

The Stella input configuration should therefore correspond to:

```yaml
Camera:
  setup: monocular
  model: equirectangular
```

with image width/height and Stella's appropriate angular thresholds.

---

## 7.2 Camera-IMU Rotation

Estimate:

\[
R_{CI}
\]

accurately.

This is critical.

A few degrees of rotational calibration error can contaminate:

- gravity estimation
- velocity estimation
- accelerometer integration
- metric scale

---

## 7.3 Camera-IMU Translation

Estimate:

\[
t_{CI}
\]

The lever arm is particularly relevant during rotational motion.

For 360 cameras, the effective panorama optical center may not coincide exactly with the IMU location.

Ignoring the lever arm can introduce translational acceleration errors during rotations.

---

## 7.4 Temporal Calibration

Estimate:

\[
\Delta t_{camera,imu}
\]

such that:

\[
t_{imu}=t_{camera}+\Delta t
\]

Do not assume the camera timestamps and IMU timestamps are synchronized simply because they originate from the same device.

Potential problems:

- different internal clocks
- video encoder delay
- metadata timestamp offsets
- frame buffering
- sensor batching
- variable frame rate
- stabilization pipeline delay

---

# 8. IMU Characterization

Before using the IMU, determine:

```text
gyro noise density
accelerometer noise density
gyro random walk
accelerometer random walk
```

Also estimate stationary biases.

Recommended offline tests:

## Static sequence

Keep the camera stationary for several minutes.

Verify:

\[
||a|| \approx 9.81 \ m/s^2
\]

after unit conversion.

Gyroscope mean should be near zero.

From this sequence estimate:

```text
gyro bias
accelerometer bias
noise variance
```

For higher-quality modeling, perform Allan variance / Allan deviation analysis.

---

# 9. Stage 1 — Run StellaVSLAM

Input:

```text
equirectangular frames
+
exact frame timestamps
```

Output required from Stella:

```text
keyframe timestamp
keyframe ID
camera orientation
camera translation
map landmarks
observation associations
```

A recommended exported trajectory format:

```text
timestamp,
tx, ty, tz,
qx, qy, qz, qw
```

Important:

determine whether Stella exports:

\[
T_{WC}
\]

or:

\[
T_{CW}
\]

and normalize all poses to one convention.

---

# 10. Stella Output at This Stage

Stella produces:

```text
visual trajectory
+
sparse landmark map
```

but the translation is only defined up to monocular scale.

Let the Stella keyframe locations be:

\[
\hat{p}_0,\hat{p}_1,...,\hat{p}_N
\]

The metric trajectory is initially:

\[
p_i=s\hat{p}_i
\]

where \(s\) must be estimated.

Do not immediately scale the map.

First solve the full visual-inertial initialization problem.

---

# 11. Stage 2 — Rotation Consistency Test

This is a mandatory gating test.

Do not proceed to scale estimation until visual rotation and IMU rotation agree.

For consecutive visual keyframes:

\[
\Delta R^{visual}_{ij}
=
R_i^{-1}R_j
\]

Preintegrate gyroscope measurements:

\[
\Delta R^{imu}_{ij}
=
\prod_k
Exp((\omega_k-b_g)\Delta t_k)
\]

Transform the IMU rotation into the camera frame using the calibrated extrinsic.

Compare:

```text
roll rate
pitch rate
yaw rate
```

or preferably:

\[
R_{error}
=
(\Delta R^{visual})^{-1}\Delta R^{imu}
\]

and convert the rotation error into an angle.

Plot rotation error against time.

---

# 12. Rotation Validation Failure Means Stop

If the rotations do not agree, investigate:

```text
camera/IMU timestamp offset
wrong quaternion order
degrees vs radians
wrong gyro units
coordinate-axis mismatch
right-handed vs left-handed frame
incorrect T_CI
pose inversion error
stabilized panorama
wrong frame timestamps
vendor orientation transformation
video processing latency
```

Do not attempt metric scale estimation until this validation passes.

A scale optimizer can sometimes return a numerical answer despite invalid synchronization.

That result must not be trusted.

---

# 13. Stage 3 — IMU Preintegration

Do not integrate accelerometer measurements over the entire sequence directly.

Use IMU preintegration between Stella keyframes.

For each consecutive keyframe pair:

```text
K_i
 |
 | IMU samples
 |
 v
K_j
```

compute:

\[
\Delta R_{ij}
\]

\[
\Delta v_{ij}
\]

\[
\Delta p_{ij}
\]

along with:

```text
preintegration covariance
bias Jacobians
time duration Δt
```

Bias-aware preintegration should be used.

A suitable implementation can be based on:

- Forster-style IMU preintegration
- GTSAM
- Ceres-based custom implementation
- VINS-Mono implementation concepts
- ORB-SLAM3 inertial initialization concepts

---

# 14. State to Estimate

Initial visual-inertial state:

\[
X=
\{
s,
g,
v_0,...,v_N,
b_g,
b_a
\}
\]

where:

```text
s       metric scale
g       gravity vector
v_i     velocity at each Stella keyframe
b_g     gyroscope bias
b_a     accelerometer bias
```

After initialization, represent gravity using:

```text
gravity magnitude ≈ 9.81 m/s²
+
2-DOF gravity direction
```

rather than optimizing an unconstrained arbitrary 3-vector indefinitely.

---

# 15. Core Position Constraint

For keyframes \(i\) and \(j\):

\[
p_j
=
p_i
+
v_i\Delta t
+
\frac{1}{2}g\Delta t^2
+
R_i\Delta p_{ij}
\]

The Stella translations satisfy:

\[
p_i=s\hat{p}_i
\]

Therefore:

\[
s(\hat{p}_j-\hat{p}_i)
=
v_i\Delta t
+
\frac{1}{2}g\Delta t^2
+
R_i\Delta p_{ij}
\]

with additional corrections for:

```text
accelerometer bias
gyroscope bias
camera-IMU lever arm
exact frame convention
```

---

# 16. Velocity Constraint

Use:

\[
v_j
=
v_i
+
g\Delta t
+
R_i\Delta v_{ij}
\]

again including bias corrections.

---

# 17. Rotation Constraint

Use:

\[
R_j
\approx
R_i
\Delta R_{ij}(b_g)
\]

This is particularly useful for gyroscope-bias initialization.

---

# 18. Initialization Sequence

Recommended initialization pipeline:

```text
1. Stella visual reconstruction

2. camera-IMU temporal alignment

3. visual vs gyro rotation validation

4. gyroscope bias initialization

5. IMU preintegration

6. linear solve for:
      velocities
      gravity
      scale

7. gravity magnitude constraint

8. gravity-direction refinement

9. accelerometer-bias refinement

10. nonlinear MAP optimization

11. quality checks

12. apply metric transform to Stella map
```

---

# 19. Gyroscope Bias Initialization

Use Stella's visual rotations as an orientation reference.

Estimate \(b_g\) by minimizing disagreement between:

```text
visual relative rotations
and
IMU preintegrated rotations
```

Conceptually:

\[
\min_{b_g}
\sum_{ij}
||
Log(
(\Delta R_{visual,ij})^{-1}
\Delta R_{imu,ij}(b_g)
)
||^2
\]

After solving \(b_g\), re-preintegrate the IMU.

---

# 20. Initial Linear Alignment

Solve approximately for:

```text
velocity of each keyframe
gravity
metric scale
```

using consecutive visual poses and IMU preintegration.

The system can be written as:

\[
Ax=b
\]

where the state contains:

\[
x=
[
v_0,
v_1,
...
v_N,
g,
s
]
\]

Use:

- weighted least squares
- IMU covariance
- numerical conditioning checks
- singular-value inspection

Do not accept the result if the linear system is badly conditioned.

---

# 21. Gravity Refinement

Gravity magnitude is known approximately:

\[
||g||=9.81 \ m/s^2
\]

After obtaining an initial gravity estimate \(g_0\), refine only its direction on the local tangent plane.

Construct two basis vectors:

\[
b_1,b_2 \perp g_0
\]

and optimize:

\[
g
=
g_0
+
b_1\delta\theta_1
+
b_2\delta\theta_2
\]

then normalize gravity magnitude appropriately.

This improves stability compared with freely optimizing all three gravity components.

---

# 22. Stage 4 — Nonlinear MAP Optimization

After initialization, perform nonlinear optimization.

Recommended state:

```text
global scale
gravity orientation
velocity per keyframe
gyro bias
accelerometer bias
optional camera poses
optional camera/IMU time offset
```

Optional later-stage state:

```text
camera-IMU extrinsic refinement
```

Do not optimize every quantity simultaneously during the first iteration of the implementation.

Lock calibrated extrinsics initially.

---

# 23. MAP Objective

Use something conceptually equivalent to:

\[
\min_X
\sum_{ij}
r_{imu}^{T}
\Sigma_{ij}^{-1}
r_{imu}
+
\sum_i
r_{visual,i}^{T}
W_v
r_{visual,i}
+
r_{prior}
\]

where:

```text
r_imu      inertial residual
Σ_ij       preintegration covariance
r_visual   Stella pose prior residual
W_v        visual pose confidence
r_prior    bias / gravity / calibration priors
```

Initially Stella poses can be treated as fixed.

Later, for maximum quality, allow small adjustments to Stella poses with strong visual priors.

---

# 24. Why Not Just Estimate One Global Scale?

A simple alternative is:

```text
run Stella
run independent VIO
align trajectories
estimate Sim(3)
scale Stella point cloud
```

This is useful as a diagnostic.

It is not the preferred final system.

A monocular SLAM trajectory can contain:

```text
local scale drift
trajectory deformation
loop-closure corrections
```

A single global number \(s\) cannot repair local inconsistencies.

A better final formulation uses IMU factors throughout the Stella trajectory.

---

# 25. Stage 5 — Convert Stella Map to Metric Coordinates

After obtaining:

```text
scale s
gravity alignment R_g
```

transform Stella camera positions:

\[
p_i^{metric}
=
R_g(s p_i^{stella})
\]

Transform Stella landmarks:

\[
X_j^{metric}
=
R_g(s X_j^{stella})
\]

Orientations become approximately:

\[
R_i^{metric}
=
R_g R_i^{stella}
\]

subject to the exact pose convention used by Stella.

---

# 26. What Becomes Metric

After successful alignment:

```text
camera translations
trajectory length
landmark distances
camera-to-object distance
object-to-object distance
camera height changes
3D point cloud geometry
```

can be expressed in metres.

---

# 27. What IMU Does Not Automatically Determine

IMU does not by itself provide a globally unique:

```text
world origin
global yaw
geographic position
```

Even after metric scale recovery, gauge freedoms remain.

To obtain absolute world coordinates or heading, add one or more of:

```text
GNSS
known survey point
AprilTag / fiducial
known camera starting position
magnetometer
known world direction
external tracking system
```

---

# 28. Required Motion / Observability

Metric initialization requires sufficient excitation.

Bad calibration sequence:

```text
camera stationary
pure rotation
perfectly straight constant-speed motion
very slow smooth motion with almost no acceleration
```

Preferred calibration motion:

```text
translation
accelerate
decelerate
left/right motion
yaw
pitch
roll
small vertical motion
```

The motion should provide sufficiently rich 6-DoF excitation without excessive blur.

---

# 29. Observability Checks

Before accepting a metric solution, verify:

## Scale stability

Estimate scale across multiple temporal windows.

Example:

```text
0-10 s
5-15 s
10-20 s
...
```

Scale estimates should converge toward a consistent value.

---

## Gravity consistency

Check:

\[
||g|| \approx 9.81
\]

and verify that its direction is stable across optimization windows.

---

## Bias consistency

Gyroscope and accelerometer biases should not diverge.

---

## Velocity sanity

Estimated camera velocities should remain physically plausible.

---

## Residual consistency

Inspect:

```text
gyro residual
accelerometer residual
position residual
velocity residual
rotation residual
```

Reject sequences where residuals remain systematically biased.

---

# 30. Metric Validation

The pipeline must be validated using known physical measurements.

Examples:

```text
known hallway length
known floor tiles
known object spacing
measured camera travel distance
known wall-to-wall distance
surveyed markers
```

Do not validate only against the same IMU data used by the optimizer.

Use independent geometric ground truth.

Recommended metrics:

```text
absolute scale error %
relative trajectory error
absolute trajectory error
known-distance error
landmark-distance error
```

---

# 31. Recommended Accuracy Test

Create a controlled trajectory containing:

```text
start marker
known 5 m translation
90° turn
known 5 m translation
return section
```

Measure actual distances independently.

Run:

```text
Stella only
Stella + global scale
Stella + full IMU alignment
```

Compare results.

---

# 32. Phone IMU Integration — Later Phase

Only add the phone IMU after the camera IMU pipeline works.

Prerequisites:

```text
rigid phone-camera mount
T_phone_camera calibration
phone-camera clock offset
phone IMU noise model
axis convention verified
```

Two possible use cases:

### Option A — Redundant IMU

Add a second inertial factor stream.

Useful for:

```text
noise reduction
failure detection
sensor cross-validation
```

### Option B — Validation Only

Use phone IMU independently to verify:

```text
angular motion
gravity direction
major acceleration events
```

This is safer initially.

---

# 33. Handling Equirectangular Stitching

360° stitching introduces potential violations of the ideal single rigid-camera model.

Potential issues:

```text
stitch seam
parallax near camera
moving objects
exposure mismatch
warping
optical-flow discontinuities
temporal stitching
```

Recommended actions:

- use Stella's equirectangular model
- mask known stitching boundary regions if necessary
- mask camera rig/self-view
- reject dynamic landmarks aggressively
- prefer sufficiently distant static scene structure
- test front/back hemispheres independently during debugging
- inspect feature tracks near stitching seams

---

# 34. Dense Point Cloud Consideration

StellaVSLAM primarily provides a sparse landmark map.

Metric alignment makes those landmarks metric.

If a dense point cloud is required, use the optimized metric poses as input to a separate dense reconstruction pipeline.

Possible downstream approaches:

```text
multi-view stereo
depth estimation + fusion
Gaussian splatting
NeRF-style reconstruction
360 stereo/depth network
TSDF fusion if depth becomes available
```

The dense system should consume the **metric optimized trajectory**, not raw Stella translations.

---

# 35. Software Architecture

Recommended project structure:

```text
metric_360_slam/
|
+-- data/
|   +-- video/
|   +-- imu/
|   +-- calibration/
|
+-- extraction/
|   +-- extract_video_frames.py
|   +-- extract_video_timestamps.py
|   +-- extract_camera_imu.py
|   +-- extract_metadata.py
|
+-- stella/
|   +-- run_stella.sh
|   +-- equirectangular_config.yaml
|   +-- export_keyframes.cpp
|   +-- export_landmarks.cpp
|
+-- calibration/
|   +-- imu_static_analysis.py
|   +-- temporal_alignment.cpp
|   +-- camera_imu_extrinsics/
|
+-- inertial/
|   +-- imu_types.hpp
|   +-- imu_loader.cpp
|   +-- imu_preintegration.cpp
|   +-- bias_model.cpp
|
+-- alignment/
|   +-- gyro_bias_initializer.cpp
|   +-- visual_imu_alignment.cpp
|   +-- gravity_refinement.cpp
|   +-- observability_check.cpp
|
+-- optimizer/
|   +-- state.hpp
|   +-- imu_factor.cpp
|   +-- visual_pose_factor.cpp
|   +-- metric_optimizer.cpp
|
+-- reconstruction/
|   +-- metricize_trajectory.cpp
|   +-- metricize_landmarks.cpp
|   +-- export_ply.cpp
|
+-- diagnostics/
|   +-- compare_rotations.py
|   +-- plot_imu_residuals.py
|   +-- plot_scale_convergence.py
|   +-- plot_biases.py
|   +-- trajectory_evaluation.py
|
+-- configs/
|   +-- imu.yaml
|   +-- extrinsics.yaml
|   +-- optimization.yaml
|
+-- output/
    +-- stella_trajectory.txt
    +-- metric_trajectory.txt
    +-- stella_map.ply
    +-- metric_map.ply
    +-- diagnostics/
```

---

# 36. Suggested Intermediate Data Formats

## Camera trajectory

```csv
timestamp,keyframe_id,px,py,pz,qx,qy,qz,qw
```

## IMU

```csv
timestamp,gx,gy,gz,ax,ay,az
```

## Calibration

```yaml
camera_imu:
  rotation: [...]
  translation: [...]

time_offset:
  camera_to_imu: 0.0

imu:
  gyro_noise_density: ...
  accel_noise_density: ...
  gyro_random_walk: ...
  accel_random_walk: ...
  gravity: 9.81
```

---

# 37. Diagnostic Plots That Must Exist

The pipeline should automatically generate:

```text
visual vs IMU angular velocity
rotation residual vs time
acceleration magnitude vs time
estimated gyro bias vs time
estimated accel bias vs time
gravity direction
scale vs optimization iteration
scale vs temporal window
trajectory before metric alignment
trajectory after metric alignment
IMU residual histogram
translation residual histogram
```

A metric reconstruction should not be accepted without passing these diagnostics.

---

# 38. Failure Conditions

Reject the result if any of the following occurs:

```text
visual/IMU rotation disagreement
scale changes significantly between windows
gravity magnitude far from 9.81 m/s²
bias estimates diverge
optimizer becomes poorly conditioned
translation excitation is insufficient
camera timestamps are unreliable
camera and IMU clocks cannot be aligned
stabilized video transform is unknown
visual tracking contains severe stitching artifacts
Stella loses tracking repeatedly
```

---

# 39. Specific Stella-Related Concern

There has been practical evidence from equirectangular Stella workflows where users obtained incorrect scale after combining Stella output with IMU-derived information.

This should not automatically be interpreted as a Stella scale bug.

The first suspects should be:

```text
stabilization
camera/IMU synchronization
pose convention
camera/IMU extrinsic
IMU units
effective panorama optical center
equirectangular stitching
```

The design should therefore make these quantities directly observable through diagnostics.

---

# 40. Recommended Development Order

## Milestone 1 — Visual pipeline

Goal:

```text
360 video
-> Stella
-> stable trajectory
-> sparse map
```

Validate loop closure and visual reconstruction quality.

---

## Milestone 2 — IMU parsing

Goal:

```text
raw IMU
-> standardized SI units
-> monotonic timestamps
-> noise characterization
```

---

## Milestone 3 — Rotation alignment

Goal:

```text
Stella rotation
≈
camera-frame integrated gyro rotation
```

This milestone must pass before any scale work.

---

## Milestone 4 — Gyro bias

Estimate and validate gyroscope bias.

---

## Milestone 5 — IMU preintegration

Produce:

```text
ΔR
Δv
Δp
covariance
Jacobians
```

between Stella keyframes.

---

## Milestone 6 — Linear metric initializer

Estimate:

```text
scale
gravity
velocities
```

---

## Milestone 7 — Bias + gravity refinement

Add:

```text
accelerometer bias
gravity tangent-space refinement
```

---

## Milestone 8 — Nonlinear optimization

Implement full inertial MAP refinement.

---

## Milestone 9 — Metric map export

Transform:

```text
Stella trajectory
Stella landmarks
```

into metres.

---

## Milestone 10 — Ground-truth validation

Measure known scene dimensions and quantify metric error.

---

# 41. First Production Version

The first production-worthy implementation should intentionally remain constrained.

Use:

```text
360° equirectangular video
360 camera IMU only
fixed calibrated T_CI
fixed camera/IMU time offset
Stella poses fixed during initialization
IMU preintegration
gyro-bias initialization
scale + velocity + gravity solve
gravity refinement
accelerometer bias refinement
nonlinear inertial optimization
metric map export
```

Do not initially add:

```text
phone IMU
online extrinsic estimation
online time-offset estimation
Stella source modification
full joint visual BA
dense reconstruction
```

Those can be introduced after the metric initialization pipeline is demonstrably correct.

---

# 42. Second Production Version

Once the first implementation is stable:

```text
allow slight Stella pose corrections
add visual pose factors
optimize camera/IMU time offset
refine camera/IMU extrinsics
add second IMU
perform global VI optimization
add dense metric reconstruction
```

---

# 43. Acceptance Criteria

The system is considered successful only when:

### Visual

- Stella maintains tracking.
- Loop closure behaves correctly.
- Panorama stitching artifacts do not dominate tracking.

### Rotation

- IMU and visual relative rotations closely agree.

### Inertial

- gravity magnitude is physically plausible.
- biases remain stable.
- preintegration residuals are bounded.

### Metric

- estimated scale is stable over time.
- independent known distances are reconstructed accurately.
- trajectory distance is physically plausible.
- landmark distances are repeatable.

### Engineering

- every input/output coordinate convention is documented.
- every timestamp source is documented.
- calibration files are version controlled.
- diagnostics are generated automatically.
- failed observability/calibration tests cause explicit pipeline rejection.

---

# 44. Final Target Pipeline

```text
                   RAW 360 CAMERA DATA
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
   Equirectangular                     Camera IMU
       Frames                                |
          |                                  |
          v                                  v
     StellaVSLAM                      Unit Conversion
          |                                  |
          |                           Timestamp Cleanup
          |                                  |
          v                                  v
   Visual Keyframes <------ Temporal/Extrinsic Calibration
          |                                  |
          +--------------+-------------------+
                         |
                         v
                Rotation Consistency
                         |
                    PASS / FAIL
                         |
                         v
                Gyro Bias Estimation
                         |
                         v
                 IMU Preintegration
                         |
                         v
            Visual-Inertial Initialization
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
           Scale       Gravity    Velocities
             |           |           |
             +-----------+-----------+
                         |
                         v
                  Bias Refinement
                         |
                         v
                Nonlinear VI MAP
                         |
                         v
               Metric Transformation
                         |
              +----------+----------+
              |                     |
              v                     v
       Metric Trajectory       Metric Landmarks
              |                     |
              +----------+----------+
                         |
                         v
                  Metric 3D Map
```

---

# 45. Key Engineering Principle

The system should never assume that a numerical scale returned by an optimizer is correct.

The metric solution is accepted only after confirming:

```text
visual rotation agrees with IMU
timestamps are aligned
stabilization is handled
camera-IMU extrinsics are valid
gravity is physically correct
biases are stable
motion provides sufficient observability
scale is repeatable
known real-world distances validate the reconstruction
```

This validation-first design is significantly more important than the choice of nonlinear optimizer.

---

# 46. Recommended Technical References

The implementation should be informed by:

1. **StellaVSLAM**
   - equirectangular camera model
   - visual pose/map generation
   - loop closure / visual SLAM

2. **VINS-Mono**
   - visual-first initialization
   - gyroscope-bias initialization
   - scale / gravity / velocity alignment
   - gravity refinement

3. **ORB-SLAM3**
   - inertial initialization
   - inertial-only MAP estimation
   - scale and gravity observability
   - visual-inertial bundle adjustment concepts

4. **Forster et al. IMU Preintegration**
   - bias-aware IMU preintegration
   - covariance propagation
   - manifold rotation integration

5. **360-VIO**
   - visual-inertial estimation specifically using 360° equirectangular imagery
   - spherical projection considerations

6. **Kalibr concepts**
   - camera-IMU spatial calibration
   - temporal calibration
   - IMU noise characterization

---

# 47. Final Engineering Recommendation

The primary implementation should be:

```text
StellaVSLAM
    +
native 360° camera IMU
    +
external VINS/ORB-SLAM3-style
visual-inertial metric initializer
    +
global inertial refinement
```

Do not modify StellaVSLAM's backend until this external approach has been fully tested.

Do not use the phone IMU until the primary camera-IMU solution works.

Do not use a simple accelerometer double-integration scale estimate.

Do not trust stabilized panorama + raw IMU unless the stabilization transform is known.

The highest-priority technical milestone is:

> Achieve tight agreement between Stella-derived camera rotation and the camera IMU's integrated rotation.

Once that passes, solving metric scale becomes a well-defined visual-inertial estimation problem rather than guesswork.
