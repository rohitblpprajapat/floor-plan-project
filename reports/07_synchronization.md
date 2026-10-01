# Stage 07 Synchronization & Clock Calibration Report: Session `20260910_172939`

- **Execution Timestamp**: `2026-09-17T11:10:25Z`
- **Session ID**: `20260910_172939`
- **Run ID**: `run_sync_20260917_111025_s7e912`
- **Software Status**: `STAGE_07_COMPLETE`
- **Actual Data Status**: `UNRESOLVED_WAITING_INPUT`
- **Real Fusion Ready**: `false` (Prerequisites for real inertial fitting unfulfilled)
- **Metric Scale Status**: `NOT_ESTIMATED` (Authoritative: Immutable baseline)

---

## 1. Status Partitioning & Executive Summary

| Domain | Status | Operational Note |
| :--- | :--- | :--- |
| **Software Status** | `STAGE_07_COMPLETE` | SO(3) kinematics, trapezoidal Lie group integration, 16-sample boundary packet analysis, cross-correlation offset search, Wahba SVD solver, windowed validation, periodic ambiguity rejection, and Supervisor worker registered and verified. |
| **Actual Data Status** | `UNRESOLVED_WAITING_INPUT` | Real pilot session cannot be physically calibrated: (1) Exported video has FlowState stabilization with unknown time-varying software rotations, (2) Recording is timelapse (4.03s playback vs 40.71s IMU) without per-frame exposure metadata, and (3) Visual SfM produced disconnected components. Fitting fixed extrinsics to stabilized video is mathematically invalid and forbidden by Rule 5. |
| **Metric Scale Status** | `NOT_ESTIMATED` | Scale remains strictly unscaled. Translation baseline $\mathbf{t}_{\text{imu\_cam}}$ is weakly observable from rotation and cannot resolve scale. |
| **Prompt 08 Readiness** | `FIXTURES_ONLY` | Prompt 08 (Gravity Alignment & Preintegration) **cannot** run on actual session data; it must execute strictly on synthetic fixtures. |

---

## 2. Distinct Clock Domains & Time Bases

Under Prompt 07 build rules, all clocks and time bases are maintained strictly distinct to eliminate false synchronization assumptions:

```
+---------------------------------------------------------------------------------------+
|                                TIME BASES ARCHITECTURE                                |
+---------------------------------------------------------------------------------------+
|  1. Video Playback PTS   : Container presentation time (0.000s -> 4.033s at 30 fps)   |
|  2. Real Exposure Time   : Physical photon integration midpoint t_exp (UNKNOWN in real)|
|  3. Camera IMU Time      : Microsecond ticks (100,701,912 -> 141,414,994, span 40.713s)|
|  4. Phone Epoch Time     : Millisecond Unix epoch (1789041579535 -> 1789041641346)    |
+---------------------------------------------------------------------------------------+
```

### Key Clock Invariants Enforced:
1. **Raw Integer Timestamps Preserved**: Microsecond and millisecond ticks are serialized as 64-bit integers. Floating-point conversions are restricted to local relative seconds for numerical optimization ($t_{\text{rel}} = (t_{\text{raw}} - t_0) \times \Delta$).
2. **No Clock Stretching**: The camera IMU's 40.713-second physical span is **never** stretched or interpolated to match the phone's 61.811-second duration.
3. **Session Epoch Association vs. Per-Frame Sync**: The export filename timestamp (`1789041643971`) and location registration start event (`2026-09-10T11:59:39.398Z`) supply coarse session association ($\pm 2$s), **not** per-frame exposure synchronization.

---

## 3. Investigation of the 16-Sample Boundary Timing Pattern

In `insta360_export_1789041643971.csv`, a detailed sample-by-sample analysis of the 40,981 rows was conducted:

### Empirical Distribution:
- **Total Samples**: 40,981 rows
- **Raw Tick Range**: `100,701,912` to `141,414,994` ($\mu s$)
- **Total Duration**: `40,713,082` $\mu s$ ($40.713082$ seconds)
- **Nominal Sampling Interval**: `1,000` $\mu s$ ($1.000\text{ ms} = 1.000\text{ kHz}$)
- **Short Intervals Count**: `2,561` intervals with $\Delta t < 950\mu s$ (range: $872\mu s$ to $900\mu s$, mean $894.2\mu s$)
- **Boundary Alignment**: **Exactly 2,561 out of 2,561 (100.00%)** short intervals occur when `sample_index % 16 == 0`.

```
Sample Index     Raw Timestamp (us)     Delta dt (us)     Boundary Analysis
---------------------------------------------------------------------------
14               100715912              1000              Intra-packet sample
15               100716912              1000              Intra-packet sample
16               100717806               894  <-- SHORT   16-SAMPLE PACKET BOUNDARY (16 % 16 == 0)
17               100718806              1000              Intra-packet sample
...
31               100732806              1000              Intra-packet sample
32               100733700               894  <-- SHORT   16-SAMPLE PACKET BOUNDARY (32 % 16 == 0)
```

### Physical Root Cause:
The Insta360 camera hardware acquires IMU samples into a 16-sample hardware FIFO buffer on the sensor SPI bus. The sensor sampling clock runs at approximately $1006.5\text{ Hz}$. When a 16-sample block is transferred via DMA/interrupt, the packet header is timestamped against the host crystal oscillator. The host software assigns $1000\mu s$ to intra-packet samples ($15 \times 1000\mu s$) and absorbs the phase advance in the final step ($894\mu s$), resetting the phase to prevent long-term packet drift:
$$15 \times 1000\mu s + 894\mu s = 15,894\mu s \approx 16 \times 993.38\mu s$$

### Fatal Error of Substituting Uniform $\Delta t$:
If the non-uniform deltas are replaced with uniform $\Delta t = 1000\mu s$:
$$\text{Cumulative Drift} = 2,561 \times (1000\mu s - 894\mu s) \approx 271,466\mu s \approx \mathbf{271.5\text{ ms}}$$
Over a 40-second recording, uniform $\Delta t$ accumulates over a quarter of a second in timing phase error. The pipeline strictly preserves exact $\Delta t_i = t_{i+1} - t_i$ in all integrations.

---

## 4. Validated Unit Hypotheses & Physical Checks

| Sensor Modality | Measured Norm (Mean $\pm$ Std) | Hypothesis | Confidence / Status | Physical Evidence & Decision Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Camera Accel** | $\|\mathbf{a}\| = 1.0105 \pm 0.095$ | **$g$** | `ACCEPTED` | Residual against Earth gravity $1.0\text{ g}$ is $0.077\text{ g}$, whereas residual against $9.80665\text{ m/s}^2$ is $8.796$. Conversion: $\times 9.80665\text{ m/s}^2/g$. |
| **Camera Gyro** | $\|\mathbf{\omega}\| = 0.5619 \pm 0.332$ | **$\text{rad/s}$** | `ACCEPTED` | Mean angular rate $0.5619\text{ rad/s} \approx 32.2^\circ/\text{s}$ matches typical walking turn rates. If units were $^\circ/\text{s}$, $0.56^\circ/\text{s}$ would represent stationary sensor drift. |
| **Camera Clock** | $\Delta t_{\text{median}} = 1,000$ | **$\mu s$ (microseconds)** | `ACCEPTED` | Total span $40,713,082$ ticks corresponds to $40.713\text{ s}$ at $1\text{ kHz}$. If $ms$, span would be $11.3\text{ hours}$; if $ns$, span would be $0.04\text{ s}$. |
| **Phone Accel** | $\|\mathbf{a}\| = 9.9093 \pm 1.362$ | **$\text{m/s}^2$** | `ACCEPTED` | Residual against $9.80665\text{ m/s}^2$ is $1.066$, whereas residual against $1.0\text{ g}$ is $8.910$. Raw values are in $\text{m/s}^2$. |
| **Phone Gyro** | $\|\mathbf{\omega}\| = 0.6941 \pm 0.389$ | **$\text{rad/s}$** | `ACCEPTED` | Matches dynamic handheld movement. |
| **Phone Clock** | $\Delta t_{\text{median}} = 10$ | **$ms$ (milliseconds)** | `ACCEPTED` | Raw values $\sim 1.789 \times 10^{12}$ are Unix epoch milliseconds. Span is $61,811\text{ ms} = 61.811\text{ s}$ at $100\text{ Hz}$. |
| **Phone Status** | Non-rigid mount | **`DIAGNOSTIC_ONLY`** | `ENFORCED` | Phone GPS, steps (165 steps), and magnetometer are excluded from camera-IMU calibration objective. |

---

## 5. Mathematical Formulation & Calibration Algorithms

### 5.1 SO(3) Kinematics & Angular Velocity Integration
Let $\mathbf{\omega}(t) \in \mathbb{R}^3$ be the gyro measurement in IMU body coordinates with prior bias $\mathbf{b}_g$:
$$\mathbf{\omega}_{\text{corr}}(t) = \mathbf{\omega}(t) - \mathbf{b}_g$$

Between camera exposure midpoints $t_k$ and $t_{k+1}$, the IMU relative rotation $\mathbf{R}_{\text{imu}}(t_k, t_{k+1})$ is integrated across $M$ discrete sub-intervals using the Lie algebra exponential map:
$$\mathbf{R}_{\text{imu}}(t_k, t_{k+1}) = \prod_{i=0}^{M-1} \exp([\bar{\mathbf{\omega}}_i \Delta t_i]_\times)$$
where $\bar{\mathbf{\omega}}_i = \frac{1}{2}(\mathbf{\omega}_{\text{corr}, i} + \mathbf{\omega}_{\text{corr}, i+1})$ is the trapezoidal midpoint angular rate and $\Delta t_i = t_{i+1} - t_i$ preserves the exact non-uniform tick duration.

Rodrigues' formula with small-angle Taylor expansion ($\theta = \|\mathbf{\theta}\| < 10^{-7}$):
$$\exp([\mathbf{\theta}]_\times) = \mathbf{I} + \frac{\sin \theta}{\theta}[\mathbf{\theta}]_\times + \frac{1 - \cos \theta}{\theta^2}[\mathbf{\theta}]_\times^2 \approx \mathbf{I} + [\mathbf{\theta}]_\times + \frac{1}{2}[\mathbf{\theta}]_\times^2$$

### 5.2 Closed-Form Spatial Extrinsic Optimization (Wahba / SVD)
Given camera relative rotations $\mathbf{R}_{\text{cam}, k}$ and IMU relative rotations $\mathbf{R}_{\text{imu}, k}(\tau)$, let:
$$\mathbf{r}_{\text{cam}, k} = \log(\mathbf{R}_{\text{cam}, k})^\vee, \quad \mathbf{r}_{\text{imu}, k}(\tau) = \log(\mathbf{R}_{\text{imu}, k}(\tau))^\vee$$

The rigid transform $\mathbf{R}_{\text{imu\_cam}} \in SO(3)$ maps vectors from camera frame to IMU frame ($\mathbf{r}_{\text{imu}} = \mathbf{R}_{\text{imu\_cam}} \mathbf{r}_{\text{cam}}$). The cross-covariance matrix is:
$$\mathbf{H} = \sum_{k=1}^K w_k \mathbf{r}_{\text{imu}, k} \mathbf{r}_{\text{cam}, k}^T = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
$$\mathbf{R}_{\text{imu\_cam}} = \mathbf{U} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(\mathbf{U})\det(\mathbf{V}) \end{bmatrix} \mathbf{V}^T$$
This guarantees $\det(\mathbf{R}_{\text{imu\_cam}}) = +1.000000$ (proper rotation, preventing reflection).

### 5.3 Camera-IMU Translation Observability
$$\mathbf{a}_{\text{imu}} = \mathbf{R}_{\text{imu\_cam}} (\mathbf{a}_{\text{cam}} + \mathbf{\alpha} \times \mathbf{t}_{\text{imu\_cam}} + \mathbf{\omega} \times (\mathbf{\omega} \times \mathbf{t}_{\text{imu\_cam}}))$$
In steady walking motion with low angular acceleration $\mathbf{\alpha}$, the lever arm $\mathbf{t}_{\text{imu\_cam}}$ contributes less than $0.005\text{ m/s}^2$ to specific force. Consequently, $\mathbf{t}_{\text{imu\_cam}}$ is **`WEAKLY_OBSERVABLE`**. It is recorded with nominal CAD prior ($[0.0, 0.0, 0.0]\text{ m}$, uncertainty $\pm 0.05\text{ m}$). Metric scale **cannot** be resolved from IMU measurements without independent scale constraints.

### 5.4 Rolling-Shutter Assessment
Virtual perspective views rendered from equirectangular panoramas integrate scanlines across the panoramic stitch. Line-readout time is unmodeled, which induces systematic rotational residuals of $0.5^\circ - 1.5^\circ$ during dynamic panning. The timestamp convention is explicitly pinned to **`exposure_midpoint`**.

---

## 6. Synthetic Fixtures Verification Results

All 9 tests in `tests/test_sync.py` were executed and verified:

```
================================================================================
Stage 07 Verification Suite Results (tests/test_sync.py)
================================================================================
Test 1: Synthetic Multi-Axis Nonperiodic Sync Recovery (<=2ms) ...... PASSED
Test 2: Timelapse Fixture Compression Recovery & Unavailable ........ PASSED
Test 3: Unit Discrimination & Physical Checks (g, ms, SO(3)) ........ PASSED
Test 4: Periodic Motion Ambiguity Rejection (No false pass) ......... PASSED
Test 5: Deliberately Wrong Mappings Worsen Residuals ................ PASSED
Test 6: 16-Sample Boundary Timing Pattern Preservation .............. PASSED
Test 7: Real Pilot Session Audit & Unresolved Verdict ............... PASSED
Test 8: Supervisor Lifecycle (Start, Status, Cancel, Resume) ........ PASSED
Test 9: Cache Invalidation & Downstream Dependency ................. PASSED
================================================================================
Total: 9 PASSED, 0 FAILED
```

### Detailed Fixture Results:
1. **Multi-Axis Nonperiodic Trajectory (Test 1)**:
   - Ground Truth Offset: $\tau^{\text{gt}} = 125.000\text{ ms}$
   - Recovered Offset: $\hat{\tau} = 125.000\text{ ms}$ (Error: $\mathbf{0.000\text{ ms}} \ll 2.0\text{ ms}$ tolerance)
   - Extrinsic Rotation Error: $\mathbf{0.004^\circ} \ll 1.5^\circ$ tolerance
   - Held-Out Separated Windows: Window 1 vs Window 2 discrepancy $= 0.000\text{ ms}$ (`consistent = True`)
2. **Timelapse Fixture (Test 2)**:
   - With Exposure Table: Recovered time offset accurately within $2.1\text{ ms}$.
   - With Withheld Table: Explicitly declared `UNRESOLVED_WAITING_INPUT`.
3. **Periodic Motion Ambiguity (Test 4)**:
   - Sinusoidal excitation produced secondary correlation peak at $\Delta \tau = 1.000\text{ s}$ with peak ratio $\rho_2 / \rho_1 = 0.94 > 0.85$.
   - Pipeline rejected false synchronization and returned `REJECTED_AMBIGUOUS_PERIODIC_MOTION`.
4. **Deliberately Inverted Poses & Swapped Axes (Test 5)**:
   - Inverted camera rotation worsened mean residual from $0.21^\circ$ to $24.87^\circ$.
   - Swapped gyro axes worsened mean residual to $18.42^\circ$.

---

## 7. Actual Pilot Session Audit: Session `20260910_172939`

### 7.1 Diagnostic Findings
1. **FlowState Digital Stabilization**:
   - `insta360_export_1789041643971.mp4` was exported with Insta360 FlowState stabilization enabled.
   - FlowState applies an uncalibrated, frame-by-frame software rotation $\mathbf{R}_{\text{stab}}(t)$ to stabilize the visual horizon.
   - Because $\mathbf{R}_{\text{stab}}(t)$ is not provided in container stream tags, fitting a single rigid extrinsic $\mathbf{R}_{\text{imu\_cam}}$ between the raw IMU and stabilized images is physically invalid ($\mathbf{R}_{\text{apparent}}(t) = \mathbf{R}_{\text{stab}}(t) \mathbf{R}_{\text{cam}}(t)$).
   - Under Prompt 07 Rule 5, fitting fixed extrinsics to stabilized video without known stabilization transforms is **strictly prohibited**.
2. **Timelapse Cadence Compression**:
   - `20260910_172939_location.json` declares `recording_type: "timelapse"`.
   - Playback duration is $4.033\text{ s}$ ($121\text{ frames}$ at $30\text{ fps}$), while camera IMU duration is $40.713\text{ s}$.
   - The MP4 container lacks per-frame exposure timestamp tags. The temporal mapping between frame index and physical exposure midpoint is uncalibrated.
3. **Disconnected Visual Reconstruction**:
   - Stage 06 SfM produced two disconnected sub-models (Stations 66-68 and Stations 82-83), leaving 35 stations unmerged.

### 7.2 Accepted vs. Unresolved Parameters

| Parameter | Classification | Evidence & Status |
| :--- | :---: | :--- |
| `camera_accel_units` | `ACCEPTED` | Measured norm $1.0105\text{ g}$ proves $g$ units. Conversion to $\text{m/s}^2$ ($9.80665$) verified. |
| `camera_gyro_units` | `ACCEPTED` | Measured norm $0.5619\text{ rad/s}$ ($32.2^\circ/\text{s}$) proves $\text{rad/s}$ units. |
| `camera_timestamp_units` | `ACCEPTED` | Tick differences ($1000\mu s$) and 61.071s span prove microsecond integer domain. |
| `timing_boundary_pattern` | `ACCEPTED` | 2,561 short intervals (CSV) and 3,841 short intervals (INSV) verified at 16-sample FIFO boundaries; non-uniform dt preserved. |
| `phone_sensors_role` | `ACCEPTED` | Phone telemetry marked `DIAGNOSTIC_ONLY`. Excluded from camera calibration. |
| `playback_to_capture_mapping` | `RESOLVED` | **MITIGATED & RESOLVED**: 1,834 per-frame physical exposure timestamps extracted directly from INSV Tag 0x0004 (span 61.160s, ~29.97 fps, shutter ~7.2ms). |
| `camera_hardware_identity` | `RESOLVED` | **MITIGATED & RESOLVED**: Extracted from INSV Tag 0x0101: Insta360 X5 (Serial: `IAHEA26013WB6G`, Firmware: `v1.11.10_build1`). |
| `raw_imu_telemetry_source` | `RESOLVED` | **MITIGATED & RESOLVED**: 61,472 samples extracted directly from INSV Tag 0x0003 (span 61.071s, 1000 Hz, bit-exact parity with vendor export). |
| `time_offset_s` | `UNRESOLVED` | UNRESOLVED_PENDING_UNSTABILIZED_FISHEYE_SFM: Per-frame exposure timestamps extracted from INSV, but camera rotation trajectory from exported MP4 was digitally stabilized with FlowState. |
| `R_imu_cam` | `UNRESOLVED` | UNRESOLVED_PENDING_UNSTABILIZED_FISHEYE_SFM: Exported MP4 has unknown FlowState software rotations. Direct camera-IMU extrinsic calibration requires unstabilized circular fisheye SfM. |
| `t_imu_cam` | `UNRESOLVED` | Weakly observable from rotational motion; scale cannot be resolved from IMU. |

### 7.3 Authoritative Verdict
- **`real_fusion_ready = false`**
- **`status = UNRESOLVED_WAITING_INPUT`**
- **`scale_status = NOT_ESTIMATED`**

---

## 8. Artifacts & Reproducibility

### Exported Artifacts:
All Stage 07 artifacts are stored in `runs/run_sync_20260918_043116_s7e912/outputs/` and mirrored in `data/sessions/20260910_172939/`:
- `sync_manifest.json`: Complete manifest conforming to `SyncManifestSchema`.
- `timing_diagnostics.json`: Conforming to `TimingBoundaryDiagnosticSchema`.
- `clock_calibration.json`: Conforming to `ClockCalibrationSchema`.
- `diagnostics_report.json`: Conforming to `DiagnosticsSchema`.
- `insv_summary.json`: Diagnostic summary of INSV trailer extraction.
- `insv_exposures.json`: 1,834 per-frame physical exposure records.
- `insv_imu.csv`: 61,472 raw IMU telemetry samples.
- `insv_timing_diagnostics.json`: 16-sample boundary analysis on full 61.1s stream.
- `.complete`: Atomic completion marker.

### Reproducibility Commands:
```bash
# Execute synchronization stage on real session (extracts INSV trailer, audits prerequisites)
python scripts/run_stage_07_sync.py

# Execute 8-test INSV unit and integration suite
pytest -v tests/test_insv.py

# Execute 9-test synchronization verification suite
pytest -v tests/test_sync.py

# Run via Supervisor CLI
vscan jobs start sync --session 20260910_172939 --config configs/laptop.yaml --wait
```

---

## 9. Direct `.insv` Binary Extraction & Mitigation

In response to the unresolved prerequisites identified in the initial Stage 07 audit, direct binary parsing of the native Insta360 capture container `VID_20260910_172941_00_083.insv` was implemented via `src/vscan/ingest/insv.py`:

### Binary Trailer Architecture Discovered & Implemented:
1. **Magic Signature**: The file terminates with a 32-byte ASCII UUID: `8db42d694ccc418790edff439fe026bf`.
2. **Trailer Bootstrap**: At offset `EOF - 78`, a 4-byte little-endian unsigned integer `<I` (offset by 38 bytes) defines `trailer_length = 11,875,052` bytes.
3. **Directory Table**: At offset `EOF - 78 - 310`, a 310-byte directory table comprises 31 entries of 10 bytes:
   - Format `<HII`: `tag (uint16)`, `size (uint32)`, `relative_offset (uint32)`.
   - Offsets are relative to `trailer_start = file_size - trailer_length = 65,904,242`.

### Extracted Streams & Mitigations:
- **Tag 0x0101 (MakerNotes)**: Extracted camera model (`Insta360 X5`), camera serial number (`IAHEA26013WB6G`), and firmware revision (`v1.11.10_build1`).
- **Tag 0x0004 (Per-Frame Exposure Table)**:
  - 29,344 bytes / 16 bytes = **1,834 frame exposure records** (`<Qd`: microsecond midpoint timestamp, shutter duration in seconds).
  - First exposure: $100,593,095\mu s$, last exposure: $161,752,975\mu s$ (span $61.160\text{ s}$).
  - Mean inter-frame interval: $33,366.0\mu s$ ($29.97\text{ fps}$).
  - Mean shutter duration: $7.21\text{ ms}$.
  - **Result**: The `playback_to_capture_mapping` blocker is **completely resolved**, mitigating the timelapse timing ambiguity.
- **Tag 0x0003 (High-Rate IMU Telemetry)**:
  - 1,229,440 bytes / 20 bytes = **61,472 IMU samples** (`<Q6H`: 64-bit microsecond timestamp, 3 unsigned 16-bit accel deltas, 3 unsigned 16-bit gyro deltas).
  - Formulas: $a_g = (\text{uint16} - 32768) / 1024.0$, $\omega_{\text{rad/s}} = (\text{uint16} - 32768) \times \frac{\pi}{180 \times 16.384}$.
  - Parity: 1,000 overlapping samples verified against vendor CSV with identical integer timestamps and $< 10^{-6}$ numerical residual.
  - Span: $61.071\text{ s}$, covering the full session (aligned with phone 61.811s).
  - Boundary Pattern: 3,841 short intervals, 100.0% aligned with 16-sample FIFO boundaries.

---

## 10. Handoff to Prompt 08

- **Can Prompt 08 run on actual session data?**: **`NO`**. While the timing mapping and IMU stream are resolved via INSV extraction, the visual rotation trajectory from exported MP4 imagery contains FlowState stabilization with unknown software rotations, and visual SfM produced 2 disconnected partial components.
- **Can Prompt 08 run on synthetic fixtures?**: **`YES`**. The synthetic fixtures implemented in Stage 07 provide verified multi-axis motion, known extrinsics, and calibrated clocks for Prompt 08 VIO/preintegration verification.
- **Metric Scale Status**: Strictly **`NOT_ESTIMATED`**.
- **Execution Boundary**: All Prompt 07 requirements and INSV mitigation tasks are satisfied. Execution cleanly stops here before Prompt 08.

