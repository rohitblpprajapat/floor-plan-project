# Stage 04 Input Audit Report: Session `20260910_172939`

- **Execution Timestamp**: `2026-09-16T11:44:19Z`
- **Input Directory**: `/home/rohit/projects/insta360_mapper/data/2026-09-10`
- **Storage Mode**: `reference`
- **Total Input Size**: `127.27 MB`
- **Overall Status**: `COMPLETE`
- **Scale Status**: `NOT_ESTIMATED` (Authoritative: Immutable baseline)

---

## 1. Status Partitioning

| Domain | Status | Operational Note |
| :--- | :--- | :--- |
| **Software Status** | `STAGE_04_COMPLETE` | Ingestion, hashing, video probing, and sensor audits verified |
| **Actual Data Status** | `COMPLETE` | All expected input files present and verified |
| **Metric Scale Status** | `NOT_ESTIMATED` | Strictly unscaled; steps/GPS/pin assumptions forbidden |

---

## 2. Ingested Input Assets & Checksums

| Filename | Asset ID | Type | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- | :---: | :--- |
| `20260910_172939_location.json` | `asset_20260910_172939_20260910_172939_location_json` | `location_registration_json` | `567` | `34a6eb169ce7336f...` |
| `20260910_172939_phone_sensors.csv` | `asset_20260910_172939_20260910_172939_phone_sensors_csv` | `phone_sensors_csv` | `743,525` | `0a036309e4acc03a...` |
| `LRV_20260910_172941_01_083.mp4` | `asset_20260910_172939_LRV_20260910_172941_01_083_mp4` | `raw_video_lrv` | `19,452,248` | `769f7826ff93700d...` |
| `VID_20260910_172941_00_083.insv` | `asset_20260910_172939_VID_20260910_172941_00_083_insv` | `raw_video_insv` | `77,779,294` | `c505476ac46fbe5d...` |
| `insta360_export_1789041643971.csv` | `asset_20260910_172939_insta360_export_1789041643971_csv` | `camera_telemetry_csv` | `5,904,033` | `9dfb8800a5f235a2...` |
| `insta360_export_1789041643971.mp4` | `asset_20260910_172939_insta360_export_1789041643971_mp4` | `exported_panoramic_video` | `29,575,783` | `78ca29a6a7e08c8b...` |

### Source-Derived Relationships

- **VENDOR_EXPORT**: `asset_20260910_172939_VID_20260910_172941_00_083_insv` -> `asset_20260910_172939_insta360_export_1789041643971_mp4`
  - *Evidence*: Matching capture timestamp sequence (1789041643971) and camera session metadata
- **PREVIEW_STREAM**: `asset_20260910_172939_VID_20260910_172941_00_083_insv` -> `asset_20260910_172939_LRV_20260910_172941_01_083_mp4`
  - *Evidence*: Simultaneous dual-stream recording indicator (LRV vs VID) from Insta360 hardware
- **IMU_EXPORT**: `asset_20260910_172939_VID_20260910_172941_00_083_insv` -> `asset_20260910_172939_insta360_export_1789041643971_csv`
  - *Evidence*: Camera IMU export matching export session ID 1789041643971

---

## 3. Video Streams & Container Probe

### Asset: `LRV_20260910_172941_01_083.mp4` (4.04s)
- **Container Format**: `mov,mp4,m4a,3gp,3g2,mj2`
- **File Size**: `19,452,248` bytes
- **Contact Sheet**: `outputs/contact_sheets/contact_sheet_LRV_20260910_172941_01_083.jpg`

| Track # | Type | Codec | Resolution | Aspect Ratio | Time Base | Frames | Cadence | Projection Candidate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | `video` | `h264` | `1664x832` | `2.00:1` | `1/30000` | 121 | Constant | `preview_equirectangular` |

- **Unknown / Non-fabricated Fields**:
  - Exposure duration: `UNKNOWN` (not stored in MP4 container stream tags)
  - Rolling-shutter readout duration: `UNKNOWN` (requires vendor EEPROM dump)

### Asset: `VID_20260910_172941_00_083.insv` (4.04s)
- **Container Format**: `mov,mp4,m4a,3gp,3g2,mj2`
- **File Size**: `77,779,294` bytes
- **Contact Sheet**: `outputs/contact_sheets/contact_sheet_VID_20260910_172941_00_083.jpg`
- **Dual-Lens Stream Verification**: `CONFIRMED` - Both required original circular fisheye streams (Track 0 and Track 1) are present.

| Track # | Type | Codec | Resolution | Aspect Ratio | Time Base | Frames | Cadence | Projection Candidate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | `video` | `hevc` | `2880x2880` | `1.00:1` | `1/30000` | 121 | Constant | `circular_fisheye_single` |
| 1 | `video` | `hevc` | `2880x2880` | `1.00:1` | `1/30000` | 121 | Constant | `circular_fisheye_single` |

- **Unknown / Non-fabricated Fields**:
  - Exposure duration: `UNKNOWN` (not stored in MP4 container stream tags)
  - Rolling-shutter readout duration: `UNKNOWN` (requires vendor EEPROM dump)

### Asset: `insta360_export_1789041643971.mp4` (4.03s)
- **Container Format**: `mov,mp4,m4a,3gp,3g2,mj2`
- **File Size**: `29,575,783` bytes
- **Contact Sheet**: `outputs/contact_sheets/contact_sheet_insta360_export_1789041643971.jpg`

| Track # | Type | Codec | Resolution | Aspect Ratio | Time Base | Frames | Cadence | Projection Candidate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | `video` | `h264` | `3840x1920` | `2.00:1` | `1/30000` | 121 | Constant | `equirectangular_360` |

- **Unknown / Non-fabricated Fields**:
  - Exposure duration: `UNKNOWN` (not stored in MP4 container stream tags)
  - Rolling-shutter readout duration: `UNKNOWN` (requires vendor EEPROM dump)

---

## 4. Recomputed Sensor Audits (Camera vs. Phone)

### 4.1 Camera IMU & Telemetry (`insta360_export_1789041643971.csv`)
- **Row Count**: `40,981` rows (recomputed from file)
- **Raw Timestamp Span**: `40,713,082` integer ticks (`100,701,912` to `141,414,994`)
- **Proposed Unit Hypothesis**: `acceleration: g (norm ~ 1.0); gyro: rad/s; timestamps: microseconds (nominal delta: 1000us = 1kHz)`
- **Proposed Physical Span**: `40.7131 seconds` (at 1 us/tick, span is ~40.713s)
- **Monotonicity**: `STRICTLY_MONOTONIC` (Duplicates: 0, Backward steps: 0)
- **Nominal Interval**: `~1,000 microseconds` (1.000 ms = 1 kHz sample rate)
- **Short Timestamp Intervals**: `2,561` occurrences
- **16-Sample Boundary Verification**: `2561 short intervals (<950); 2561 align with 16-sample boundaries (sample index % 16 == 0)`
- **Channel Norms**:
  - Acceleration Norm $\|a\|$: mean `1.0105` g (std `0.095`, min `0.6765`, max `1.3132`) -> **Hypothesis: g units**
  - Gyro Norm $\|g\|$: mean `0.5619` rad/s (std `0.3316`)
- **Zero Startup Vectors**: `0`
- **Repeated Channel Vectors**: `0`
- **GPS Constancy**: `is_constant = True` (Unique fixes: 1)
  - *Constraint Policy*: Camera GPS has 1 constant fix across entire session; must NOT be expanded into multiple constraints.

### 4.2 Phone Sensors (`20260910_172939_phone_sensors.csv`)
- **Row Count**: `6,120` rows (recomputed from file)
- **Raw Timestamp Span**: `61,811 ms` (`61.811 seconds`)
- **Proposed Unit Hypothesis**: `acceleration: m/s^2 (norm ~ 9.8); gyro: rad/s; mag: uT; timestamps: milliseconds (nominal delta: ~10ms = 100Hz)`
- **Monotonicity**: `STRICTLY_MONOTONIC`
- **Initial Zero Startup Vectors**: `2` rows at start with null vectors
- **Channel Norms**:
  - Acceleration Norm $\|a\|$: mean `9.9125` m/s^2 (std `1.3499`) -> **Hypothesis: m/s^2 units**
  - Gyro Norm $\|g\|$: mean `0.6941` rad/s
  - Magnetometer Norm $\|m\|$: mean `42.292` uT
- **Step Counter Progression**: `165 steps detected (0 -> 165)`
- **GPS Fixes**: `5822` fixes recorded (is_constant = False)
  - *Operational Policy*: Phone GPS is noisy and multi-path prone; strictly diagnostic.

### 4.3 Location Registration (`location.json`)
- **Session Label**: `20260910_172939`
- **Recording Type**: `timelapse` (Timelapse: `True`)
- **Recording Started At**: `2026-09-10T11:59:39.398343Z`
- **Building**: `Tower Alpha (North Block)` (`Building 1`)
- **Floor**: `Floor 1`
- **Start Coordinates**: `worldX=0.0, worldY=0.0`
- **Normalized Pin**: `(0.1267, 0.6395)`
- **Timelapse Advisory**: RECORDING TYPE IS TIMELAPSE: Video frame intervals do NOT correspond to real-time physical exposure intervals. Do NOT assume standard 30fps continuous motion. Exposure cadence requires independent clock investigation.
- **Scale & Registration Constraints**: `worldX/Y=0` and normalized floor pin provide **zero metric scale** and **no full map orientation**.

---

## 5. Granular Stream Validity Flags

| Stream Modality | Status | Unit Hypothesis | Diagnostic Only? | Factual Notes |
| :--- | :--- | :--- | :---: | :--- |
| `imagery` | `VALID` | `N/A` | NO | Stitched 360 MP4 and/or raw INSV files available. A 2:1 aspect ratio or .insv extension does not prove camera geometry or calibration. |
| `timing` | `PROVISIONAL` | `camera: microseconds; phone: milliseconds` | NO | Raw integer timestamps preserved without floating point precision loss. Cross-clock mapping required. |
| `camera_imu` | `VALID` | `g (acceleration norm ~ 1.0) and rad/s (gyro)` | NO | Camera IMU stream verified at ~1kHz with 2,561 short intervals at 16-sample boundaries. |
| `phone_imu` | `DIAGNOSTIC_ONLY` | `m/s^2 (acceleration norm ~ 9.8) and rad/s (gyro)` | YES | Phone sensor stream is strictly diagnostic-only unless a rigid mechanical mount to camera is proven. |
| `gps` | `CONSTANT_SINGLE_FIX` | `WGS84 degrees, altitude meters` | YES | Camera GPS contains exactly 1 static coordinate fix throughout entire recording; must NOT be expanded into multiple independent spatial position constraints. |

---

## 6. Prompt 05 Operational Readiness Verdict

- **Can Prompt 05 Use Actual Imagery**: `YES` (`True`)
- **Needs Vendor Export**: `NO` (`False` - Stitched panoramic MP4 already present locally)
- **Fixture Fallback Available**: `YES` (`True`)
- **Authoritative Scale Status**: `NOT_ESTIMATED`

### Unresolved Facts

1. Rigid spatial transformation between phone and camera is unmeasured and unverified.
1. Raw camera lens distortion and factory calibration parameters remain unextracted from INSV EEPROM.
1. Recording type is timelapse; frame cadence does not imply continuous physical motion or real exposure duration.
1. worldX=0.0, worldY=0.0 and normalized floor pin do not provide metric scale or full map registration.

---
*Report generated automatically by vscan stage 04 ingest worker.*
