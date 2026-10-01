# stella_vslam + Pangolin live viewer — full setup & run guide

Self-contained build guide for `stella_vslam` (the stella-cv fork, actively
maintained successor to OpenVSLAM) with the Pangolin desktop viewer, so you
can watch tracking/mapping happen live in a 3D window while it processes a
video or image sequence. Everything here is copy-pasteable; by the end you'll
have working `run_video_slam`, `run_image_slam`, and `dump_map_ply` binaries.

Tested on Ubuntu 22.04, GCC (default), 4GB+ GPU not required (Pangolin's
viewer works fine on CPU-only OpenGL, but a discrete GPU makes the 3D view
smoother on larger maps).

## 0. Architecture — what you're actually building

stella_vslam ships as **three separate git repos** that get built and
installed independently, then linked together:

```
stella_vslam            <- the core SLAM library (tracking, mapping, loop closing).
                            No viewer code at all, no idea Pangolin exists.
pangolin_viewer         <- thin viewer library. Depends on BOTH stella_vslam
                            (to read its map/frame publisher classes) and
                            Pangolin (to actually draw).
stella_vslam_examples   <- the executables you actually run (run_video_slam,
                            run_image_slam, dump_map_ply, ...). At configure
                            time it does find_package(pangolin_viewer QUIET) --
                            if that succeeds, it compiles in `--viewer
                            pangolin_viewer` support; if you skip installing
                            pangolin_viewer, you still get a working headless
                            SLAM binary, just no `--viewer` option.
```

Below that, stella_vslam itself depends on: Eigen3, OpenCV, g2o (bundle
adjustment backend), yaml-cpp (config files), FBoW (bag-of-words loop
closure — vendored as a submodule, built automatically), and optionally
backward-cpp (stack traces on crash).

Pick one root directory to hold all of this, e.g.:

```bash
export SLAM_ROOT=$HOME/stella_vslam_workspace
mkdir -p "$SLAM_ROOT"
cd "$SLAM_ROOT"
```

Every command below assumes you're working under `$SLAM_ROOT` unless stated
otherwise. Every dependency installs to **its own** `install/` subfolder
(not `/usr/local`) — this keeps versions isolated and reproducible, and is
exactly what lets `CMAKE_PREFIX_PATH` wire everything together explicitly in
the later steps instead of relying on system-wide search paths.

## 1. System packages (apt)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential pkg-config cmake git wget curl tar unzip \
  `# g2o deps` \
  libatlas-base-dev libsuitesparse-dev libglew-dev \
  `# OpenCV deps` \
  libgtk-3-dev libjpeg-dev libpng++-dev libtiff-dev libopenexr-dev \
  libwebp-dev ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
  libswscale-dev libtbb-dev \
  `# backward-cpp (stack traces) deps` \
  binutils-dev \
  `# misc` \
  libyaml-cpp-dev sqlite3 libsqlite3-dev
```

This list is taken directly from stella_vslam's own `Dockerfile.desktop`
(the officially-tested recipe for a viewer-enabled build), so it's not a
guess — it's what upstream actually uses.

## 2. Eigen 3.3.7

Ubuntu 22.04's system Eigen (3.4.0 via `libeigen3-dev`) also works and is
faster to get — `sudo apt-get install libeigen3-dev` and skip to step 3 if
you'd rather not build from source. Building the exact pinned version
upstream tests against:

```bash
cd "$SLAM_ROOT"
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2
tar xf eigen-3.3.7.tar.bz2 && rm eigen-3.3.7.tar.bz2
cd eigen-3.3.7 && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SLAM_ROOT/eigen/install" ..
make -j"$(nproc)" && make install
```

## 3. g2o (bundle adjustment backend)

stella_vslam needs the **upstream RainerKuemmerle/g2o** (not a fork). Pin a
recent commit for reproducibility:

```bash
cd "$SLAM_ROOT"
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
git checkout 20230223_git   # pinned by stella_vslam's own Dockerfile
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$SLAM_ROOT/g2o/install" \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_UNITTESTS=OFF \
      -DG2O_USE_CHOLMOD=OFF \
      -DG2O_USE_CSPARSE=ON \
      -DG2O_USE_OPENGL=OFF \
      -DG2O_USE_OPENMP=OFF \
      -DG2O_BUILD_APPS=OFF \
      -DG2O_BUILD_EXAMPLES=OFF \
      -DG2O_BUILD_LINKED_APPS=OFF \
      ..
make -j"$(nproc)" && make install
```

## 4. OpenCV

If you already have OpenCV >= 3.3.1 (4.x recommended) installed system-wide
or in a conda env, you can skip this and just make sure `find_package(OpenCV)`
resolves to it later. To build a known-good version from source:

```bash
cd "$SLAM_ROOT"
wget -q https://github.com/opencv/opencv/archive/4.7.0.zip
unzip -q 4.7.0.zip && rm 4.7.0.zip
cd opencv-4.7.0 && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$SLAM_ROOT/opencv/install" \
      -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_JASPER=OFF \
      -DBUILD_OPENEXR=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF \
      -DBUILD_PROTOBUF=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_dnn=OFF \
      -DBUILD_opencv_ml=OFF -DBUILD_opencv_python_bindings_generator=OFF \
      -DENABLE_CXX11=ON -DENABLE_FAST_MATH=ON \
      -DWITH_EIGEN=ON -DWITH_FFMPEG=ON -DWITH_TBB=ON -DWITH_OPENMP=ON \
      ..
make -j"$(nproc)" && make install
```

## 5. Pangolin (the actual 3D viewer library)

```bash
cd "$SLAM_ROOT"
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
git checkout eab3d3449a33a042b1ee7225e1b8b593b1b21e3e   # pinned, known-working with pangolin_viewer
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$SLAM_ROOT/Pangolin/install" \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_PANGOLIN_DEPTHSENSE=OFF -DBUILD_PANGOLIN_FFMPEG=OFF \
      -DBUILD_PANGOLIN_LIBDC1394=OFF -DBUILD_PANGOLIN_LIBJPEG=OFF \
      -DBUILD_PANGOLIN_LIBOPENEXR=OFF -DBUILD_PANGOLIN_LIBPNG=OFF \
      -DBUILD_PANGOLIN_LIBTIFF=OFF -DBUILD_PANGOLIN_LIBUVC=OFF \
      -DBUILD_PANGOLIN_LZ4=OFF -DBUILD_PANGOLIN_OPENNI=OFF \
      -DBUILD_PANGOLIN_OPENNI2=OFF -DBUILD_PANGOLIN_PLEORA=OFF \
      -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_TELICAM=OFF \
      -DBUILD_PANGOLIN_UVC_MEDIAFOUNDATION=OFF -DBUILD_PANGOLIN_V4L=OFF \
      -DBUILD_PANGOLIN_ZSTD=OFF \
      ..
make -j"$(nproc)" && make install
```

(All those `BUILD_PANGOLIN_*=OFF` flags just disable camera-driver backends
Pangolin can optionally support — DepthSense, OpenNI, V4L webcams, etc. None
of that is needed to display a SLAM map; leaving them ON also works, it just
means more optional third-party deps to satisfy at configure time.)

**If CMake can't find OpenGL/GLEW/X11 headers here**, install
`libgl1-mesa-dev libglu1-mesa-dev libx11-dev` and re-run cmake.

## 6. backward-cpp (optional — nicer crash stack traces)

Skip this and skip `-DUSE_STACK_TRACE_LOGGER=ON` in step 9 if you don't care
about symbolized crash backtraces.

```bash
cd "$SLAM_ROOT"
git clone https://github.com/bombela/backward-cpp.git
cd backward-cpp
git checkout 5ffb2c879ebdbea3bdb8477c671e32b1c984beaa
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SLAM_ROOT/backward-cpp/install" ..
make -j"$(nproc)" && make install
```

## 7. stella_vslam (the core library)

```bash
cd "$SLAM_ROOT"
git clone --recursive https://github.com/stella-cv/stella_vslam.git
cd stella_vslam
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$SLAM_ROOT/stella_vslam/install" \
      -DCMAKE_PREFIX_PATH="$SLAM_ROOT/eigen/install;$SLAM_ROOT/g2o/install;$SLAM_ROOT/opencv/install" \
      -DBOW_FRAMEWORK=FBoW \
      -DUSE_OPENMP=OFF \
      -DBUILD_TESTS=OFF \
      ..
make -j"$(nproc)" && make install
```

Notes:
- `--recursive` matters: FBoW is a git submodule vendored inside
  `stella_vslam/3rd/FBoW` and gets built as part of this same step. If you
  already cloned without `--recursive`, run
  `git submodule update --init --recursive` inside `stella_vslam/` first.
- Drop any of the three paths from `-DCMAKE_PREFIX_PATH` that don't apply
  (e.g. if you used system/apt OpenCV or Eigen instead of building them).
- `BOW_FRAMEWORK=FBoW` is the modern default (DBoW2 is the legacy
  alternative upstream still supports but FBoW is faster and what the vocab
  file in step 10 matches).

## 8. pangolin_viewer

```bash
cd "$SLAM_ROOT"
git clone --recursive https://github.com/stella-cv/pangolin_viewer.git
cd pangolin_viewer
git checkout f3564675d420bcdbe6380b29b5e2f2b623987dd0   # pinned, known-working with the stella_vslam/Pangolin versions above
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$SLAM_ROOT/pangolin_viewer/install" \
      -DCMAKE_PREFIX_PATH="$SLAM_ROOT/Pangolin/install;$SLAM_ROOT/stella_vslam/install;$SLAM_ROOT/opencv/install" \
      ..
make -j"$(nproc)" && make install
```

## 9. stella_vslam_examples (the actual executables)

```bash
cd "$SLAM_ROOT"
git clone --recursive https://github.com/stella-cv/stella_vslam_examples.git
cd stella_vslam_examples
git checkout 3606f68c9c3fb05a838e992230cb4a17106a7c41   # pinned, known-working with the stella_vslam/pangolin_viewer versions above
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$SLAM_ROOT/stella_vslam/install;$SLAM_ROOT/g2o/install;$SLAM_ROOT/pangolin_viewer/install;$SLAM_ROOT/Pangolin/install;$SLAM_ROOT/opencv/install" \
      ..
      # add -DUSE_STACK_TRACE_LOGGER=ON here if you built backward-cpp in step 6
make -j"$(nproc)"
```

**Check the cmake configure output** for this line (or its absence):
```
-- Found pangolin_viewer: ...
```
If `pangolin_viewer` isn't found, `find_package(pangolin_viewer QUIET)`
silently no-ops and you'll get a headless-only build (no crash, but no
`--viewer pangolin_viewer` option at runtime either — the binary won't even
recognize the flag as a valid viewer name). Fix by double-checking
`pangolin_viewer_DIR` is reachable via the `CMAKE_PREFIX_PATH` above and
re-running cmake from a clean `build/` dir.

You should now have these binaries in `stella_vslam_examples/build/`:
`run_video_slam`, `run_image_slam`, `run_camera_slam`, `run_euroc_slam`,
`run_kitti_slam`, `run_tum_rgbd_slam`, `run_loop_closure`, `dump_map_ply`.

## 10. ORB vocabulary file (required at runtime)

stella_vslam needs a pre-trained bag-of-words vocabulary for loop closure /
relocalization. Download the official one (44MB):

```bash
mkdir -p "$SLAM_ROOT/vocab"
curl -L -o "$SLAM_ROOT/vocab/orb_vocab.fbow" \
  https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow
```

## 11. Runtime library path

All the shared libs you just built live in non-standard `install/lib`
folders, so the dynamic linker needs to be told where to find them every
time you run a binary:

```bash
export LD_LIBRARY_PATH="$SLAM_ROOT/stella_vslam/install/lib:$SLAM_ROOT/g2o/install/lib:$SLAM_ROOT/pangolin_viewer/install/lib:$SLAM_ROOT/Pangolin/install/lib:$SLAM_ROOT/opencv/install/lib:$LD_LIBRARY_PATH"
```

(Add this to your `.bashrc` if you'll be running this often, or just export
it once per shell session before running the binaries below.)

---

## 12. Camera config file (`config.yaml`)

stella_vslam is configured entirely through one YAML file per camera/video.
Here's a fully-annotated **equirectangular** (360°) example — this is the
camera model for any true 360 video (Insta360, Ricoh Theta, etc. once
stitched to a single equirect frame); for an ordinary phone/webcam use
`model: "perspective"` instead and add `fx/fy/cx/cy/k1/k2/p1/p2/k3` under
`Camera` (see `stella_vslam/example/*/` for perspective/fisheye examples).

```yaml
Camera:
  name: "my 360 camera"
  setup: "monocular"          # stella_vslam also supports stereo/RGBD setups
  model: "equirectangular"
  fps: 29.97                  # get this right -- wrong fps skews velocity assumptions
  cols: 3840                  # full equirect frame width
  rows: 1920                  # full equirect frame height -- true equirect
                               # footage is always 2:1 (cols=2*rows); the
                               # code maps width to the full 360deg longitude
                               # range and height to the full 180deg latitude
                               # range independently and won't error on a
                               # different ratio, it'll just silently compute
                               # wrong bearing vectors, so get this right
  color_order: "RGB"          # or "BGR", "GRAY" -- must match your decoder's output

Preprocessing:
  min_size: 800
  # Mask out the top/bottom bands of the equirect frame -- these correspond
  # to straight up/down (ceiling/floor), where ORB features are unreliable
  # (low texture, or your own tripod/selfie-stick handle in frame) and where
  # equirect distortion is worst. Each row is [x_min, x_max, y_min, y_max]
  # as FRACTIONS of the frame (0.0-1.0), and pixels inside any listed box
  # are excluded from feature detection.
  mask_rectangles:
    - [0.0, 1.0, 0.0, 0.1]     # top 10% (looking straight up)
    - [0.0, 1.0, 0.84, 1.0]    # bottom 16% (looking straight down)

Feature:
  name: "default ORB feature extraction setting"
  scale_factor: 1.2           # pyramid downscale ratio per level
  num_levels: 8                # pyramid levels
  ini_fast_threshold: 20      # FAST corner threshold, tried first
  min_fast_threshold: 7       # fallback threshold if too few corners found

Initializer:
  use_fixed_seed: true        # reproducible RANSAC in the map initializer --
                               # without this, re-running on identical input
                               # can give a different map every time (the
                               # initializer's RANSAC has no seed by default)

Mapping:
  backend: "g2o"
  baseline_dist_thr_ratio: 0.02
  redundant_obs_ratio_thr: 0.95
  num_covisibilities_for_landmark_generation: 20
  num_covisibilities_for_landmark_fusion: 20
  residual_deg_thr: 0.4

Tracking:
  backend: "g2o"
  num_matches_thr: 6           # min feature matches to keep tracking
  num_tracked_lms_thr: 10      # min tracked landmarks to keep tracking

LoopDetector:
  backend: "g2o"
  enabled: true
  reject_by_graph_distance: true
  min_distance_on_graph: 50    # min keyframes apart (along the pose graph)
                                # before two keyframes are even considered
                                # for a loop closure -- prevents "closing a
                                # loop" against the frame from 2 seconds ago

GraphOptimizer:
  min_num_shared_lms: 200

GlobalOptimizer:
  thr_neighbor_keyframes: 100

System:
  map_format: "msgpack"        # map.msg save format
  num_grid_cols: 96            # feature-matching acceleration grid
  num_grid_rows: 48
```

Get `cols`/`rows`/`fps` from your actual video with:
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate -of csv=p=0 your_video.mp4
```

## 13. Run it live with the Pangolin viewer

```bash
"$SLAM_ROOT/stella_vslam_examples/build/run_video_slam" \
  -v "$SLAM_ROOT/vocab/orb_vocab.fbow" \
  -m /path/to/your_video.mp4 \
  -c /path/to/config.yaml \
  --viewer pangolin_viewer \
  -o /path/to/output/map.msg \
  --eval-log-dir /path/to/output/ \
  --log-level info
```

Two windows pop up:
- **A Pangolin 3D window** — the growing point cloud + keyframe camera
  frustums + pose graph, with a menu panel down the left side: "Follow
  Camera", "Show Grid", "Show Keyframes", "Show Landmarks", "Show Local
  Map", "Show covisibility graph", "Show essential graph", "Show image",
  "Mapping" (toggle mapping on/off), "Loop Detection" (toggle on/off),
  "Pause", "Reset", "Terminate". Standard Pangolin mouse controls in the 3D
  view: **left-drag** orbits, **right-drag/scroll** zooms, **middle-drag**
  pans.
- **A plain OpenCV window** — the current video frame with tracked ORB
  keypoints drawn on top: while actively mapping, each keypoint linked to a
  3D landmark is colored by a Turbo colormap keyed to that landmark's
  observed-ratio (how consistently it's been seen across keyframes — dark
  blue = newer/less-confirmed, red/yellow = long-tracked and reliable);
  during localization-only tracking (`--disable-mapping`) matched keypoints
  are drawn in a single fixed cyan-ish color instead. Either way, this
  window is the fastest way to eyeball whether tracking is finding enough
  keypoints frame-to-frame.

Full flag reference for `run_video_slam` (same set of flags apply to
`run_image_slam`, just swap `-m <video>` for `-d <image_dir>`):

| Flag | Meaning |
|---|---|
| `-v, --vocab` | path to `orb_vocab.fbow` |
| `--without-vocab` | skip loading a vocabulary (disables loop closure/relocalization) |
| `-m, --video` | input video file |
| `-c, --config` | path to the `config.yaml` from step 12 |
| `--mask` | optional grayscale mask image (alternative/addition to `mask_rectangles`) |
| `--frame-skip N` | process every Nth frame |
| `-s, --start-time MS` | start partway into the video |
| `--no-sleep` | process as fast as possible instead of pacing to the video's real fps (use this for offline/batch processing) |
| `-t, --start-timestamp` | timestamp to assign to the first frame (for aligning against external logs) |
| `--wait-loop-ba` | block program exit until a running loop-closure bundle adjustment finishes |
| `--auto-term` | close the viewer automatically once the input ends (otherwise it stays open until you click Terminate) |
| `--eval-log-dir DIR` | write `frame_trajectory.txt` + `keyframe_trajectory.txt` (TUM format: `timestamp tx ty tz qx qy qz qw`) + timing stats here |
| `-i, --map-db-in PATH` | load a previously-saved map and continue/relocalize into it |
| `-o, --map-db-out PATH` | save the final map (msgpack) here |
| `--disable-mapping` | localization-only mode against a loaded map, no new mapping |
| `--temporal-mapping` | keep mapping active even during relocalization gaps |
| `--viewer` | `pangolin_viewer`, `iridescence_viewer`, `socket_publisher`, or `none` — only the ones actually compiled in (step 9) are valid |
| `--log-level` | `trace`/`debug`/`info`/`warn`/`error` |

For a folder of still images (e.g. a phone timelapse) instead of a video
file, use `run_image_slam` with `-d /path/to/image_dir` in place of
`-m video.mp4` — it processes every image in that directory in filename
sort order. Everything else (config, vocab, viewer, output flags) is
identical.

## 14. Export the map to a point cloud (PLY)

```bash
"$SLAM_ROOT/stella_vslam_examples/build/dump_map_ply" \
  -v "$SLAM_ROOT/vocab/orb_vocab.fbow" \
  -c /path/to/config.yaml \
  -i /path/to/output/map.msg \
  -o /path/to/output/landmarks.ply
```
Opens in MeshLab/CloudCompare/Blender like any other PLY point cloud.
Landmark points come out white; if you also see colored points, those are
keyframe-camera-frustum wireframe decoration vertices `dump_map_ply` adds
for visualization — filter to pure-white RGB if you want landmarks only.

## 15. Known issue: exit code -11 on viewer close

`run_video_slam`/`run_image_slam` can segfault (exit code -11) during
Pangolin's own GL context teardown **after** it has already finished
tracking and written `map.msg` successfully — this is a viewer cleanup bug,
not a tracking failure. If you're scripting this, check that the output
map file exists and is non-empty before treating a nonzero exit code as
fatal:

```bash
run_video_slam ... -o map.msg
if [ $? -ne 0 ] && [ -s map.msg ]; then
  echo "viewer cleanup crashed but map.msg was saved fine -- not fatal"
fi
```

## 16. Troubleshooting

- **"empty map" / "no valid frames" / 0 keyframes on a real-looking
  video**: check `ffprobe` output for the actual frame count and interval.
  A "video" that's really a very sparse timelapse (say, one frame every
  several seconds) violates stella_vslam's implicit small-motion-between-
  frames assumption for feature matching, and tracking will simply never
  initialize. Fix by increasing the effective frame rate (e.g. AI frame
  interpolation to synthesize in-between frames) rather than tuning SLAM
  parameters — no amount of threshold-tweaking fixes too-large baseline
  between consecutive frames.
- **CMake can't find `pangolin_viewer`/`stella_vslam`/`Pangolin` in a later
  step**: the `_DIR` cmake-config files land at
  `<install_prefix>/lib/cmake/<PackageName>/`. Double check your
  `-DCMAKE_PREFIX_PATH` (step 9) lists every `install/` folder you actually
  used, semicolon-separated, no trailing slashes needed.
- **Runtime `error while loading shared libraries: libpango_*.so.0`**: you
  forgot to `export LD_LIBRARY_PATH=...` (step 11) in this shell session.
- **Map looks fine but geometry/scale is wrong**: this is expected —
  monocular SLAM (this whole setup) has no absolute scale by construction.
  stella_vslam's units are arbitrary; recovering real-world meters needs an
  external reference (known object/room size, IMU-based dead reckoning, a
  second sensor, etc.) applied as a post-process on the saved trajectory/PLY.
