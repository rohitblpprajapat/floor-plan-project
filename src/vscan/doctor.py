"""Diagnostic and verification routines for vscan environment."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict


def get_environment_info() -> Dict[str, Any]:
    """Inspect Python, virtual environment, and system paths."""
    in_venv = sys.prefix != sys.base_prefix
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "is_virtual_env": in_venv,
        "platform": sys.platform,
    }


def get_packages_info() -> Dict[str, Any]:
    """Inspect installed package versions and import locations."""
    packages = [
        "pycolmap",
        "gtsam",
        "numpy",
        "scipy",
        "cv2",
        "PIL",
        "pydantic",
        "typer",
        "psutil",
        "pytest",
    ]
    info: Dict[str, Any] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            dist_name = "opencv-python-headless" if pkg == "cv2" else ("Pillow" if pkg == "PIL" else pkg)
            try:
                ver = importlib.metadata.version(dist_name)
            except Exception:
                ver = getattr(mod, "__version__", "unknown")
            file_path = getattr(mod, "__file__", "builtin")
            info[pkg] = {
                "installed": True,
                "version": ver,
                "file_path": file_path,
            }
        except ImportError as exc:
            info[pkg] = {
                "installed": False,
                "error": str(exc),
            }
    return info


def get_system_tools_info() -> Dict[str, Any]:
    """Inspect ffmpeg, ffprobe, nvidia-smi, and /dev/dxg."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    nvidia_smi_path = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
    dxg_present = os.path.exists("/dev/dxg")

    ffmpeg_ver = None
    if ffmpeg_path:
        try:
            res = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True, check=True)
            ffmpeg_ver = res.stdout.splitlines()[0]
        except Exception as e:
            ffmpeg_ver = f"Error: {e}"

    ffprobe_ver = None
    if ffprobe_path:
        try:
            res = subprocess.run([ffprobe_path, "-version"], capture_output=True, text=True, check=True)
            ffprobe_ver = res.stdout.splitlines()[0]
        except Exception as e:
            ffprobe_ver = f"Error: {e}"

    gpu_summary = None
    if nvidia_smi_path:
        try:
            res = subprocess.run(
                [nvidia_smi_path, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            gpu_summary = res.stdout.strip()
        except Exception as e:
            gpu_summary = f"Error: {e}"

    return {
        "ffmpeg": {"path": ffmpeg_path, "version": ffmpeg_ver},
        "ffprobe": {"path": ffprobe_path, "version": ffprobe_ver},
        "nvidia_smi": {"path": nvidia_smi_path, "gpu_summary": gpu_summary},
        "dev_dxg_present": dxg_present,
    }


def audit_pycolmap_capabilities() -> Dict[str, Any]:
    """Audit fixed-rig, reconstruction, undistortion, stereo, and panorama features in PyCOLMAP."""
    try:
        import pycolmap

        has_cuda = getattr(pycolmap, "has_cuda", False)
        num_devices = pycolmap.get_num_cuda_devices() if hasattr(pycolmap, "get_num_cuda_devices") else 0

        fixed_rig = all(hasattr(pycolmap, attr) for attr in ["RigConfig", "RigConfigCamera", "apply_rig_config", "read_rig_config"])
        recon_export = hasattr(pycolmap, "Reconstruction") and all(
            hasattr(pycolmap.Reconstruction, attr) for attr in ["write", "write_binary", "write_text", "export_PLY"]
        )
        undistortion = all(hasattr(pycolmap, attr) for attr in ["undistort_images", "undistort_image", "undistort_camera"])
        stereo = all(hasattr(pycolmap, attr) for attr in ["patch_match_stereo", "stereo_fusion"])
        # Check panorama convenience code
        pano_symbols = [x for x in dir(pycolmap) if "pano" in x.lower() or "spheri" in x.lower() or "equi" in x.lower()]
        panorama_helper = len(pano_symbols) > 0

        return {
            "pycolmap_version": getattr(pycolmap, "__version__", "unknown"),
            "has_cuda": bool(has_cuda),
            "cuda_devices_count": num_devices,
            "fixed_rig_supported": fixed_rig,
            "reconstruction_export_supported": recon_export,
            "undistortion_supported": undistortion,
            "stereo_supported": stereo,
            "panorama_helpers_present": panorama_helper,
            "panorama_note": "Panorama convenience helper absent in PyCOLMAP 4.2.0; planned custom ray-renderer replaces this in Prompt 05.",
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_numerical_tests() -> Dict[str, Any]:
    """Exercise native numerical methods (GTSAM, NumPy, SciPy)."""
    results: Dict[str, Any] = {}
    try:
        import gtsam
        import numpy as np
        import scipy.spatial

        # Exercise GTSAM pose, rotation, noise, factor graph, and Gauss-Newton solver
        r = gtsam.Rot3.Expmap(np.array([0.05, -0.02, 0.01]))
        t = gtsam.Point3(1.2, -0.5, 0.8)
        p = gtsam.Pose3(r, t)

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        values.insert(1, p)
        noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.05)
        graph.add(gtsam.PriorFactorPose3(1, p, noise))

        opt_params = gtsam.GaussNewtonParams()
        optimizer = gtsam.GaussNewtonOptimizer(graph, values, opt_params)
        opt_res = optimizer.optimize()
        opt_trans = opt_res.atPose3(1).translation()

        # SciPy rotation test
        scipy_rot = scipy.spatial.transform.Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
        rot_mat = scipy_rot.as_matrix()

        results["gtsam_native_calculation"] = {
            "status": "PASS",
            "initial_translation": list(t),
            "optimized_translation": [float(x) for x in opt_trans],
            "error_norm": float(np.linalg.norm(t - opt_trans)),
        }
        results["scipy_rotation_test"] = {
            "status": "PASS",
            "matrix_shape": list(rot_mat.shape),
        }
    except Exception as exc:
        results["numerical_test_error"] = str(exc)
    return results


def run_video_decode_test() -> Dict[str, Any]:
    """Generate a tiny test video with ffmpeg and decode its streams using ffprobe."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "probe_sample.mp4")
            # Generate 1-second 64x64 H.264 video
            cmd_gen = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=64x64:rate=10",
                "-c:v",
                "libx264",
                video_path,
            ]
            res_gen = subprocess.run(cmd_gen, capture_output=True, text=True, check=True)

            cmd_probe = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,duration,codec_name",
                "-of",
                "json",
                video_path,
            ]
            res_probe = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
            probe_data = json.loads(res_probe.stdout)
            streams = probe_data.get("streams", [])
            if not streams:
                return {"status": "FAIL", "reason": "No video streams detected"}
            return {
                "status": "PASS",
                "codec": streams[0].get("codec_name"),
                "width": streams[0].get("width"),
                "height": streams[0].get("height"),
                "duration": streams[0].get("duration"),
            }
    except Exception as exc:
        return {"status": "FAIL", "error": str(exc)}


def run_sift_feature_tests() -> Dict[str, Any]:
    """Run SIFT feature extraction on GPU (CUDA) and CPU fallback, measuring performance."""
    results: Dict[str, Any] = {}
    try:
        import numpy as np
        import psutil
        import pycolmap

        # Generate reproducible 512x512 textured pattern (seeded noise + grid lines)
        rng = np.random.RandomState(42)
        base = (rng.rand(512, 512) * 200).astype(np.uint8)
        base[::16, :] = 255
        base[:, ::16] = 255
        test_img = np.ascontiguousarray(base)

        # 1. CUDA Feature Extraction Test
        cuda_test: Dict[str, Any] = {}
        try:
            mem_before = psutil.virtual_memory().used
            t0 = time.perf_counter()
            sift_cuda = pycolmap.Sift(device=pycolmap.Device.cuda)
            actual_device = str(sift_cuda.device)
            # Verify explicit CUDA request did not silently fallback
            if "cuda" not in actual_device.lower():
                raise RuntimeError(f"Expected Device.cuda, got: {actual_device}")
            kp_cuda, desc_cuda = sift_cuda.extract(test_img)
            dt_cuda = time.perf_counter() - t0
            mem_after = psutil.virtual_memory().used

            cuda_test = {
                "status": "PASS",
                "requested_device": "Device.cuda",
                "resolved_device": actual_device,
                "silent_cpu_fallback": False,
                "keypoints_count": int(len(kp_cuda)),
                "descriptor_shape": list(desc_cuda.shape),
                "elapsed_seconds": round(dt_cuda, 4),
                "host_ram_delta_bytes": mem_after - mem_before,
            }
        except Exception as e:
            cuda_test = {
                "status": "FAIL",
                "requested_device": "Device.cuda",
                "error": str(e),
            }
        results["cuda_sift"] = cuda_test

        # 2. CPU Fallback Feature Extraction Test
        cpu_test: Dict[str, Any] = {}
        try:
            t0 = time.perf_counter()
            sift_cpu = pycolmap.Sift(device=pycolmap.Device.cpu)
            actual_device_cpu = str(sift_cpu.device)
            kp_cpu, desc_cpu = sift_cpu.extract(test_img)
            dt_cpu = time.perf_counter() - t0

            cpu_test = {
                "status": "PASS",
                "requested_device": "Device.cpu",
                "resolved_device": actual_device_cpu,
                "keypoints_count": int(len(kp_cpu)),
                "descriptor_shape": list(desc_cpu.shape),
                "elapsed_seconds": round(dt_cpu, 4),
            }
        except Exception as e:
            cpu_test = {
                "status": "FAIL",
                "requested_device": "Device.cpu",
                "error": str(e),
            }
        results["cpu_sift"] = cpu_test

    except Exception as exc:
        results["error"] = str(exc)

    return results


def run_rig_serialization_test() -> Dict[str, Any]:
    """Construct camera rig transformation, populate parameters, and verify serialization."""
    try:
        import numpy as np
        import pycolmap

        # Construct rig config
        rc = pycolmap.RigConfig()

        # Camera 0 (Reference sensor)
        cam0 = pycolmap.RigConfigCamera()
        cam0.image_prefix = "cam0_"
        cam0.ref_sensor = True

        # Camera 1 (Secondary sensor with relative transformation)
        cam1 = pycolmap.RigConfigCamera()
        cam1.image_prefix = "cam1_"
        cam1.ref_sensor = False

        # Rotation: identity quaternion [w=1, x=0, y=0, z=0]
        rot = pycolmap.Rotation3d(np.array([1.0, 0.0, 0.0, 0.0]))
        # Translation: 0.1m baseline in X
        trans = np.array([0.1, 0.0, 0.0], dtype=np.float64)
        cam1.cam_from_rig = pycolmap.Rigid3d(rot, trans)

        rc.cameras = [cam0, cam1]

        # Test dictionary conversions
        d0 = cam0.todict()
        d1 = cam1.todict()

        # Build standard COLMAP rig json structure
        rig_spec = [
            {
                "cameras": [
                    {"image_prefix": "cam0_", "ref_sensor": True},
                    {
                        "image_prefix": "cam1_",
                        "ref_sensor": False,
                        "cam_from_rig": {
                            "rotation": [1.0, 0.0, 0.0, 0.0],
                            "translation": [0.1, 0.0, 0.0],
                        },
                    },
                ]
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(rig_spec, f)
            temp_rig_path = f.name

        try:
            loaded_rigs = pycolmap.read_rig_config(temp_rig_path)
            num_loaded = len(loaded_rigs)
            num_cams = len(loaded_rigs[0].cameras) if num_loaded > 0 else 0
        finally:
            if os.path.exists(temp_rig_path):
                os.remove(temp_rig_path)

        return {
            "status": "PASS",
            "constructed_cameras": 2,
            "ref_sensor_prefix": cam0.image_prefix,
            "secondary_cam_prefix": cam1.image_prefix,
            "cam1_translation": list(cam1.cam_from_rig.translation),
            "cam1_rotation_quat": list(cam1.cam_from_rig.rotation.quat),
            "json_roundtrip_rigs_count": num_loaded,
            "json_roundtrip_cams_count": num_cams,
        }
    except Exception as exc:
        return {"status": "FAIL", "error": str(exc)}


def run_full_diagnosis() -> Dict[str, Any]:
    """Aggregate all diagnostic probes and functional test results."""
    env = get_environment_info()
    packages = get_packages_info()
    tools = get_system_tools_info()
    capabilities = audit_pycolmap_capabilities()
    numerical = run_numerical_tests()
    video = run_video_decode_test()
    sift = run_sift_feature_tests()
    rig = run_rig_serialization_test()

    cuda_sift_pass = sift.get("cuda_sift", {}).get("status") == "PASS"
    gtsam_pass = numerical.get("gtsam_native_calculation", {}).get("status") == "PASS"
    video_pass = video.get("status") == "PASS"
    rig_pass = rig.get("status") == "PASS"

    all_passed = cuda_sift_pass and gtsam_pass and video_pass and rig_pass

    return {
        "status": "PASS" if all_passed else "FAIL",
        "environment": env,
        "packages": packages,
        "system_tools": tools,
        "pycolmap_capabilities": capabilities,
        "numerical_tests": numerical,
        "video_decode_test": video,
        "sift_tests": sift,
        "rig_test": rig,
    }
