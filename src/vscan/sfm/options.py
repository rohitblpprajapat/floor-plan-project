"""PyCOLMAP 4.2.0 option builders with explicit parameter logging and constraint enforcement."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import pycolmap

from vscan.sfm.config import SfmConfig


def resolve_device(config: SfmConfig) -> Tuple[pycolmap.Device, Dict[str, Any]]:
    """Resolve and validate requested execution device without silent downgrades."""
    requested = config.device_requested.lower().strip()
    gpu_info: Dict[str, Any] = {}

    if requested == "cuda":
        if not pycolmap.has_cuda:
            if config.allow_cpu_fallback:
                return pycolmap.Device.cpu, {"fallback": True, "reason": "CUDA not supported by pycolmap binary"}
            raise RuntimeError(
                "Requested CUDA device for SfM feature extraction, but pycolmap.has_cuda is False. "
                "Silent CPU fallback is prohibited by contract. Set allow_cpu_fallback=True to test CPU fallback explicitly."
            )
        num_gpus = pycolmap.get_num_cuda_devices()
        if num_gpus <= 0:
            if config.allow_cpu_fallback:
                return pycolmap.Device.cpu, {"fallback": True, "reason": "No CUDA devices detected"}
            raise RuntimeError(
                "Requested CUDA device for SfM feature extraction, but 0 CUDA devices were detected. "
                "Silent CPU fallback is prohibited by contract."
            )
        gpu_info = {
            "device_index": config.gpu_index,
            "total_cuda_devices": num_gpus,
            "has_cuda": True,
        }
        return pycolmap.Device.cuda, gpu_info
    elif requested == "cpu":
        return pycolmap.Device.cpu, {"mode": "explicit_cpu"}
    else:
        raise ValueError(f"Unknown device requested: '{config.device_requested}'. Must be 'cuda' or 'cpu'.")


def build_feature_extraction_options(config: SfmConfig, device: pycolmap.Device) -> pycolmap.FeatureExtractionOptions:
    """Build PyCOLMAP FeatureExtractionOptions with pinned option names."""
    opts = pycolmap.FeatureExtractionOptions()
    opts.num_threads = config.num_threads
    opts.use_gpu = (device == pycolmap.Device.cuda)
    opts.gpu_index = str(config.gpu_index)

    # SIFT options
    opts.sift.max_num_features = config.max_num_features
    opts.sift.peak_threshold = config.peak_threshold
    opts.sift.edge_threshold = config.edge_threshold

    return opts


def build_feature_matching_options(config: SfmConfig, device: pycolmap.Device) -> pycolmap.FeatureMatchingOptions:
    """Build PyCOLMAP FeatureMatchingOptions with pinned option names."""
    opts = pycolmap.FeatureMatchingOptions()
    opts.num_threads = config.num_threads
    opts.use_gpu = (device == pycolmap.Device.cuda)
    opts.gpu_index = str(config.gpu_index)
    opts.rig_verification = True

    return opts


def build_sequential_pairing_options(config: SfmConfig) -> pycolmap.SequentialPairingOptions:
    """Build PyCOLMAP SequentialPairingOptions for temporal sliding-window matching."""
    opts = pycolmap.SequentialPairingOptions()
    opts.overlap = config.temporal_overlap
    opts.quadratic_overlap = config.quadratic_overlap
    opts.expand_rig_images = True
    opts.num_threads = config.num_threads
    opts.loop_detection = config.enable_loop_detection

    return opts


def build_two_view_geometry_options(config: SfmConfig) -> pycolmap.TwoViewGeometryOptions:
    """Build PyCOLMAP TwoViewGeometryOptions for geometric verification."""
    opts = pycolmap.TwoViewGeometryOptions()
    opts.min_num_inliers = config.min_inlier_matches
    opts.ransac.max_error = config.max_error_px

    return opts


def build_incremental_pipeline_options(config: SfmConfig) -> pycolmap.IncrementalPipelineOptions:
    """Build PyCOLMAP IncrementalPipelineOptions keeping intrinsics and rig extrinsics fixed."""
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = config.num_threads
    opts.multiple_models = config.multiple_models
    opts.min_model_size = config.min_model_size

    # Strict constraint enforcement: keep known virtual-view intrinsics fixed
    opts.ba_refine_focal_length = not config.fix_intrinsics
    opts.ba_refine_principal_point = not config.fix_intrinsics
    opts.ba_refine_extra_params = not config.fix_intrinsics

    # Strict constraint enforcement: keep known relative rig extrinsics fixed
    opts.ba_refine_sensor_from_rig = not config.fix_rig_extrinsics

    # Mapper sub-options
    opts.mapper.abs_pose_refine_focal_length = not config.fix_intrinsics
    opts.mapper.abs_pose_refine_extra_params = not config.fix_intrinsics
    opts.mapper.init_min_tri_angle = config.init_min_tri_angle
    opts.mapper.init_min_num_inliers = config.init_min_num_inliers
    opts.mapper.abs_pose_min_num_inliers = config.abs_pose_min_num_inliers
    opts.mapper.filter_max_reproj_error = config.filter_max_reproj_error

    return opts


def log_effective_options(config: SfmConfig, device: pycolmap.Device) -> Dict[str, Any]:
    """Produce dictionary of effective options for immutable audit trail."""
    return {
        "device": device.name,
        "num_threads": config.num_threads,
        "max_num_features": config.max_num_features,
        "peak_threshold": config.peak_threshold,
        "edge_threshold": config.edge_threshold,
        "use_masks": config.use_masks,
        "temporal_overlap": config.temporal_overlap,
        "quadratic_overlap": config.quadratic_overlap,
        "enable_loop_detection": config.enable_loop_detection,
        "fix_intrinsics": config.fix_intrinsics,
        "fix_rig_extrinsics": config.fix_rig_extrinsics,
        "init_min_tri_angle": config.init_min_tri_angle,
        "init_min_num_inliers": config.init_min_num_inliers,
        "abs_pose_min_num_inliers": config.abs_pose_min_num_inliers,
        "filter_max_reproj_error": config.filter_max_reproj_error,
        "min_model_size": config.min_model_size,
        "multiple_models": config.multiple_models,
        "scale_status": "NOT_ESTIMATED",
    }
