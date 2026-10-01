"""Frames, perspective projection, masks, and synthetic fixtures package."""

from vscan.frames.masks import (
    MaskManager,
    create_hardware_nadir_mask,
    rasterize_polygon_mask,
    split_seam_crossing_polygon,
)
from vscan.frames.projection import (
    CUBEMAP_FACES,
    CUBEMAP_ROTATIONS,
    PinholeIntrinsics,
    build_virtual_camera_poses,
    crop_pinhole_intrinsics,
    get_cubemap_rotations,
    get_default_pinhole_intrinsics,
    matrix_to_quaternion_wxyz,
    pixel_to_ray_equirectangular,
    pixel_to_ray_pinhole,
    quaternion_wxyz_to_matrix,
    ray_to_pixel_equirectangular,
    ray_to_pixel_pinhole,
    scale_pinhole_intrinsics,
)
from vscan.frames.renderer import EquirectangularRenderer
from vscan.frames.rig import (
    build_colmap_rig_json,
    export_rig_config_json,
    reproduce_rays_from_serialized_rig,
    verify_rig_config_with_pycolmap,
    verify_rig_database_serialization,
)
from vscan.frames.scoring import (
    evaluate_panorama_quality,
    score_exposure,
    score_inter_frame_difference,
    score_sharpness,
)
from vscan.frames.selector import (
    VideoFrameDecoder,
    inspect_and_score_video_frames,
    select_contiguous_pilot_frames,
)
from vscan.frames.synthetic import (
    PlanarSurface,
    SyntheticRoom,
    generate_synthetic_rotation_only_fixture,
    generate_synthetic_translation_fixture,
)

__all__ = [
    "CUBEMAP_FACES",
    "CUBEMAP_ROTATIONS",
    "PinholeIntrinsics",
    "build_virtual_camera_poses",
    "get_cubemap_rotations",
    "get_default_pinhole_intrinsics",
    "scale_pinhole_intrinsics",
    "crop_pinhole_intrinsics",
    "pixel_to_ray_equirectangular",
    "ray_to_pixel_equirectangular",
    "pixel_to_ray_pinhole",
    "ray_to_pixel_pinhole",
    "matrix_to_quaternion_wxyz",
    "quaternion_wxyz_to_matrix",
    "EquirectangularRenderer",
    "MaskManager",
    "create_hardware_nadir_mask",
    "rasterize_polygon_mask",
    "split_seam_crossing_polygon",
    "score_sharpness",
    "score_exposure",
    "score_inter_frame_difference",
    "evaluate_panorama_quality",
    "VideoFrameDecoder",
    "inspect_and_score_video_frames",
    "select_contiguous_pilot_frames",
    "build_colmap_rig_json",
    "export_rig_config_json",
    "verify_rig_config_with_pycolmap",
    "verify_rig_database_serialization",
    "reproduce_rays_from_serialized_rig",
    "PlanarSurface",
    "SyntheticRoom",
    "generate_synthetic_translation_fixture",
    "generate_synthetic_rotation_only_fixture",
]
