"""Structure from Motion (SfM) package wrapping PyCOLMAP 4.2.0."""

from vscan.sfm.config import SfmConfig
from vscan.sfm.export import export_reconstruction_component, parse_image_name
from vscan.sfm.matching import run_matching, verify_correspondence_distribution
from vscan.sfm.options import (
    build_feature_extraction_options,
    build_feature_matching_options,
    build_incremental_pipeline_options,
    build_sequential_pairing_options,
    build_two_view_geometry_options,
    log_effective_options,
    resolve_device,
)
from vscan.sfm.pipeline import SfmPipeline
from vscan.sfm.synthetic import (
    align_sim3_umeyama,
    generate_synthetic_sfm_dataset,
    render_pinhole_vectorized,
)

__all__ = [
    "SfmConfig",
    "SfmPipeline",
    "resolve_device",
    "log_effective_options",
    "build_feature_extraction_options",
    "build_feature_matching_options",
    "build_sequential_pairing_options",
    "build_two_view_geometry_options",
    "build_incremental_pipeline_options",
    "run_matching",
    "verify_correspondence_distribution",
    "export_reconstruction_component",
    "parse_image_name",
    "render_pinhole_vectorized",
    "generate_synthetic_sfm_dataset",
    "align_sim3_umeyama",
]
