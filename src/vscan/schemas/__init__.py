"""vscan data schemas and contracts."""

from vscan.schemas.common import (
    CoordinateFrame,
    PoseMatrix,
    SCHEMA_VERSION,
    Units,
)
from vscan.schemas.assets import (
    ClockMappingSchema,
    InputAssetSchema,
    SourceFrameSchema,
)
from vscan.schemas.cameras import (
    CalibrationSchema,
    CameraModel,
    CameraRigSchema,
)
from vscan.schemas.poses import (
    LandmarkSchema,
    MapVersionSchema,
    PoseSchema,
)
from vscan.schemas.surfaces import (
    AnnotationObservation,
    AnnotationSchema,
    DepthMapSchema,
    MeasurementSchema,
    MeshSchema,
    SurfaceSchema,
)
from vscan.schemas.jobs import (
    CompletionMarkerSchema,
    ErrorClassification,
    JobProgressSchema,
    JobRecordSchema,
    JobStatus,
)
from vscan.schemas.manifest import (
    InputManifestSchema,
    LocationMetadataSchema,
    SensorSummarySchema,
    SourceDerivedRelationshipSchema,
    ValidityFlagSchema,
    VideoProbeSummarySchema,
    VideoTrackProbeSchema,
)
from vscan.schemas.sync import (
    ClockCalibrationSchema,
    DiagnosticsSchema,
    ExtrinsicCalibrationSchema,
    SyncManifestSchema,
    TimingBoundaryDiagnosticSchema,
    UnitHypothesisSchema,
)

__all__ = [
    "SCHEMA_VERSION",
    "Units",
    "CoordinateFrame",
    "PoseMatrix",
    "InputAssetSchema",
    "SourceFrameSchema",
    "ClockMappingSchema",
    "CameraModel",
    "CameraRigSchema",
    "CalibrationSchema",
    "PoseSchema",
    "LandmarkSchema",
    "MapVersionSchema",
    "SurfaceSchema",
    "DepthMapSchema",
    "MeshSchema",
    "AnnotationObservation",
    "AnnotationSchema",
    "MeasurementSchema",
    "JobStatus",
    "ErrorClassification",
    "CompletionMarkerSchema",
    "JobProgressSchema",
    "JobRecordSchema",
    "InputManifestSchema",
    "LocationMetadataSchema",
    "SensorSummarySchema",
    "SourceDerivedRelationshipSchema",
    "ValidityFlagSchema",
    "VideoProbeSummarySchema",
    "VideoTrackProbeSchema",
    "ClockCalibrationSchema",
    "ExtrinsicCalibrationSchema",
    "SyncManifestSchema",
    "TimingBoundaryDiagnosticSchema",
    "UnitHypothesisSchema",
    "DiagnosticsSchema",
]
