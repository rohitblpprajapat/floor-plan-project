"""Worker registry and pipeline stage dispatch."""

from typing import Dict, Optional, Type
from vscan.workers.base import BaseWorker
from vscan.workers.smoke import SmokeWorker
from vscan.workers.gpu_smoke import GPUSmokeWorker
from vscan.workers.failure import FailureInjectionWorker
from vscan.workers.ingest import IngestWorker
from vscan.workers.frames import FramesWorker
try:
    from vscan.workers.sfm import SfmWorker
except ImportError:
    SfmWorker = None

from vscan.workers.sync import SyncWorker

WORKER_REGISTRY: Dict[str, Optional[Type[BaseWorker]]] = {
    "smoke": SmokeWorker,
    "gpu_smoke": GPUSmokeWorker,
    "fail_test": FailureInjectionWorker,
    "ingest": IngestWorker,
    "frames": FramesWorker,
    "unpack": FramesWorker,
    "sfm": SfmWorker,
    "sync": SyncWorker,
}


def get_worker_class(stage_name: str) -> Type[BaseWorker]:
    """Retrieve worker implementation for given stage."""
    if stage_name not in WORKER_REGISTRY:
        raise ValueError(f"Unknown stage '{stage_name}'. Available stages: {list(WORKER_REGISTRY.keys())}")
    worker_cls = WORKER_REGISTRY[stage_name]
    if worker_cls is None:
        raise ImportError(f"Worker class for stage '{stage_name}' could not be imported (missing optional native dependency, e.g. pycolmap).")
    return worker_cls


__all__ = [
    "BaseWorker",
    "SmokeWorker",
    "GPUSmokeWorker",
    "FailureInjectionWorker",
    "IngestWorker",
    "FramesWorker",
    "SfmWorker",
    "SyncWorker",
    "WORKER_REGISTRY",
    "get_worker_class",
]

