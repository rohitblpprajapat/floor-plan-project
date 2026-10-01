"""Base worker class and interface for pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import time

from vscan.schemas.jobs import CompletionMarkerSchema, JobProgressSchema
from vscan.storage.artifacts import (
    compute_cache_key,
    compute_sha256_file,
    write_atomic_completion_marker,
)


class BaseWorker(ABC):
    """Abstract base class for all vscan stage workers."""

    stage_name: str = "base"
    requires_gpu: bool = False

    def __init__(
        self,
        run_id: str,
        session_id: str,
        config: Dict[str, Any],
        run_dir: Path,
        progress_callback: Optional[Callable[[JobProgressSchema], None]] = None,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self.config = config
        self.run_dir = Path(run_dir)
        self.outputs_dir = self.run_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback
        self.cancel_checker = cancel_checker
        self.start_time = time.time()

    def is_cancellation_requested(self) -> bool:
        """Check if cancellation was requested cooperatively."""
        cancel_file = self.run_dir / "cancel_requested"
        if cancel_file.exists():
            return True
        if self.cancel_checker and self.cancel_checker():
            return True
        return False

    def report_progress(
        self,
        progress_percent: float,
        current_step: str,
        items_completed: int = 0,
        items_total: int = 0,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a progress update."""
        elapsed = time.time() - self.start_time
        eta = None
        if progress_percent > 0.0:
            eta = max(0.0, (elapsed / (progress_percent / 100.0)) - elapsed)

        schema = JobProgressSchema(
            run_id=self.run_id,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            progress_percent=min(100.0, max(0.0, progress_percent)),
            current_step=current_step,
            items_completed=items_completed,
            items_total=items_total,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            telemetry=telemetry or {},
        )
        if self.progress_callback:
            self.progress_callback(schema)

    def write_completion_marker(
        self,
        produced_outputs: Dict[str, str],
        cache_key: str,
    ) -> CompletionMarkerSchema:
        """Compute sha256 checksums and write atomic completion marker."""
        output_checksums: Dict[str, str] = {}
        output_sizes: Dict[str, int] = {}

        for logical_name, rel_path in produced_outputs.items():
            full_path = self.outputs_dir / rel_path
            if not full_path.is_file():
                raise FileNotFoundError(f"Worker declared output {logical_name} at {rel_path}, but file is missing!")
            output_checksums[rel_path] = compute_sha256_file(full_path)
            output_sizes[rel_path] = full_path.stat().st_size

        marker = CompletionMarkerSchema(
            run_id=self.run_id,
            stage=self.stage_name,
            session_id=self.session_id,
            completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cache_key=cache_key,
            output_checksums=output_checksums,
            output_sizes_bytes=output_sizes,
        )
        write_atomic_completion_marker(marker, self.outputs_dir)
        return marker

    @abstractmethod
    def execute(self) -> Dict[str, str]:
        """Execute worker logic.

        Must return dictionary of {logical_name: relative_output_path}.
        Raises exceptions on failure.
        """
        raise NotImplementedError
