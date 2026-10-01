"""Artifact management, hashing, cache key calculation, and atomic completion markers.

Guarantees:
- Atomic completion markers written via temporary file replacement.
- Output validation verifies existence, file size, and sha256 checksums.
- Corrupt markers or corrupted outputs are safely rejected without deleting raw data.
- Deterministic cache key generation based on stage, config hash, input hashes, and calibration.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from vscan.schemas.jobs import CompletionMarkerSchema


def compute_sha256_file(file_path: Path, block_size: int = 65536) -> str:
    """Compute sha256 hash of a file efficiently in 64KB blocks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """Compute sha256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_sha256_dict(data: Dict[str, Any]) -> str:
    """Compute deterministic sha256 hash of a dictionary."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return compute_sha256_bytes(encoded)


def compute_cache_key(
    stage: str,
    config_hash: str,
    input_hashes: Dict[str, str],
    calibration_hash: str = "",
    pose_hash: str = "",
) -> str:
    """Compute a deterministic cache key for a stage execution.

    Any alteration to stage, config, inputs, calibration, or prior poses
    produces a completely distinct cache key.
    """
    sorted_inputs = sorted(input_hashes.items())
    payload = {
        "stage": stage,
        "config_hash": config_hash,
        "inputs": sorted_inputs,
        "calibration_hash": calibration_hash,
        "pose_hash": pose_hash,
    }
    return compute_sha256_dict(payload)


def write_atomic_completion_marker(
    marker: CompletionMarkerSchema,
    outputs_dir: Path,
) -> Path:
    """Write an atomic completion marker inside outputs_dir/.complete.

    Uses a temporary file in the same directory followed by os.replace
    to guarantee atomic creation.
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)
    target_path = outputs_dir / ".complete"
    tmp_path = outputs_dir / f".complete.tmp.{os.getpid()}"

    marker_dict = marker.model_dump(mode="json")
    # Compute marker self-hash without marker_sha256 field
    marker_dict["marker_sha256"] = None
    content_bytes = json.dumps(marker_dict, indent=2, sort_keys=True).encode("utf-8")
    marker_dict["marker_sha256"] = compute_sha256_bytes(content_bytes)

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(marker_dict, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    # Atomic rename/replace
    os.replace(tmp_path, target_path)
    return target_path


def verify_completion_marker(
    outputs_dir: Path,
    expected_cache_key: Optional[str] = None,
) -> Tuple[bool, str, Optional[CompletionMarkerSchema]]:
    """Verify that an output directory has a valid, intact atomic completion marker.

    Validates:
    1. Existence of .complete
    2. JSON structure matches CompletionMarkerSchema
    3. Cache key matches expected_cache_key (if provided)
    4. Every output file exists, matches expected size, and matches sha256 checksum.

    Returns:
        (is_valid, message, marker_schema)
    """
    marker_path = outputs_dir / ".complete"
    if not marker_path.is_file():
        return False, f"Missing completion marker at {marker_path}", None

    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        marker = CompletionMarkerSchema.model_validate(data)
    except Exception as exc:
        return False, f"Corrupted completion marker JSON: {exc}", None

    if expected_cache_key and marker.cache_key != expected_cache_key:
        return False, f"Cache key mismatch: got {marker.cache_key}, expected {expected_cache_key}", marker

    # Verify each output file
    for rel_path, expected_sha in marker.output_checksums.items():
        file_path = outputs_dir / rel_path
        if not file_path.is_file():
            return False, f"Missing output file: {rel_path}", marker

        # Check size if available
        expected_size = marker.output_sizes_bytes.get(rel_path)
        if expected_size is not None:
            actual_size = file_path.stat().st_size
            if actual_size != expected_size:
                return (
                    False,
                    f"Size mismatch for {rel_path}: actual {actual_size} != expected {expected_size}",
                    marker,
                )

        # Check sha256
        actual_sha = compute_sha256_file(file_path)
        if actual_sha != expected_sha:
            return (
                False,
                f"Checksum corruption for {rel_path}: actual {actual_sha} != expected {expected_sha}",
                marker,
            )

    return True, "Valid completion marker and verified output checksums", marker
