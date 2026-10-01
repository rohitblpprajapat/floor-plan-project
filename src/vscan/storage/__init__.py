"""Storage layer: database and artifact management."""

from vscan.storage.database import DatabaseManager, DEFAULT_DB_PATH
from vscan.storage.artifacts import (
    compute_sha256_file,
    compute_sha256_bytes,
    compute_sha256_dict,
    compute_cache_key,
    write_atomic_completion_marker,
    verify_completion_marker,
)

__all__ = [
    "DatabaseManager",
    "DEFAULT_DB_PATH",
    "compute_sha256_file",
    "compute_sha256_bytes",
    "compute_sha256_dict",
    "compute_cache_key",
    "write_atomic_completion_marker",
    "verify_completion_marker",
]
