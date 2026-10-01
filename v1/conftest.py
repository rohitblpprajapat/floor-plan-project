import os
import sys
from pathlib import Path

# Add v1 directory to python path for all tests
V1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = V1_DIR.parent
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

def get_data_file(filename: str) -> str:
    """Finds test data files in data/ directories or project root."""
    candidates = [
        PROJECT_ROOT / "data" / "2026-08-31" / filename,
        PROJECT_ROOT / "data" / "2026-09-10" / filename,
        PROJECT_ROOT / filename,
        V1_DIR / filename
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return filename

