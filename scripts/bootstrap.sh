#!/usr/bin/env bash
# bootstrap.sh - Initialize and verify the vscan Python 3.11 environment in WSL2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
REQUIRED_PYTHON="3.11.16"

echo "=== vscan Environment Bootstrap ==="
echo "Project Root: ${PROJECT_ROOT}"

# Set CPU thread limits based on Prompt 01 measurements (6 physical cores, 12 logical threads)
# Preserve Windows host responsiveness (3.5 GiB WSL RAM budget)
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
echo "[INFO] Set CPU worker thread limits: 6"

# Check or install uv
if ! command -v uv &>/dev/null; then
    if [ -f "${HOME}/.local/bin/uv" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        echo "[INFO] Installing uv dependency manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
fi

# Check or install Python 3.11.16 via uv
if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] Creating isolated Python ${REQUIRED_PYTHON} virtual environment..."
    uv python install "${REQUIRED_PYTHON}"
    uv venv "${VENV_DIR}" --python "${REQUIRED_PYTHON}"
else
    echo "[INFO] Existing virtual environment detected at ${VENV_DIR}. Preserving environment."
fi

# Activate virtual environment
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Export CUDA runtime shared library paths for pycolmap dynamic linker
NVIDIA_CUDA_LIB="${VENV_DIR}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
NVIDIA_CURAND_LIB="${VENV_DIR}/lib/python3.11/site-packages/nvidia/curand/lib"
export LD_LIBRARY_PATH="${NVIDIA_CUDA_LIB}:${NVIDIA_CURAND_LIB}:${LD_LIBRARY_PATH:-}"

# Check Python version
INSTALLED_PY="$(python -c 'import sys; print(sys.version.split()[0])')"
echo "[INFO] Python version in venv: ${INSTALLED_PY}"
if [ "${INSTALLED_PY}" != "${REQUIRED_PYTHON}" ]; then
    echo "[ERROR] Python version mismatch! Expected ${REQUIRED_PYTHON}, found ${INSTALLED_PY}"
    exit 1
fi

# Install dependencies if not already installed (idempotent lockfile sync)
if ! python -c "import pycolmap, gtsam, vscan" &>/dev/null; then
    echo "[INFO] Installing dependencies and editable vscan package..."
    uv pip install -e "${PROJECT_ROOT}"
else
    echo "[INFO] Core packages (pycolmap, gtsam, vscan) already verified."
fi

# Run vscan doctor verification
echo "[INFO] Running vscan doctor diagnostic..."
python -m vscan.cli doctor

echo "=== Bootstrap Completed Successfully ==="
