#!/usr/bin/env bash
# Setup script for Ubuntu 26.04 / WSL2 environment
# Installs GTSAM, Ceres, OpenCV, CMake, and Python scientific stack.

set -e

echo "=== Updating package lists ==="
sudo apt-get update

echo "=== Installing core build tools & libraries ==="
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libopencv-dev \
    libceres-dev \
    libgtsam-dev \
    libyaml-cpp-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-pandas \
    python3-matplotlib \
    python3-yaml \
    python3-tqdm \
    python3-pytest

echo "=== Setup completed successfully ==="
