#!/usr/bin/env bash
set -e

# Sync all files from Windows workspace to WSL
rsync -av --exclude='.venv' --exclude='.pytest_cache' --exclude='__pycache__' \
  "/mnt/d/Documents/GitHub/Simhatel/floor plan project/" \
  /home/rohit/projects/insta360_mapper/

# Run pytest on tests/test_sfm.py
cd /home/rohit/projects/insta360_mapper
source .venv/bin/activate
pytest tests/test_sfm.py -v
