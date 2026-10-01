#!/usr/bin/env bash
set -e

# Sync all files from Windows workspace to WSL
rsync -av --exclude='.venv' --exclude='.pytest_cache' --exclude='__pycache__' \
  "/mnt/d/Documents/GitHub/Simhatel/floor plan project/" \
  /home/rohit/projects/insta360_mapper/

# Run the pilot sfm script
/home/rohit/projects/insta360_mapper/.venv/bin/python /home/rohit/projects/insta360_mapper/scripts/test_pilot_sfm.py
