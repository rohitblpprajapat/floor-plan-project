#!/usr/bin/env bash
set -euo pipefail

SRC="/mnt/d/Documents/GitHub/Simhatel/floor plan project"
DST="/home/rohit/projects/insta360_mapper"

mkdir -p "${DST}/src" "${DST}/tests" "${DST}/configs" "${DST}/docs" "${DST}/reports" "${DST}/scripts" "${DST}/data/sessions" "${DST}/runs"

rsync -av "${SRC}/src/" "${DST}/src/"
rsync -av "${SRC}/tests/" "${DST}/tests/"
rsync -av "${SRC}/configs/" "${DST}/configs/"
rsync -av "${SRC}/docs/" "${DST}/docs/"
rsync -av "${SRC}/reports/" "${DST}/reports/"
rsync -av "${SRC}/scripts/" "${DST}/scripts/"
cp "${SRC}/pyproject.toml" "${DST}/pyproject.toml"
cp "${SRC}/.gitignore" "${DST}/.gitignore"

echo "[SYNC] Synced code, tests, configs, docs, and scripts to WSL."
