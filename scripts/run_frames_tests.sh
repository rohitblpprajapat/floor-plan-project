#!/usr/bin/env bash
set -e
cd /home/rohit/projects/insta360_mapper
source .venv/bin/activate
echo "=== Testing imports ==="
python -c "
from vscan.frames.projection import CUBEMAP_FACES
from vscan.frames.renderer import EquirectangularRenderer
from vscan.frames.masks import MaskManager
from vscan.frames.rig import export_rig_config_json
from vscan.frames.synthetic import SyntheticRoom
from vscan.frames.scoring import evaluate_panorama_quality
from vscan.frames.selector import VideoFrameDecoder
from vscan.workers.frames import FramesWorker
from vscan.schemas.frames import PilotManifestSchema
print('All frames imports OK')
"
echo "=== Running frames tests ==="
python -m pytest tests/test_frames.py -v --tb=short 2>&1 | head -200
echo "=== Done ==="
