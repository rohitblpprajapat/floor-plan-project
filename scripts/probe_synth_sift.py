import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pycolmap
from vscan.frames.synthetic import SyntheticRoom
from vscan.frames.projection import get_default_pinhole_intrinsics, CUBEMAP_ROTATIONS, CUBEMAP_FACES

room = SyntheticRoom()
intrinsics = get_default_pinhole_intrinsics(320, 320, 90.0)

# Render face_front from center [0, 0, 0]
rgb, depth = room.render_analytic_pinhole(np.array([0.0, 0.0, 0.0]), CUBEMAP_ROTATIONS[CUBEMAP_FACES[0]], intrinsics)

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    img_dir = p / "images"
    img_dir.mkdir()
    Image.fromarray(rgb).save(img_dir / "face_front_000.jpg")

    db_path = p / "test.db"
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.use_gpu = False

    pycolmap.extract_features(str(db_path), str(img_dir), extraction_options=extract_opts)
    with pycolmap.Database.open(str(db_path)) as db:
        kps = db.read_keypoints(1)
        print("Keypoints extracted on synthetic face_front:", len(kps))
