import tempfile
from pathlib import Path
import pycolmap

from vscan.sfm.synthetic import generate_synthetic_sfm_dataset

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    dataset = generate_synthetic_sfm_dataset(p, num_stations=2)
    img_dir = dataset["images_dir"]
    images = list(img_dir.glob("*.jpg"))
    print("Generated images:", len(images), [im.name for im in images[:3]])

    db_path = p / "test.db"
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.use_gpu = False

    # Extract WITHOUT masks and WITHOUT import first
    pycolmap.extract_features(str(db_path), str(img_dir), extraction_options=extract_opts)

    with pycolmap.Database.open(str(db_path)) as db:
        print("Cameras:", db.num_cameras())
        print("Images:", db.num_images())
        for img in db.read_all_images():
            print(f"Direct extract: {img.name} -> {db.num_keypoints_for_image(img.image_id)} kps")
