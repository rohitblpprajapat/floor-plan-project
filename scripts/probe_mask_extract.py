import tempfile
from pathlib import Path
import pycolmap
from vscan.sfm.synthetic import generate_synthetic_sfm_dataset

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    dataset = generate_synthetic_sfm_dataset(p, num_stations=2)
    img_dir = dataset["images_dir"]
    masks_dir = dataset["masks_dir"]

    # Test 1: mask_path = masks_dir
    db_path = p / "test1.db"
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.mask_path = str(masks_dir)
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.use_gpu = False

    print("--- Extract with mask_path = masks_dir ---")
    pycolmap.extract_features(str(db_path), str(img_dir), reader_options=reader_opts, extraction_options=extract_opts)

    with pycolmap.Database.open(str(db_path)) as db:
        for img in db.read_all_images():
            print(f"  {img.name} -> {db.num_keypoints_for_image(img.image_id)} kps")
