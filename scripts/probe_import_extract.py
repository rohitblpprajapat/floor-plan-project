import tempfile
from pathlib import Path
import pycolmap
from PIL import Image

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    img_dir = p / "images"
    img_dir.mkdir()
    im = Image.new("RGB", (100, 100), color=(100, 100, 100))
    im.save(img_dir / "test1.jpg")

    db_path = p / "test.db"
    with pycolmap.Database.open(str(db_path)) as db:
        cam = pycolmap.Camera(model="PINHOLE", width=100, height=100, params=[50.0, 50.0, 50.0, 50.0], has_prior_focal_length=True)
        cid = db.write_camera(cam)
        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.existing_camera_id = cid
        pycolmap.import_images(str(db_path), str(img_dir), image_names=["test1.jpg"], options=reader_opts)

    print("After import_images, images in db:")
    with pycolmap.Database.open(str(db_path)) as db:
        for img in db.read_all_images():
            print(f"  Img: {img.image_id}, {img.name}")

    # Now extract_features without image_names
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.use_gpu = False
    pycolmap.extract_features(str(db_path), str(img_dir), extraction_options=extract_opts)

    with pycolmap.Database.open(str(db_path)) as db:
        for img in db.read_all_images():
            print(f"  After extract 1: {img.name} -> {db.num_keypoints_for_image(img.image_id)} kps")

    # Now extract_features WITH image_names
    pycolmap.extract_features(str(db_path), str(img_dir), image_names=["test1.jpg"], extraction_options=extract_opts)
    with pycolmap.Database.open(str(db_path)) as db:
        for img in db.read_all_images():
            print(f"  After extract 2: {img.name} -> {db.num_keypoints_for_image(img.image_id)} kps")
