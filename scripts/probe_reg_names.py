import tempfile
from pathlib import Path
import pycolmap
from vscan.sfm.config import SfmConfig
from vscan.sfm.pipeline import SfmPipeline

with tempfile.TemporaryDirectory() as td:
    out_dir = Path(td) / "sfm_out"
    cfg = SfmConfig(
        session_id="synth_test_session",
        run_id="run_sfm_test_001",
        output_dir=out_dir,
        is_synthetic=True,
        synthetic_stations_count=5,
        device_requested="cpu",
        num_threads=4,
    )
    pipeline = SfmPipeline(cfg)
    result = pipeline.run()
    recon = result["reconstructions"][0]
    print("Registered image IDs:", recon.reg_image_ids())
    for iid in recon.reg_image_ids():
        im = recon.image(iid)
        print(f"  ID {iid}: name='{im.name}', has_pose={im.has_pose}")
