"""
Generates production StellaVSLAM configuration for 360 equirectangular camera.
Conforms to StellaVSLAM equirectangular model specification (Section 7.1).
"""

import os
import yaml
from typing import Dict, Any

def generate_stella_equirectangular_config(
    cols: int = 1664,
    rows: int = 832,
    fps: float = 25.0,
    max_num_keypoints: int = 2000,
    output_path: str = "configs/stella_equirectangular.yaml"
) -> str:
    """
    Creates equirectangular YAML configuration file for StellaVSLAM.
    """
    config: Dict[str, Any] = {
        "Camera": {
            "name": "360_Equirectangular_Camera",
            "setup": "monocular",
            "model": "equirectangular",
            "color_order": "BGR",
            "cols": int(cols),
            "rows": int(rows),
            "fps": float(fps)
        },
        "Feature": {
            "max_num_keypoints": int(max_num_keypoints),
            "scale_factor": 1.2,
            "num_levels": 8,
            "ini_fast_threshold": 20,
            "min_fast_threshold": 7
        },
        "Mapping": {
            "baseline_dist_thr_ratio": 0.02,
            "redundant_obs_ratio_thr": 0.9
        },
        "Tracking": {
            "min_num_fnd_pts": 30
        },
        "Initializer": {
            "num_ransac_iters": 100,
            "min_num_triangulated": 50
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return os.path.abspath(output_path)

if __name__ == "__main__":
    path = generate_stella_equirectangular_config()
    print(f"Generated StellaVSLAM equirectangular config at: {path}")
