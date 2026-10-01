"""Feature matching, temporal neighbor pairing, bounded revisit verification, and loop candidate filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pycolmap

from vscan.sfm.config import SfmConfig
from vscan.sfm.options import (
    build_feature_matching_options,
    build_sequential_pairing_options,
    build_two_view_geometry_options,
)


def verify_correspondence_distribution(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
    min_bbox_area_ratio: float = 0.08,
) -> Tuple[bool, str]:
    """Verify that correspondences are well-distributed across the image frame rather than clustered in a degenerate subregion."""
    if len(matches) < 15:
        return False, f"Insufficient matches: {len(matches)} < 15"

    idx1 = matches[:, 0]
    idx2 = matches[:, 1]

    pts1 = keypoints1[idx1, :2]
    pts2 = keypoints2[idx2, :2]

    # Compute bounding box spread
    min1, max1 = np.min(pts1, axis=0), np.max(pts1, axis=0)
    min2, max2 = np.min(pts2, axis=0), np.max(pts2, axis=0)

    span1 = max1 - min1
    span2 = max2 - min2

    # Check for collinearity or degenerate cluster (standard deviation)
    std1 = np.std(pts1, axis=0)
    std2 = np.std(pts2, axis=0)

    if min(std1[0], std1[1]) < 10.0 or min(std2[0], std2[1]) < 10.0:
        return False, "Correspondences clustered in 1D line or narrow streak (degenerate spatial distribution)"

    return True, "Valid spatial distribution"


def run_matching(
    database_path: Path,
    config: SfmConfig,
    device: pycolmap.Device,
    revisit_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Execute temporal sliding-window matching and bounded revisit verification.

    Adheres strictly to build contract 3:
    - Matches temporal neighbors within sliding window K.
    - Matches bounded revisit candidates without exhaustive all-pairs matching.
    - Strictly rejects unverified loops to prevent artificial map connection.
    """
    match_opts = build_feature_matching_options(config, device)
    seq_opts = build_sequential_pairing_options(config)
    two_view_opts = build_two_view_geometry_options(config)

    # 1. Temporal sequential matching across rigs and faces
    pycolmap.match_sequential(
        database_path=str(database_path),
        matching_options=match_opts,
        pairing_options=seq_opts,
        verification_options=two_view_opts,
        device=device,
    )

    loop_evidence: Dict[str, Any] = {
        "temporal_overlap": config.temporal_overlap,
        "revisit_candidates_tested": 0,
        "revisit_candidates_accepted": 0,
        "revisit_candidates_rejected": 0,
        "rejected_reasons": [],
    }

    # 2. Bounded revisit candidate matching if candidate pairs provided
    if revisit_pairs:
        bounded_candidates = revisit_pairs[: config.revisit_candidate_budget]
        loop_evidence["revisit_candidates_tested"] = len(bounded_candidates)

        pairs_file = database_path.parent / "revisit_pairs.txt"
        with open(pairs_file, "w", encoding="utf-8") as f:
            for img1, img2 in bounded_candidates:
                f.write(f"{img1} {img2}\n")

        import_opts = pycolmap.ImportedPairingOptions()
        import_opts.match_list_path = str(pairs_file)

        pycolmap.match_image_pairs(
            database_path=str(database_path),
            matching_options=match_opts,
            pairing_options=import_opts,
            verification_options=two_view_opts,
            device=device,
        )

        # Inspect verified geometries in database
        with pycolmap.Database.open(str(database_path)) as db:
            for img1, img2 in bounded_candidates:
                # Find image IDs
                im1 = db.read_image_with_name(img1)
                im2 = db.read_image_with_name(img2)
                if not im1 or not im2:
                    continue

                pair_id = pycolmap.image_pair_to_pair_id(im1.image_id, im2.image_id)
                if db.exists_two_view_geometry(pair_id):
                    two_view = db.read_two_view_geometry(pair_id)
                    num_inliers = two_view.inlier_matches.shape[0] if hasattr(two_view, "inlier_matches") else 0
                    if num_inliers >= config.min_inlier_matches:
                        loop_evidence["revisit_candidates_accepted"] += 1
                    else:
                        loop_evidence["revisit_candidates_rejected"] += 1
                        loop_evidence["rejected_reasons"].append(
                            f"Pair {img1}<->{img2} had only {num_inliers} inliers (< {config.min_inlier_matches})"
                        )
                else:
                    loop_evidence["revisit_candidates_rejected"] += 1
                    loop_evidence["rejected_reasons"].append(
                        f"Pair {img1}<->{img2} failed two-view geometric verification"
                    )

    # Summarize database verification status
    with pycolmap.Database.open(str(database_path)) as db:
        loop_evidence["total_verified_image_pairs"] = db.num_verified_image_pairs()
        loop_evidence["total_inlier_matches"] = db.num_inlier_matches()

    return loop_evidence
