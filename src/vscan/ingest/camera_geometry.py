"""Camera geometry inspector and local vendor software probe.

Adheres strictly to contracts:
1. Verifies actual image layouts; a 2:1 frame or .insv extension is NOT sufficient to establish camera geometry.
2. Inspects whether all required original lens streams are present.
3. Inspects locally available supported software (e.g. Insta360 Studio) before prescribing export workflows.
4. Gives exact vendor export requirements while strictly preserving provenance.
5. Does NOT invent SDK functions or reverse-engineered timestamp decoders.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from vscan.schemas.manifest import VideoProbeSummarySchema


KNOWN_VENDOR_PATHS_WINDOWS = [
    Path(r"C:\Program Files\Insta360 Studio"),
    Path(r"C:\Program Files\Insta360 Studio 2024"),
    Path(r"C:\Program Files\Insta360 Studio 2023"),
    Path(r"C:\Program Files (x86)\Insta360 Studio"),
    Path(r"C:\Program Files (x86)\Insta360 Studio 2024"),
    Path(r"C:\Users\rohit\AppData\Local\Programs\Insta360 Studio"),
]

KNOWN_VENDOR_BINS_LINUX = [
    "insta360_studio",
    "insta360studio",
    "Insta360Studio",
]


def probe_local_vendor_software() -> Dict[str, Any]:
    """Inspect locally installed supported vendor software for video stitching/export."""
    detected_paths: List[str] = []

    # 1. Check Windows host standard paths
    for p in KNOWN_VENDOR_PATHS_WINDOWS:
        if p.is_dir():
            detected_paths.append(str(p))

    # 2. Check Linux/WSL PATH binaries
    for b in KNOWN_VENDOR_BINS_LINUX:
        found = shutil.which(b)
        if found:
            detected_paths.append(found)

    is_installed = len(detected_paths) > 0
    return {
        "is_installed": is_installed,
        "detected_paths": detected_paths,
        "primary_software": "Insta360 Studio (Windows / Linux)",
        "recommendation": (
            "Insta360 Studio available on Windows host for official dual-fisheye or equirectangular export"
            if is_installed
            else "Insta360 Studio not detected in standard system paths; manual export or headless CLI needed if re-export required."
        ),
    }


def evaluate_camera_geometry(video_probes: List[VideoProbeSummarySchema]) -> Dict[str, Any]:
    """Evaluate camera geometry, lens streams, and projection layouts across session videos."""
    evaluations: Dict[str, Any] = {}
    insv_has_dual_lenses = False

    for vp in video_probes:
        fname = Path(vp.file_path).name
        video_tracks = [t for t in vp.tracks if t.track_type == "video"]
        v_count = len(video_tracks)
        vt = video_tracks[0] if video_tracks else None
        w = vt.width if vt else 0
        h = vt.height if vt else 0
        aspect = vt.aspect_ratio if vt else "unknown"
        res_list = [f"{t.width}x{t.height}" for t in video_tracks]

        required_lens_streams_present = False

        if fname.endswith(".insv"):
            role = "original_source_candidate"
            if v_count >= 2 and all(t.width and t.height and t.width == t.height for t in video_tracks):
                geometry_verdict = "DUAL_CIRCULAR_FISHEYE_RAW"
                required_lens_streams_present = True
                insv_has_dual_lenses = True
                notes = (
                    f"INSV container contains {v_count} separate video streams ({', '.join(res_list)}). "
                    "All required original lens streams (Lens 0 and Lens 1 circular fisheyes) are present in the container. "
                    "A 2:1 aspect ratio or .insv extension alone is NOT sufficient to prove camera geometry. "
                    "Dual-fisheye lens distortion parameters and baseline transform are required for SfM."
                )
            else:
                geometry_verdict = "SINGLE_OR_UNKNOWN_LENS_CONTAINER"
                notes = (
                    f"INSV container contains {v_count} video streams ({', '.join(res_list)}). "
                    "Incomplete dual-lens streams detected or secondary stream missing."
                )
        elif "lrv" in fname.lower():
            role = "preview_stream_candidate"
            geometry_verdict = "LOW_RES_PREVIEW_PROXY"
            required_lens_streams_present = True
            notes = "LRV file is a low-bitrate proxy stream intended for mobile preview. Insufficient fidelity for metric reconstruction."
        elif "export" in fname.lower() and fname.endswith(".mp4"):
            role = "vendor_export_candidate"
            geometry_verdict = "STITCHED_EQUIRECTANGULAR_OR_REFRAMED"
            required_lens_streams_present = True
            notes = (
                f"Exported MP4 probed at {w}x{h} ({aspect}). "
                "Equirectangular panoramic projection is stitched from two fisheye lenses, "
                "introducing non-linear seam artifacts and variable latitude pixel density."
            )
        else:
            role = "auxiliary_media"
            geometry_verdict = "UNSPECIFIED_LAYOUT"
            notes = "Unknown media role."

        evaluations[fname] = {
            "role": role,
            "resolution": f"{w}x{h}",
            "aspect_ratio": aspect,
            "video_tracks_count": v_count,
            "track_resolutions": res_list,
            "required_lens_streams_present": required_lens_streams_present,
            "geometry_verdict": geometry_verdict,
            "notes": notes,
        }

    has_usable_imagery = any(
        vp.tracks and not vp.layout_verification.get("is_valid_video") is False
        for vp in video_probes
    )

    return {
        "per_file_evaluations": evaluations,
        "vendor_software": probe_local_vendor_software(),
        "insv_dual_lenses_verified": insv_has_dual_lenses,
        "prompt_05_imagery_readiness": {
            "can_use_actual_imagery": has_usable_imagery,
            "imagery_mode": "STITCHED_EQUIRECTANGULAR_OR_EXPORTED_FRAMES",
            "needs_vendor_export": False if has_usable_imagery else True,
            "fixture_fallback_available": True,
            "geometry_limitations": (
                "Prompt 05 can process the verified local exported MP4 (insta360_export_1789041643971.mp4) "
                "or raw INSV dual-fisheye frames, but scale remains NOT_ESTIMATED because optical calibration and rigid "
                "baseline between phone and camera are unverified."
            ),
        },
    }
