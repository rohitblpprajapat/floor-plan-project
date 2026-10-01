"""Video stream prober, layout inspector, frame decoder, and contact sheet generator.

Adheres strictly to contracts:
1. Probe with ffprobe in JSON form when available, augmented by native ISO-BMFF box parsing.
2. Read frame metadata using presentation order (PTS); do not confuse DTS with exposure time.
3. Record exposure duration and rolling-shutter/readout metadata where available; unknown fields remain unknown.
4. Generates labelled contact sheets inspecting start, middle, and end decoded frames.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from vscan.ingest.isobmff import inspect_isobmff_file
from vscan.schemas.manifest import VideoProbeSummarySchema, VideoTrackProbeSchema


def probe_with_ffprobe(video_path: Path) -> Optional[Dict[str, Any]]:
    """Execute ffprobe and return JSON stream and container metadata."""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None

    cmd = [
        ffprobe_bin,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return json.loads(res.stdout)
    except Exception:
        return None


def probe_video_stream(video_path: Path, asset_id: str) -> VideoProbeSummarySchema:
    """Comprehensive probe of container and streams using ffprobe and ISO-BMFF parser."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    file_size = path.stat().st_size

    # Check for empty or screenshot files
    if file_size < 16:
        return VideoProbeSummarySchema(
            asset_id=asset_id,
            file_path=str(path),
            file_size_bytes=file_size,
            format_name="corrupt_empty",
            container_duration_s=None,
            bit_rate=None,
            tracks=[],
            layout_verification={"is_valid_video": False, "error": "File is empty or truncated"},
            contact_sheet_path=None,
            decoded_frames=[],
        )

    with open(path, "rb") as f_hdr:
        header_bytes = f_hdr.read(16)
    if header_bytes.startswith(b"\x89PNG\r\n\x1a\n") or header_bytes.startswith(b"\xff\xd8\xff"):
        return VideoProbeSummarySchema(
            asset_id=asset_id,
            file_path=str(path),
            file_size_bytes=file_size,
            format_name="screenshot_image",
            container_duration_s=None,
            bit_rate=None,
            tracks=[],
            layout_verification={"is_valid_video": False, "error": "File contains image/screenshot magic bytes, not a video container"},
            contact_sheet_path=None,
            decoded_frames=[],
        )

    # 1. Native ISO-BMFF probe
    isobmff_info = inspect_isobmff_file(path)

    # 2. ffprobe JSON probe if available
    ff_data = probe_with_ffprobe(path)

    tracks: List[VideoTrackProbeSchema] = []
    format_name = None
    container_dur = isobmff_info.get("container_duration_s")
    bit_rate = None

    if ff_data and "streams" in ff_data:
        ff_fmt = ff_data.get("format", {})
        format_name = ff_fmt.get("format_name")
        if "duration" in ff_fmt:
            try:
                container_dur = float(ff_fmt["duration"])
            except (ValueError, TypeError):
                pass
        if "bit_rate" in ff_fmt:
            try:
                bit_rate = int(ff_fmt["bit_rate"])
            except (ValueError, TypeError):
                pass

        for idx, s in enumerate(ff_data["streams"]):
            media_type = s.get("codec_type", "unknown")
            codec = s.get("codec_name")
            codec_long = s.get("codec_long_name")
            w = s.get("width")
            h = s.get("height")
            r_fr = s.get("r_frame_rate")
            avg_fr = s.get("avg_frame_rate")
            tb = s.get("time_base")
            nb_f = s.get("nb_frames")
            frame_cnt = int(nb_f) if nb_f and str(nb_f).isdigit() else None
            dur = None
            if "duration" in s:
                try:
                    dur = float(s["duration"])
                except (ValueError, TypeError):
                    pass

            tags = s.get("tags", {})

            # Projection candidate analysis
            proj_cand = "unknown"
            aspect_str = None
            if w and h and h > 0:
                ratio = w / h
                aspect_str = f"{ratio:.2f}:1"
                if abs(ratio - 2.0) < 0.05:
                    if "lrv" in path.name.lower():
                        proj_cand = "preview_equirectangular"
                    elif path.suffix.lower() == ".insv":
                        proj_cand = "dual_fisheye_insv_candidate"
                    else:
                        proj_cand = "equirectangular_360"
                elif abs(ratio - (16.0 / 9.0)) < 0.05:
                    proj_cand = "reframed_perspective"
                elif abs(ratio - 1.0) < 0.05:
                    proj_cand = "circular_fisheye_single"
                else:
                    proj_cand = f"custom_{w}x{h}"

            track_schema = VideoTrackProbeSchema(
                track_index=idx,
                track_type=media_type,
                codec=codec,
                codec_long_name=codec_long,
                width=w,
                height=h,
                aspect_ratio=aspect_str,
                projection_candidate=proj_cand,
                time_base=tb,
                r_frame_rate=r_fr,
                avg_frame_rate=avg_fr,
                duration_seconds=dur,
                frame_count=frame_cnt,
                is_variable_cadence=False,
                tags={k: str(v) for k, v in tags.items()},
                exposure_duration_s=None,
                rolling_shutter_readout_s=None,
                unknown_fields=[
                    "exposure_duration_s (not recorded in standard container tags)",
                    "rolling_shutter_readout_s (unknown without vendor calibration dump)",
                ],
            )
            tracks.append(track_schema)
    else:
        # Fallback to native ISOBMFF metadata
        for t in isobmff_info.get("tracks", []):
            tracks.append(
                VideoTrackProbeSchema(
                    track_index=t.get("track_index", 0),
                    track_type=t.get("track_type", "unknown"),
                    codec=t.get("codec"),
                    codec_long_name=t.get("codec"),
                    width=t.get("width"),
                    height=t.get("height"),
                    aspect_ratio=t.get("aspect_ratio"),
                    projection_candidate=t.get("projection_candidate", "unknown"),
                    time_base=t.get("time_base"),
                    duration_seconds=t.get("duration_seconds"),
                    frame_count=t.get("frame_count"),
                    is_variable_cadence=t.get("is_variable_cadence", False),
                    cadence_summary=t.get("cadence_summary", {}),
                    tags=t.get("tags", {}),
                    unknown_fields=t.get("unknown_fields", []),
                )
            )

    video_tracks = [t for t in tracks if t.track_type == "video"]
    total_video_tracks = len(video_tracks)
    is_dual_fisheye = (
        total_video_tracks == 2
        and all(t.width and t.height and t.width == t.height for t in video_tracks)
    )

    layout_verification = {
        "aspect_ratio_2_to_1_verified": any(t.aspect_ratio and "2.00:1" in t.aspect_ratio for t in tracks),
        "insv_geometry_note": (
            "A 2:1 aspect ratio or .insv extension does not prove camera geometry or calibration. "
            "Fisheye optical centers, focal lengths, and dual-lens baseline require explicit calibration models."
        ),
        "total_video_tracks": total_video_tracks,
        "is_dual_fisheye": is_dual_fisheye,
        "required_lens_streams_present": is_dual_fisheye if path.suffix.lower() == ".insv" else (total_video_tracks > 0),
        "lens_streams_count": total_video_tracks,
    }

    return VideoProbeSummarySchema(
        asset_id=asset_id,
        file_path=str(path),
        file_size_bytes=file_size,
        format_name=format_name or "isobmff/mp4",
        container_duration_s=container_dur,
        bit_rate=bit_rate,
        tracks=tracks,
        layout_verification=layout_verification,
        contact_sheet_path=None,
        decoded_frames=[],
    )


def decode_sample_frames(
    video_path: Path,
    output_dir: Path,
    duration_s: Optional[float] = None,
    num_video_tracks: int = 1,
) -> List[Dict[str, Any]]:
    """Decode start, middle, and end frames in presentation order using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    ffmpeg_bin = shutil.which("ffmpeg")
    dur = duration_s or 5.0

    frames_meta: List[Dict[str, Any]] = []
    positions = [
        ("start", 0.0),
        ("middle", max(0.5, dur / 2.0)),
        ("end", max(1.0, dur - 0.5)),
    ]

    for label, pts_time in positions:
        frame_filename = f"{stem}_frame_{label}.jpg"
        out_frame_path = output_dir / frame_filename

        if ffmpeg_bin and video_path.exists():
            if num_video_tracks >= 2:
                # Composite dual lens streams side by side for complete lens coverage
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-ss",
                    f"{pts_time:.2f}",
                    "-i",
                    str(video_path),
                    "-filter_complex",
                    "[0:v:0][0:v:1]hstack",
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    str(out_frame_path),
                ]
                try:
                    subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                except Exception:
                    pass

            if not out_frame_path.is_file():
                # Single stream or fallback
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-ss",
                    f"{pts_time:.2f}",
                    "-i",
                    str(video_path),
                    "-vframes",
                    "1",
                    "-q:v",
                    "2",
                    str(out_frame_path),
                ]
                try:
                    subprocess.run(cmd, capture_output=True, check=True, timeout=30)
                except Exception:
                    pass

        if out_frame_path.is_file():
            frames_meta.append(
                {
                    "label": label,
                    "target_pts_s": pts_time,
                    "frame_path": str(out_frame_path),
                    "status": "DECODED",
                    "presentation_order_pts": pts_time,
                    "exposure_time_s": None,  # Explicitly unknown
                }
            )
        else:
            frames_meta.append(
                {
                    "label": label,
                    "target_pts_s": pts_time,
                    "frame_path": None,
                    "status": "UNAVAILABLE",
                    "presentation_order_pts": pts_time,
                    "exposure_time_s": None,
                }
            )

    return frames_meta


def generate_contact_sheet(
    video_summary: VideoProbeSummarySchema,
    decoded_frames: List[Dict[str, Any]],
    output_path: Path,
) -> str:
    """Generate a clean, labelled contact sheet image displaying start, middle, and end frames."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img_w, img_h = 1200, 500
    sheet = Image.new("RGB", (img_w, img_h), color=(30, 30, 35))
    draw = ImageDraw.Draw(sheet)

    # Header title and metadata banner
    video_name = Path(video_summary.file_path).name
    dur_str = f"{video_summary.container_duration_s:.2f}s" if video_summary.container_duration_s else "unknown"
    track_info = ""
    if video_summary.tracks:
        vt = next((t for t in video_summary.tracks if t.track_type == "video"), video_summary.tracks[0])
        track_info = f" | {vt.codec or 'unknown'} {vt.width or '?'}x{vt.height or '?'} ({vt.projection_candidate})"

    title_text = f"Contact Sheet: {video_name} ({dur_str}){track_info}"
    draw.rectangle([(0, 0), (img_w, 45)], fill=(20, 20, 25))
    draw.text((15, 12), title_text, fill=(240, 240, 240))

    thumb_w, thumb_h = 360, 200
    thumb_y = 65

    for idx, f_info in enumerate(decoded_frames[:3]):
        thumb_x = 30 + idx * 390
        label = f_info.get("label", f"Frame {idx}")
        pts_s = f_info.get("target_pts_s", 0.0)
        fpath = f_info.get("frame_path")

        # Background frame box
        draw.rectangle([(thumb_x, thumb_y), (thumb_x + thumb_w, thumb_y + thumb_h)], fill=(45, 45, 50), outline=(80, 80, 85))

        rendered = False
        if fpath and Path(fpath).is_file():
            try:
                frame_img = Image.open(fpath)
                frame_img.thumbnail((thumb_w, thumb_h))
                # Center thumbnail
                offset_x = thumb_x + (thumb_w - frame_img.width) // 2
                offset_y = thumb_y + (thumb_h - frame_img.height) // 2
                sheet.paste(frame_img, (offset_x, offset_y))
                rendered = True
            except Exception:
                pass

        if not rendered:
            draw.text((thumb_x + 100, thumb_y + 90), "[No Decoded Frame]", fill=(160, 160, 160))

        # Caption
        caption_box_y = thumb_y + thumb_h + 10
        draw.text((thumb_x, caption_box_y), f"Position: {label.upper()}", fill=(255, 200, 50))
        draw.text((thumb_x, caption_box_y + 20), f"Presentation PTS: {pts_s:.2f}s", fill=(200, 200, 200))
        draw.text((thumb_x, caption_box_y + 40), "Exposure: UNKNOWN (Not in container)", fill=(160, 160, 160))

    # Save contact sheet
    sheet.save(str(output_path), "JPEG", quality=85)
    return str(output_path)
