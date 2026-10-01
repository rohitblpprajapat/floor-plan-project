"""Pure-Python ISO-BMFF (MP4, QuickTime, INSV, LRV) container and atom parser.

Parses box hierarchies (ftyp, moov, trak, tkhd, mdia, mdhd, hdlr, minf, stbl, stsd, stts, ctts, stss, udta)
without requiring external binaries or ffmpeg. Accurately extracts track counts, dimensions,
FourCC codecs, time scales, durations, exact frame counts, variable cadence profiles, and tags.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple


class ISOBMFFBox:
    """Represents a single parsed ISO-BMFF box / atom."""

    def __init__(self, box_type: str, offset: int, size: int, header_size: int, payload: bytes = b"") -> None:
        self.box_type = box_type
        self.offset = offset
        self.size = size
        self.header_size = header_size
        self.payload = payload
        self.children: List[ISOBMFFBox] = []

    def get_child(self, box_type: str) -> Optional[ISOBMFFBox]:
        for c in self.children:
            if c.box_type == box_type:
                return c
        return None

    def get_children(self, box_type: str) -> List[ISOBMFFBox]:
        return [c for c in self.children if c.box_type == box_type]


CONTAINER_BOXES = {"moov", "trak", "mdia", "minf", "stbl", "udta", "edts", "dinf"}


def parse_boxes(stream: BinaryIO, start_offset: int, end_offset: int, max_depth: int = 6) -> List[ISOBMFFBox]:
    """Recursively parse ISO-BMFF boxes from binary stream between start and end offsets."""
    boxes: List[ISOBMFFBox] = []
    current_offset = start_offset

    while current_offset + 8 <= end_offset:
        stream.seek(current_offset)
        header = stream.read(8)
        if len(header) < 8:
            break

        size, raw_type = struct.unpack(">I4s", header)
        try:
            box_type = raw_type.decode("ascii", errors="replace").strip()
        except Exception:
            box_type = "unknown"

        header_size = 8
        if size == 1:
            # 64-bit large size
            ext_hdr = stream.read(8)
            if len(ext_hdr) < 8:
                break
            size = struct.unpack(">Q", ext_hdr)[0]
            header_size = 16
        elif size == 0:
            # Extends to end of file
            size = end_offset - current_offset

        if size < header_size:
            # Corrupted box header
            break

        payload_offset = current_offset + header_size
        payload_size = size - header_size
        box = ISOBMFFBox(box_type, current_offset, size, header_size)

        if box_type in CONTAINER_BOXES and max_depth > 0:
            sub_end = min(current_offset + size, end_offset)
            box.children = parse_boxes(stream, payload_offset, sub_end, max_depth=max_depth - 1)
        elif payload_size > 0 and payload_size <= 1048576:  # Read payloads up to 1MB into memory
            stream.seek(payload_offset)
            box.payload = stream.read(payload_size)

        boxes.append(box)
        current_offset += size

    return boxes


def parse_mvhd(payload: bytes) -> Dict[str, Any]:
    """Parse Movie Header (mvhd) payload."""
    if len(payload) < 20:
        return {}
    version = payload[0]
    if version == 1:
        if len(payload) < 32:
            return {}
        # creation(8), mod(8), time_scale(4), duration(8)
        _, _, time_scale, duration = struct.unpack(">QQIQ", payload[4:36])
    else:
        # creation(4), mod(4), time_scale(4), duration(4)
        _, _, time_scale, duration = struct.unpack(">IIII", payload[4:20])

    duration_s = duration / time_scale if time_scale > 0 else 0.0
    return {
        "time_scale": time_scale,
        "duration_units": duration,
        "duration_seconds": duration_s,
    }


def parse_tkhd(payload: bytes) -> Dict[str, Any]:
    """Parse Track Header (tkhd) payload."""
    if len(payload) < 84:
        return {}
    version = payload[0]
    offset = 4
    if version == 1:
        offset += 16 + 4 + 4 + 8  # creation(8), mod(8), track_id(4), reserved(4), duration(8)
    else:
        offset += 8 + 4 + 4 + 4  # creation(4), mod(4), track_id(4), reserved(4), duration(4)

    # After duration comes: reserved(8), layer(2), alt_group(2), volume(2), reserved(2), matrix(36) = 52 bytes
    offset += 52
    if len(payload) >= offset + 8:
        # width and height are 16.16 fixed point numbers
        w_int, w_frac, h_int, h_frac = struct.unpack(">HHHH", payload[offset : offset + 8])
        width = w_int + (w_frac / 65536.0)
        height = h_int + (h_frac / 65536.0)
        return {"width": int(round(width)), "height": int(round(height))}
    return {}


def parse_mdhd(payload: bytes) -> Dict[str, Any]:
    """Parse Media Header (mdhd) payload."""
    if len(payload) < 20:
        return {}
    version = payload[0]
    if version == 1:
        if len(payload) < 32:
            return {}
        _, _, time_scale, duration = struct.unpack(">QQIQ", payload[4:36])
    else:
        _, _, time_scale, duration = struct.unpack(">IIII", payload[4:20])

    duration_s = duration / time_scale if time_scale > 0 else 0.0
    return {
        "time_scale": time_scale,
        "duration_units": duration,
        "duration_seconds": duration_s,
    }


def parse_hdlr(payload: bytes) -> Dict[str, Any]:
    """Parse Handler Reference (hdlr) payload."""
    if len(payload) < 24:
        return {}
    handler_type = payload[8:12].decode("ascii", errors="replace").strip()
    name = payload[24:].decode("utf-8", errors="replace").strip("\x00")
    return {
        "handler_type": handler_type,  # 'vide', 'soun', 'meta', 'hint'
        "handler_name": name,
    }


def parse_stsd(payload: bytes) -> Dict[str, Any]:
    """Parse Sample Description (stsd) payload for video codecs and dimensions."""
    if len(payload) < 8:
        return {}
    entry_count = struct.unpack(">I", payload[4:8])[0]
    entries: List[Dict[str, Any]] = []
    offset = 8

    for _ in range(entry_count):
        if offset + 8 > len(payload):
            break
        entry_size, raw_fourcc = struct.unpack(">I4s", payload[offset : offset + 8])
        fourcc = raw_fourcc.decode("ascii", errors="replace").strip()
        entry_data: Dict[str, Any] = {"fourcc": fourcc, "size": entry_size}

        # For visual sample entries (avc1, hvc1, mp4v, etc.)
        # bytes 24-28 contain width (uint16) and height (uint16)
        if entry_size >= 36 and len(payload) >= offset + 32:
            w, h = struct.unpack(">HH", payload[offset + 24 : offset + 28])
            if w > 0 and h > 0:
                entry_data["width"] = w
                entry_data["height"] = h

        entries.append(entry_data)
        offset += entry_size

    return {"entry_count": entry_count, "entries": entries}


def parse_stts(payload: bytes) -> Dict[str, Any]:
    """Parse Time-to-Sample (stts) payload to calculate frame count and cadence variability."""
    if len(payload) < 8:
        return {}
    entry_count = struct.unpack(">I", payload[4:8])[0]
    total_samples = 0
    deltas: List[int] = []
    offset = 8

    for _ in range(min(entry_count, 100000)):
        if offset + 8 > len(payload):
            break
        count, delta = struct.unpack(">II", payload[offset : offset + 8])
        total_samples += count
        deltas.append(delta)
        offset += 8

    is_variable = len(set(deltas)) > 1 if deltas else False
    return {
        "sample_count": total_samples,
        "is_variable_cadence": is_variable,
        "distinct_deltas": list(set(deltas)),
        "primary_delta": deltas[0] if deltas else 0,
    }


def parse_ctts(payload: bytes) -> Dict[str, Any]:
    """Parse Composition Time to Sample (ctts) payload (PTS/DTS offsets)."""
    if len(payload) < 8:
        return {}
    entry_count = struct.unpack(">I", payload[4:8])[0]
    has_offsets = entry_count > 0
    return {"entry_count": entry_count, "has_pts_offsets": has_offsets}


def inspect_isobmff_file(file_path: Path) -> Dict[str, Any]:
    """Inspect ISO-BMFF file and return structured container and track metadata."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = path.stat().st_size
    with open(path, "rb") as f:
        root_boxes = parse_boxes(f, 0, file_size, max_depth=5)

    moov = None
    for b in root_boxes:
        if b.box_type == "moov":
            moov = b
            break

    if not moov:
        return {
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "has_moov": False,
            "is_truncated": True,
            "error": "Missing 'moov' atom (file may be truncated or incomplete recording)",
            "tracks": [],
        }

    mvhd = moov.get_child("mvhd")
    mvhd_info = parse_mvhd(mvhd.payload) if mvhd else {}

    tracks_data: List[Dict[str, Any]] = []
    trak_boxes = moov.get_children("trak")

    for idx, trak in enumerate(trak_boxes):
        track_dict: Dict[str, Any] = {
            "track_index": idx,
            "track_type": "unknown",
            "codec": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "duration_seconds": None,
            "time_base": None,
            "frame_count": 0,
            "is_variable_cadence": False,
            "projection_candidate": "unknown",
            "tags": {},
            "unknown_fields": ["exposure_duration_s", "rolling_shutter_readout_s"],
        }

        tkhd = trak.get_child("tkhd")
        if tkhd:
            tkhd_info = parse_tkhd(tkhd.payload)
            if "width" in tkhd_info and tkhd_info["width"] > 0:
                track_dict["width"] = tkhd_info["width"]
                track_dict["height"] = tkhd_info["height"]

        mdia = trak.get_child("mdia")
        if mdia:
            mdhd = mdia.get_child("mdhd")
            if mdhd:
                mdhd_info = parse_mdhd(mdhd.payload)
                track_dict["duration_seconds"] = mdhd_info.get("duration_seconds")
                ts = mdhd_info.get("time_scale")
                if ts:
                    track_dict["time_base"] = f"1/{ts}"

            hdlr = mdia.get_child("hdlr")
            if hdlr:
                hdlr_info = parse_hdlr(hdlr.payload)
                htype = hdlr_info.get("handler_type")
                if htype == "vide":
                    track_dict["track_type"] = "video"
                elif htype == "soun":
                    track_dict["track_type"] = "audio"
                elif htype == "meta":
                    track_dict["track_type"] = "metadata"
                else:
                    track_dict["track_type"] = htype or "unknown"
                if hdlr_info.get("handler_name"):
                    track_dict["tags"]["handler_name"] = hdlr_info["handler_name"]

            minf = mdia.get_child("minf")
            if minf:
                stbl = minf.get_child("stbl")
                if stbl:
                    stsd = stbl.get_child("stsd")
                    if stsd:
                        stsd_info = parse_stsd(stsd.payload)
                        entries = stsd_info.get("entries", [])
                        if entries:
                            track_dict["codec"] = entries[0].get("fourcc")
                            if "width" in entries[0] and entries[0]["width"] > 0:
                                track_dict["width"] = entries[0]["width"]
                                track_dict["height"] = entries[0]["height"]

                    stts = stbl.get_child("stts")
                    if stts:
                        stts_info = parse_stts(stts.payload)
                        track_dict["frame_count"] = stts_info.get("sample_count", 0)
                        track_dict["is_variable_cadence"] = stts_info.get("is_variable_cadence", False)
                        track_dict["cadence_summary"] = stts_info

                    ctts = stbl.get_child("ctts")
                    if ctts:
                        ctts_info = parse_ctts(ctts.payload)
                        track_dict["has_pts_offsets"] = ctts_info.get("has_pts_offsets", False)

        # Aspect ratio & Projection heuristic
        w = track_dict.get("width")
        h = track_dict.get("height")
        if w and h and h > 0:
            ratio = w / h
            track_dict["aspect_ratio"] = f"{ratio:.2f}:1"
            if abs(ratio - 2.0) < 0.05:
                # 2:1 aspect ratio
                if "lrv" in path.name.lower():
                    track_dict["projection_candidate"] = "preview_equirectangular"
                elif path.suffix.lower() == ".insv":
                    track_dict["projection_candidate"] = "dual_fisheye_insv_candidate"
                else:
                    track_dict["projection_candidate"] = "equirectangular_360"
            elif abs(ratio - (16.0 / 9.0)) < 0.05:
                track_dict["projection_candidate"] = "reframed_perspective"
            elif abs(ratio - 1.0) < 0.05:
                track_dict["projection_candidate"] = "circular_fisheye_single"
            else:
                track_dict["projection_candidate"] = f"custom_{w}x{h}"

        tracks_data.append(track_dict)

    # Extract udta tags if present
    udta = moov.get_child("udta")
    tags: Dict[str, str] = {}
    if udta:
        for c in udta.children:
            if c.payload:
                try:
                    val = c.payload.decode("utf-8", errors="ignore").strip()
                    if val:
                        tags[c.box_type] = val
                except Exception:
                    pass

    return {
        "file_path": str(file_path),
        "file_size_bytes": file_size,
        "has_moov": True,
        "is_truncated": False,
        "container_duration_s": mvhd_info.get("duration_seconds"),
        "mvhd": mvhd_info,
        "tracks": tracks_data,
        "tags": tags,
    }
