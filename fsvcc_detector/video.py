"""
Video I/O utilities for Bushnell .mov trail-cam footage.

Responsibilities
----------------
1. Extract a sparse set of frames (at `sample_rate` fps) using OpenCV.
2. Determine the *recording* timestamp of the clip through a two-pass strategy:
     a. Primary  – QuickTime / MP4 metadata via pymediainfo
        (Bushnell cameras write "Recorded date" or "Encoded date" tags).
     b. Fallback – lightweight OCR of the burned-in date/time strip that
        Bushnell embeds in the bottom ~12 % of the frame.  Only attempted
        when the metadata parse fails.
3. Detect whether a frame is night-IR (infrared) by checking colour
   saturation; used to populate the auto-comment field.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Sequence

import cv2
import numpy as np
from dateutil import parser as du_parser

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VideoFrame:
    index: int           # 0-based frame index within the *sampled* sequence
    timestamp_ms: float  # position in video (ms)
    image: np.ndarray    # BGR uint8 HxWx3
    is_night_ir: bool    # True when frame appears to be IR (greyscale-ish)


@dataclass
class VideoMeta:
    path: Path
    recorded_at: datetime | None   # best-effort recording timestamp
    duration_s: float
    width: int
    height: int
    fps: float
    timestamp_source: str          # 'metadata', 'ocr', or 'filename'


# ── Public API ────────────────────────────────────────────────────────────────

def get_video_meta(video_path: str | Path) -> VideoMeta:
    """
    Return metadata for a .mov file including its recording timestamp.

    The timestamp is extracted in this priority order:
      1. QuickTime / MP4 tags via pymediainfo  (most accurate)
      2. OCR on the bottom strip of the first frame  (Bushnell burn-in)
      3. File modification time  (last resort — labelled as such)
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 1.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps

    # Grab first frame for OCR fallback
    ret, first_frame = cap.read()
    cap.release()

    recorded_at, source = _get_recorded_timestamp(path, first_frame if ret else None)

    return VideoMeta(
        path=path,
        recorded_at=recorded_at,
        duration_s=duration_s,
        width=width,
        height=height,
        fps=fps,
        timestamp_source=source,
    )


def extract_frames(
    video_path: str | Path,
    sample_rate: float = 1.0,
    max_frames: int = 120,
) -> Generator[VideoFrame, None, None]:
    """
    Yield frames sampled at `sample_rate` frames-per-second.

    Yields at most `max_frames` frames to cap processing time on long clips.
    Each yielded VideoFrame carries the BGR image and a night-IR flag.
    """
    path = Path(video_path)
    cap  = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute which frame indices to sample
    step = max(1, int(round(native_fps / sample_rate)))
    target_indices = list(range(0, total_frames, step))[:max_frames]

    sampled_idx = 0
    for frame_no in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, bgr = cap.read()
        if not ret:
            continue
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        yield VideoFrame(
            index=sampled_idx,
            timestamp_ms=ts_ms,
            image=bgr,
            is_night_ir=_is_night_ir(bgr),
        )
        sampled_idx += 1

    cap.release()


def crop_box(frame: np.ndarray, box_norm: Sequence[float]) -> np.ndarray:
    """
    Crop a detection box from a frame.

    Parameters
    ----------
    frame    : BGR uint8 image
    box_norm : (x1, y1, x2, y2) normalised to [0, 1]
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(box_norm[0] * w))
    y1 = max(0, int(box_norm[1] * h))
    x2 = min(w, int(box_norm[2] * w))
    y2 = min(h, int(box_norm[3] * h))
    if x2 <= x1 or y2 <= y1:
        return frame  # degenerate box – return full frame
    return frame[y1:y2, x1:x2]


# ── Night-IR detection ────────────────────────────────────────────────────────

def _is_night_ir(bgr: np.ndarray) -> bool:
    """
    Return True when the frame is likely an IR night shot.

    Bushnell IR clips are near-greyscale: the three colour channels are nearly
    equal.  We measure mean channel-wise standard deviation; if it's below a
    threshold the image is effectively monochrome.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    channel_std = np.std([b.mean(), g.mean(), r.mean()])
    return float(channel_std) < 6.0  # tuned empirically on Bushnell footage


# ── Timestamp extraction ──────────────────────────────────────────────────────

def _get_recorded_timestamp(
    path: Path, first_frame: np.ndarray | None
) -> tuple[datetime | None, str]:
    """Return (datetime, source_label) using best available method."""

    # 1. pymediainfo metadata
    dt = _ts_from_metadata(path)
    if dt:
        return dt, "metadata"

    # 2. OCR on burned-in Bushnell timestamp strip
    if first_frame is not None:
        dt = _ts_from_ocr(first_frame)
        if dt:
            return dt, "ocr"

    # 3. File modification time (last resort)
    try:
        mtime = path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        logger.warning(
            "%s: using file modification time as timestamp (no metadata/OCR)",
            path.name,
        )
        return dt, "filename"
    except Exception:
        pass

    return None, "unknown"


def _ts_from_metadata(path: Path) -> datetime | None:
    """Parse recording timestamp from QuickTime/MP4 tags via pymediainfo."""
    try:
        from pymediainfo import MediaInfo  # type: ignore

        info = MediaInfo.parse(str(path))
        for track in info.tracks:
            for attr in ("recorded_date", "encoded_date", "tagged_date"):
                raw = getattr(track, attr, None)
                if raw:
                    dt = _parse_raw_date(str(raw))
                    if dt:
                        return dt
    except Exception as exc:
        logger.debug("pymediainfo failed for %s: %s", path.name, exc)
    return None


_BUSHNELL_OCR_PATTERN = re.compile(
    r"(\d{1,2})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(\d{2,4})"
    r"\s+"
    r"(\d{1,2})\s*:\s*(\d{2})\s*:\s*(\d{2})"
    r"\s*(AM|PM)?",
    re.IGNORECASE,
)


_ocr_reader = None  # cached so it only loads once per process


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr  # type: ignore
        logger.info("Loading easyOCR for timestamp reading (one-time)…")
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def _ts_from_ocr(frame: np.ndarray) -> datetime | None:
    """
    Attempt to read the Bushnell burned-in timestamp from the bottom strip.

    Uses easyocr when available; gracefully degrades if not installed.
    The bottom 12% of the frame typically contains the date/time overlay.
    """
    try:
        import easyocr  # type: ignore  (optional dependency) -- noqa: F401

        h = frame.shape[0]
        strip = frame[int(h * 0.88):, :]  # bottom ~12 %

        # Convert to greyscale + threshold for better OCR accuracy
        grey = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        reader = _get_ocr_reader()
        results = reader.readtext(thresh, detail=0)
        text = " ".join(results)

        m = _BUSHNELL_OCR_PATTERN.search(text)
        if m:
            mo, day, yr, hr, mn, sc = (int(x) for x in m.groups()[:6])
            ampm = m.group(7)
            if yr < 100:
                yr += 2000
            if ampm and ampm.upper() == "PM" and hr < 12:
                hr += 12
            elif ampm and ampm.upper() == "AM" and hr == 12:
                hr = 0
            # Naive local time — matches what the camera burned into the frame.
            return datetime(yr, mo, day, hr, mn, sc)

    except ImportError:
        logger.debug("easyocr not installed — OCR timestamp fallback disabled.")
    except Exception as exc:
        logger.debug("OCR timestamp parse failed: %s", exc)
    return None


def _parse_raw_date(raw: str) -> datetime | None:
    """
    Parse a date string produced by pymediainfo into a timezone-aware datetime.

    Handles formats such as:
      - "UTC 2024-03-15 06:23:45"
      - "2024-03-15T06:23:45+00:00"
      - "2024-03-15 06:23:45"
    """
    raw = raw.strip().removeprefix("UTC").strip()
    try:
        dt = du_parser.parse(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
