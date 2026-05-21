"""
High-level processing pipeline.

`process_video()` ties together video I/O → MegaDetector → CLIP classifier
→ aggregation in a single callable that the GUI and CLI can both use.

It accepts an optional `progress_callback` so the GUI can update its
progress bar without the pipeline caring about UI details.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .aggregate import Aggregator, FrameRecord, VideoResult
from .classifier import SpeciesClassifier
from .config import cfg
from .detector import AnimalDetector
from .video import VideoMeta, extract_frames, get_video_meta

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]
"""Called as progress_callback(frames_done, frames_total, status_message)."""


@dataclass
class ProcessingResult:
    video_path: Path
    meta: VideoMeta
    result: VideoResult
    elapsed_s: float
    error: Optional[str] = None


class Pipeline:
    """
    Reusable processing pipeline.  Lazily initialises the heavy models on
    first call so the application starts quickly.

    Parameters
    ----------
    detector   : pre-created AnimalDetector (or None to auto-create from cfg)
    classifier : pre-created SpeciesClassifier (or None to auto-create)
    aggregator : pre-created Aggregator (or None to auto-create)
    """

    def __init__(
        self,
        detector:   AnimalDetector   | None = None,
        classifier: SpeciesClassifier | None = None,
        aggregator: Aggregator        | None = None,
    ) -> None:
        self._detector   = detector
        self._classifier = classifier
        self._aggregator = aggregator

    # Lazy accessors --------------------------------------------------

    @property
    def detector(self) -> AnimalDetector:
        if self._detector is None:
            self._detector = AnimalDetector(
                confidence_threshold=cfg.megadetector_confidence,
                device=cfg.device,
                models_dir=cfg.models_dir_resolved,
            )
        return self._detector

    @property
    def classifier(self) -> SpeciesClassifier:
        if self._classifier is None:
            self._classifier = SpeciesClassifier(
                device=cfg.device,
                models_dir=cfg.models_dir_resolved,
                custom_classifier_path=cfg.custom_classifier_path,
            )
        return self._classifier

    @property
    def aggregator(self) -> Aggregator:
        if self._aggregator is None:
            self._aggregator = Aggregator(
                confidence_threshold=cfg.classifier_confidence_threshold,
                tie_margin=cfg.tie_margin,
            )
        return self._aggregator

    # Main entry point -------------------------------------------------

    def process_video(
        self,
        video_path: str | Path,
        progress_callback: ProgressCallback | None = None,
        save_crops: bool = False,
        crops_dir: str | Path | None = None,
    ) -> ProcessingResult:
        """
        Process a single .mov file end-to-end.

        Returns a ProcessingResult regardless of errors (check `.error`).
        """
        path = Path(video_path)
        t0   = time.perf_counter()

        # ── Metadata ────────────────────────────────────────────────
        try:
            meta = get_video_meta(path)
        except Exception as exc:
            logger.error("Cannot read %s: %s", path.name, exc)
            return ProcessingResult(
                video_path=path,
                meta=None,          # type: ignore
                result=None,        # type: ignore
                elapsed_s=0.0,
                error=str(exc),
            )

        # ── Frame loop ───────────────────────────────────────────────
        frames_iter = list(
            extract_frames(
                path,
                sample_rate=cfg.frame_sample_rate,
                max_frames=cfg.max_frames_per_video,
            )
        )
        total = len(frames_iter)
        records: list[FrameRecord] = []
        _crops_dir = Path(crops_dir or cfg.crops_dir_resolved)

        _detector_failed = False

        for i, vframe in enumerate(frames_iter):
            if _detector_failed:
                records.append(FrameRecord(
                    frame_index=vframe.index, timestamp_ms=vframe.timestamp_ms,
                    is_night_ir=vframe.is_night_ir, boxes=[], classifications=[],
                ))
                continue

            if progress_callback:
                progress_callback(i, total, f"Frame {i+1}/{total}")

            try:
                boxes = self.detector.detect(vframe.image)
            except Exception as exc:
                logger.warning("Detector error on frame %d of %s: %s", i, path.name, exc)
                # If the detector recorded a permanent load failure, stop trying
                if self.detector._load_error:
                    logger.error(
                        "MegaDetector failed to load for %s — skipping remaining frames.\n"
                        "  Reason: %s", path.name, self.detector._load_error
                    )
                    _detector_failed = True
                boxes = []

            classifications = []
            for box in boxes:
                try:
                    crop = self.detector.crop(vframe.image, box)
                    clf  = self.classifier.classify(
                        crop,
                        full_frame=vframe.image,
                        box_norm=(box.x1, box.y1, box.x2, box.y2),
                    )
                    classifications.append(clf)

                    if save_crops and crops_dir:
                        _save_crop(crop, path, i, clf.species_key, _crops_dir)

                except Exception as exc:
                    logger.warning("Classifier error on frame %d of %s: %s", i, path.name, exc)

            records.append(FrameRecord(
                frame_index      = vframe.index,
                timestamp_ms     = vframe.timestamp_ms,
                is_night_ir      = vframe.is_night_ir,
                boxes            = boxes,
                classifications  = classifications,
            ))

        if progress_callback:
            progress_callback(total, total, "Aggregating…")

        # ── Aggregate ────────────────────────────────────────────────
        result = self.aggregator.aggregate(
            records,
            recorded_at      = meta.recorded_at,
            timestamp_source = meta.timestamp_source,
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "%s → %s  (conf %.2f, count %d, %.1fs)",
            path.name, result.common_name, result.confidence,
            result.count, elapsed,
        )

        return ProcessingResult(
            video_path=path,
            meta=meta,
            result=result,
            elapsed_s=round(elapsed, 2),
        )

    def process_folder(
        self,
        folder: str | Path,
        sheets_writer=None,
        progress_callback: ProgressCallback | None = None,
        video_callback: Callable[[ProcessingResult], None] | None = None,
        stop_event=None,
    ) -> list[ProcessingResult]:
        """
        Process every .mov file in `folder` (non-recursive).

        Parameters
        ----------
        sheets_writer    : SheetsWriter instance (or None to skip Sheet writes)
        progress_callback: called with (frames_done, frames_total, msg) per frame
        video_callback   : called with ProcessingResult after each video finishes
        stop_event       : threading.Event; set it to cancel mid-batch
        """
        folder = Path(folder)
        mov_files = _find_videos(folder)
        results: list[ProcessingResult] = []

        for video_path in mov_files:
            if stop_event and stop_event.is_set():
                logger.info("Processing cancelled by user.")
                break

            pr = self.process_video(
                video_path,
                progress_callback=progress_callback,
                save_crops=cfg.save_crops,
            )
            results.append(pr)

            if pr.error is None and sheets_writer is not None:
                try:
                    sheets_writer.write(pr.result, pr.video_path)
                except Exception as exc:
                    logger.error("Failed to write result for %s: %s", video_path.name, exc)

            if video_callback:
                video_callback(pr)

        return results


# ── Crop saving helper ────────────────────────────────────────────────────────

_VIDEO_EXTENSIONS = ("*.mov", "*.MOV", "*.avi", "*.AVI", "*.mp4", "*.MP4")


def _find_videos(folder: Path) -> list[Path]:
    """Return all supported video files in `folder`, sorted by name."""
    files: list[Path] = []
    for pattern in _VIDEO_EXTENSIONS:
        files.extend(folder.glob(pattern))
    return sorted(set(files))


def _save_crop(
    crop: np.ndarray,
    video_path: Path,
    frame_idx: int,
    species_key: str,
    crops_dir: Path,
) -> None:
    """Save an animal crop JPEG for later volunteer review / Phase-2 training."""
    import cv2

    dest = crops_dir / species_key
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"{video_path.stem}_f{frame_idx:04d}"
    out  = dest / f"{stem}.jpg"
    cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
