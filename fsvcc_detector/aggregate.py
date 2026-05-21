"""
Per-video result aggregation.

Given the per-frame detections + classifications, this module produces a
single `VideoResult` that gets written as one row to the Google Sheet.

Aggregation logic
─────────────────
Species   – majority vote (by number of frames in which that species was
            the top prediction, weighted by confidence).
Count     – maximum number of animal bounding boxes observed in *any single
            frame*  (best estimate of the largest group present).
Confidence– mean confidence of the winning-species frames.
Needs Review  – True when:
                  • Mean confidence < threshold (default 0.65)
                  • Top-2 species confidences within `tie_margin` (0.10)
                  • No animal detected at all
Comments  – auto-generated notes: "Night IR", "Multiple species detected",
            "No animal detected", "Low confidence", etc.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from .classifier import ClassificationResult
from .detector import AnimalBox

logger = logging.getLogger(__name__)


# ── Per-frame record ──────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    frame_index: int
    timestamp_ms: float
    is_night_ir: bool
    boxes: list[AnimalBox]                          # raw MD detections
    classifications: list[ClassificationResult]     # one per box


# ── Final per-video result ────────────────────────────────────────────────────

@dataclass
class VideoResult:
    # Core Sheet columns
    recorded_at: datetime | None
    species_key: str
    common_name: str
    scientific_name: str
    count: int               # max simultaneous animals in any frame
    confidence: float        # 0–1
    needs_review: bool
    comments: str
    timestamp_source: str    # 'metadata' | 'ocr' | 'filename'

    # Diagnostics (not written to Sheet, used by build_review_set)
    top3_species: list[tuple[str, float]] = field(default_factory=list)
    all_species_seen: list[str]           = field(default_factory=list)
    total_frames_processed: int           = 0
    frames_with_animals: int              = 0
    is_night_ir: bool                     = False


# ── Aggregator ────────────────────────────────────────────────────────────────

class Aggregator:
    """
    Build a single VideoResult from the sequence of FrameRecords produced
    while scanning a video file.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        tie_margin: float = 0.10,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.tie_margin = tie_margin

    # ------------------------------------------------------------------
    def aggregate(
        self,
        records: Sequence[FrameRecord],
        recorded_at: datetime | None = None,
        timestamp_source: str = "unknown",
    ) -> VideoResult:
        """
        Collapse all frame-level records into one VideoResult.

        Parameters
        ----------
        records         : output from the processing pipeline for one video
        recorded_at     : recording timestamp from VideoMeta
        timestamp_source: how the timestamp was obtained
        """
        total = len(records)
        animal_frames = [r for r in records if r.boxes]
        night_frames  = [r for r in records if r.is_night_ir]
        is_night_ir   = len(night_frames) > total * 0.5

        # ── No animals detected ──────────────────────────────────────
        if not animal_frames:
            return VideoResult(
                recorded_at      = recorded_at,
                species_key      = "none",
                common_name      = "No Animal Detected",
                scientific_name  = "—",
                count            = 0,
                confidence       = 0.0,
                needs_review     = False,
                comments         = self._build_comments(
                    is_night_ir=is_night_ir,
                    no_animal=True,
                ),
                timestamp_source = timestamp_source,
                all_species_seen = [],
                total_frames_processed=total,
                frames_with_animals=0,
                is_night_ir=is_night_ir,
            )

        # ── Max simultaneous count ───────────────────────────────────
        max_count = max(len(r.boxes) for r in animal_frames)

        # ── Species vote (weighted by confidence) ─────────────────────
        species_conf: dict[str, list[float]] = defaultdict(list)
        for record in animal_frames:
            for clf in record.classifications:
                if clf.species_key != "unknown":
                    species_conf[clf.species_key].append(clf.confidence)

        if not species_conf:
            # MegaDetector found boxes but classifier couldn't ID anything
            return self._no_id_result(
                max_count, is_night_ir, recorded_at,
                timestamp_source, total, len(animal_frames),
            )

        # Weighted vote: each species scores sum(confidences)
        species_scores = {
            k: sum(v) for k, v in species_conf.items()
        }
        ranked = sorted(species_scores, key=species_scores.__getitem__, reverse=True)
        winner   = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        winner_confs = species_conf[winner]
        mean_conf    = sum(winner_confs) / len(winner_confs)

        # Deduplicated sorted list of all species seen
        all_seen = list(dict.fromkeys(ranked))

        # Top-3 species with average confidence
        top3 = [
            (k, sum(species_conf[k]) / len(species_conf[k]))
            for k in ranked[:3]
        ]

        # ── Review flag ───────────────────────────────────────────────
        tie = (
            runner_up is not None
            and abs(species_scores[winner] - species_scores[runner_up])
            / max(species_scores[winner], 1e-6) < self.tie_margin
        )
        needs_review = mean_conf < self.confidence_threshold or tie

        # ── Comments ──────────────────────────────────────────────────
        from .species import by_key
        winner_entry = by_key(winner)
        comments = self._build_comments(
            is_night_ir    = is_night_ir,
            multiple       = len(all_seen) > 1,
            other_species  = [s for s in all_seen[1:] if s != winner],
            low_confidence = mean_conf < self.confidence_threshold,
            tie            = tie,
            tie_species    = runner_up,
        )

        return VideoResult(
            recorded_at           = recorded_at,
            species_key           = winner,
            common_name           = winner_entry["common_name"]   if winner_entry else winner,
            scientific_name       = winner_entry["scientific_name"] if winner_entry else "Unknown",
            count                 = max_count,
            confidence            = round(mean_conf, 4),
            needs_review          = needs_review,
            comments              = comments,
            timestamp_source      = timestamp_source,
            top3_species          = top3,
            all_species_seen      = all_seen,
            total_frames_processed= total,
            frames_with_animals   = len(animal_frames),
            is_night_ir           = is_night_ir,
        )

    # ------------------------------------------------------------------
    def _no_id_result(
        self,
        max_count: int,
        is_night_ir: bool,
        recorded_at: datetime | None,
        timestamp_source: str,
        total: int,
        animal_frames: int,
    ) -> VideoResult:
        return VideoResult(
            recorded_at           = recorded_at,
            species_key           = "unknown",
            common_name           = "Unknown Animal",
            scientific_name       = "Unknown",
            count                 = max_count,
            confidence            = 0.0,
            needs_review          = True,
            comments              = self._build_comments(
                is_night_ir=is_night_ir,
                unidentified=True,
            ),
            timestamp_source      = timestamp_source,
            total_frames_processed= total,
            frames_with_animals   = animal_frames,
            is_night_ir           = is_night_ir,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_comments(
        *,
        is_night_ir: bool     = False,
        no_animal: bool       = False,
        unidentified: bool    = False,
        multiple: bool        = False,
        other_species: list[str] | None = None,
        low_confidence: bool  = False,
        tie: bool             = False,
        tie_species: str | None = None,
    ) -> str:
        parts: list[str] = []

        if is_night_ir:
            parts.append("Night IR")
        if no_animal:
            parts.append("No animal detected")
        if unidentified:
            parts.append("Animal detected but species unidentified — review required")
        if multiple and other_species:
            from .species import by_key
            names = [
                (by_key(s) or {}).get("common_name", s)
                for s in (other_species or [])
            ]
            parts.append(f"Multiple species detected (also: {', '.join(names)})")
        if tie and tie_species:
            from .species import by_key
            runner = (by_key(tie_species) or {}).get("common_name", tie_species)
            parts.append(f"Close call with {runner} — review recommended")
        if low_confidence:
            parts.append("Low confidence")

        return "; ".join(parts) if parts else ""
