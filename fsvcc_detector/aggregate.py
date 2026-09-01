"""
Per-video result aggregation.

Aggregation logic
─────────────────
Species   – one vote per frame (top animal classification in that frame).
            Humans detected by MegaDetector are recorded as Human.
Count     – median number of winning-species boxes per frame (reduces
            double-detection inflation vs. taking the max).
Confidence– mean confidence of winning-species frame votes.
Needs Review  – True when confidence is low or top-2 species are very close.
Comments  – only notes a second species when it has meaningful support.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from .classifier import ClassificationResult
from .detector import AnimalBox

logger = logging.getLogger(__name__)

# Runner-up must reach this fraction of the winner's score to count as multi-species.
_MULTI_SPECIES_RATIO = 0.35
# Minimum frames the runner-up must appear in.
_MULTI_SPECIES_MIN_FRAMES = 2


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_ms: float
    is_night_ir: bool
    boxes: list[AnimalBox]
    classifications: list[ClassificationResult]


@dataclass
class VideoResult:
    recorded_at: datetime | None
    species_key: str
    common_name: str
    scientific_name: str
    count: int
    confidence: float
    needs_review: bool
    comments: str
    timestamp_source: str

    top3_species: list[tuple[str, float]] = field(default_factory=list)
    all_species_seen: list[str] = field(default_factory=list)
    total_frames_processed: int = 0
    frames_with_animals: int = 0
    is_night_ir: bool = False


class Aggregator:
    def __init__(
        self,
        confidence_threshold: float = 0.65,
        tie_margin: float = 0.10,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.tie_margin = tie_margin

    def aggregate(
        self,
        records: Sequence[FrameRecord],
        recorded_at: datetime | None = None,
        timestamp_source: str = "unknown",
    ) -> VideoResult:
        total = len(records)
        night_frames = [r for r in records if r.is_night_ir]
        is_night_ir = len(night_frames) > total * 0.5

        animal_frames = [r for r in records if any(b.is_animal for b in r.boxes)]
        person_frames = [r for r in records if any(b.is_person for b in r.boxes)]
        humans_present = bool(person_frames)

        if not animal_frames and not person_frames:
            return VideoResult(
                recorded_at=recorded_at,
                species_key="none",
                common_name="No Animal Detected",
                scientific_name="—",
                count=0,
                confidence=0.0,
                needs_review=False,
                comments=self._build_comments(is_night_ir=is_night_ir, no_animal=True),
                timestamp_source=timestamp_source,
                all_species_seen=[],
                total_frames_processed=total,
                frames_with_animals=0,
                is_night_ir=is_night_ir,
            )

        # Humans only — no wildlife in clip
        if not animal_frames and person_frames:
            person_confs = [
                b.confidence for r in person_frames for b in r.boxes if b.is_person
            ]
            max_people = max(
                sum(1 for b in r.boxes if b.is_person) for r in person_frames
            )
            mean_conf = sum(person_confs) / len(person_confs)
            return VideoResult(
                recorded_at=recorded_at,
                species_key="human",
                common_name="Human",
                scientific_name="Homo sapiens",
                count=max(1, max_people),
                confidence=round(mean_conf, 4),
                needs_review=False,
                comments=self._build_comments(is_night_ir=is_night_ir, human_only=True),
                timestamp_source=timestamp_source,
                all_species_seen=["human"],
                total_frames_processed=total,
                frames_with_animals=0,
                is_night_ir=is_night_ir,
            )

        # ── Per-frame species vote (one vote per frame, not per box) ──
        species_conf: dict[str, list[float]] = defaultdict(list)
        species_frame_hits: dict[str, int] = defaultdict(int)

        for record in animal_frames:
            wildlife = [
                c for c in record.classifications
                if c.species_key not in ("unknown", "human", "none")
            ]
            if not wildlife:
                continue
            best = max(wildlife, key=lambda c: c.confidence)
            species_conf[best.species_key].append(best.confidence)
            species_frame_hits[best.species_key] += 1

        if not species_conf:
            max_count = max(
                sum(1 for b in r.boxes if b.is_animal) for r in animal_frames
            )
            return self._no_id_result(
                max_count, is_night_ir, recorded_at,
                timestamp_source, total, len(animal_frames),
                humans_present=humans_present,
            )

        species_scores = {k: sum(v) for k, v in species_conf.items()}
        ranked = sorted(species_scores, key=species_scores.__getitem__, reverse=True)
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        winner_confs = species_conf[winner]
        mean_conf = sum(winner_confs) / len(winner_confs)

        # ── Count: median boxes classified as winner per frame ────────
        per_frame_counts: list[int] = []
        for record in animal_frames:
            n = sum(1 for c in record.classifications if c.species_key == winner)
            if n > 0:
                per_frame_counts.append(n)
        if per_frame_counts:
            count = max(1, round(statistics.median(per_frame_counts)))
        else:
            count = 1

        all_seen = list(dict.fromkeys(ranked))
        top3 = [
            (k, sum(species_conf[k]) / len(species_conf[k]))
            for k in ranked[:3]
        ]

        tie = (
            runner_up is not None
            and abs(species_scores[winner] - species_scores[runner_up])
            / max(species_scores[winner], 1e-6) < self.tie_margin
        )
        needs_review = mean_conf < self.confidence_threshold or tie

        # Only flag multiple species when runner-up has real support
        multi_species = False
        other_species: list[str] = []
        if runner_up is not None:
            ratio = species_scores[runner_up] / max(species_scores[winner], 1e-6)
            if (
                ratio >= _MULTI_SPECIES_RATIO
                and species_frame_hits[runner_up] >= _MULTI_SPECIES_MIN_FRAMES
            ):
                multi_species = True
                other_species = [s for s in all_seen if s != winner]

        from .species import by_key
        winner_entry = by_key(winner)
        comments = self._build_comments(
            is_night_ir=is_night_ir,
            multiple=multi_species,
            other_species=other_species,
            low_confidence=mean_conf < self.confidence_threshold,
            tie=tie,
            tie_species=runner_up,
            humans_present=humans_present,
        )

        return VideoResult(
            recorded_at=recorded_at,
            species_key=winner,
            common_name=winner_entry["common_name"] if winner_entry else winner,
            scientific_name=winner_entry["scientific_name"] if winner_entry else "Unknown",
            count=count,
            confidence=round(mean_conf, 4),
            needs_review=needs_review,
            comments=comments,
            timestamp_source=timestamp_source,
            top3_species=top3,
            all_species_seen=all_seen,
            total_frames_processed=total,
            frames_with_animals=len(animal_frames),
            is_night_ir=is_night_ir,
        )

    def _no_id_result(
        self,
        max_count: int,
        is_night_ir: bool,
        recorded_at: datetime | None,
        timestamp_source: str,
        total: int,
        animal_frames: int,
        humans_present: bool = False,
    ) -> VideoResult:
        return VideoResult(
            recorded_at=recorded_at,
            species_key="unknown",
            common_name="Unknown Animal",
            scientific_name="Unknown",
            count=max(1, max_count),
            confidence=0.0,
            needs_review=True,
            comments=self._build_comments(
                is_night_ir=is_night_ir,
                unidentified=True,
                humans_present=humans_present,
            ),
            timestamp_source=timestamp_source,
            total_frames_processed=total,
            frames_with_animals=animal_frames,
            is_night_ir=is_night_ir,
        )

    @staticmethod
    def _build_comments(
        *,
        is_night_ir: bool = False,
        no_animal: bool = False,
        unidentified: bool = False,
        multiple: bool = False,
        other_species: list[str] | None = None,
        low_confidence: bool = False,
        tie: bool = False,
        tie_species: str | None = None,
        humans_present: bool = False,
        human_only: bool = False,
    ) -> str:
        parts: list[str] = []

        if is_night_ir:
            parts.append("Night IR")
        if human_only:
            parts.append("Human detected (no wildlife)")
        if no_animal:
            parts.append("No animal detected")
        if unidentified:
            parts.append("Animal detected but species unidentified — review required")
        if humans_present and not human_only:
            parts.append("Human also present")
        if multiple and other_species:
            from .species import by_key
            names = [
                (by_key(s) or {}).get("common_name", s)
                for s in other_species
            ]
            parts.append(f"Possible second species (also: {', '.join(names)})")
        if tie and tie_species:
            from .species import by_key
            runner = (by_key(tie_species) or {}).get("common_name", tie_species)
            parts.append(f"Close call with {runner} — review recommended")
        if low_confidence:
            parts.append("Low confidence")

        return "; ".join(parts) if parts else ""
