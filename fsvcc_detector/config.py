"""
Runtime configuration for the FSVCC wildlife detector.

Settings are persisted to config.json beside the executable so volunteers
only have to set them once.  All paths resolve relative to the directory
that contains the running script / executable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory that contains this file (or the frozen .exe directory at runtime)
_BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent


@dataclass
class Config:
    # ── Detection thresholds ──────────────────────────────────────────────────
    megadetector_confidence: float = 0.20
    """MegaDetector minimum confidence to consider a box an animal (0-1)."""

    classifier_confidence_threshold: float = 0.65
    """Species classifier confidence below which 'Needs Review' is set TRUE."""

    tie_margin: float = 0.10
    """If top-2 species confidences are within this margin, flag for review."""

    # ── Video sampling ────────────────────────────────────────────────────────
    frame_sample_rate: float = 1.0
    """Frames to extract per second of video (1 fps balances speed vs coverage)."""

    max_frames_per_video: int = 120
    """Cap to prevent very long videos from stalling the queue."""

    # ── Google Sheets ─────────────────────────────────────────────────────────
    sheet_id: str = ""
    """The long ID from the Google Sheet URL (between /d/ and /edit)."""

    worksheet_name: str = "Riparian Trail"
    """Name of the specific tab inside the spreadsheet."""

    service_account_path: str = "service_account.json"
    """Path to the Google Cloud service-account JSON key file."""

    # ── Output ────────────────────────────────────────────────────────────────
    output_csv: str = "detections.csv"
    """Local CSV backup written alongside every run."""

    crops_dir: str = "crops"
    """Directory where animal crops are saved (used by build_review_set.py)."""

    save_crops: bool = False
    """Set True to save every detected-animal crop to crops_dir."""

    # ── Model / hardware ──────────────────────────────────────────────────────
    device: str = "cpu"
    """Torch device ('cpu' or 'cuda')."""

    models_dir: str = "models"
    """Directory where downloaded model weights are cached."""

    custom_classifier_path: str = ""
    """
    Path to a Phase-2 ONNX classifier (leave blank to use CLIP zero-shot).
    Once trained and exported, set this to 'models/fsvcc_classifier.onnx'.
    """

    # ── GUI preferences ───────────────────────────────────────────────────────
    theme: str = "dark"
    color_theme: str = "green"

    # ── Internal ──────────────────────────────────────────────────────────────
    _config_path: str = field(default="config.json", repr=False, compare=False)

    # ------------------------------------------------------------------
    def resolved_path(self, relative: str) -> Path:
        """Resolve a config path relative to the application base directory."""
        p = Path(relative)
        return p if p.is_absolute() else _BASE_DIR / p

    @property
    def service_account_resolved(self) -> Path:
        return self.resolved_path(self.service_account_path)

    @property
    def models_dir_resolved(self) -> Path:
        return self.resolved_path(self.models_dir)

    @property
    def output_csv_resolved(self) -> Path:
        return self.resolved_path(self.output_csv)

    @property
    def crops_dir_resolved(self) -> Path:
        return self.resolved_path(self.crops_dir)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = "config.json") -> "Config":
        """Load config from JSON; fall back to defaults on missing keys."""
        p = _BASE_DIR / path
        if p.exists():
            try:
                with p.open() as f:
                    data = json.load(f)
                obj = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
                obj._config_path = str(path)
                logger.info("Config loaded from %s", p)
                return obj
            except Exception as exc:
                logger.warning("Could not parse %s (%s) — using defaults.", p, exc)
        return cls()

    def save(self, path: str | Path | None = None) -> None:
        """Persist current settings to JSON."""
        p = _BASE_DIR / (path or self._config_path)
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        with p.open("w") as f:
            json.dump(d, f, indent=2)
        logger.debug("Config saved to %s", p)


# Module-level singleton loaded at import time
cfg = Config.load()
