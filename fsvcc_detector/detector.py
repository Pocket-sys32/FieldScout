"""
Animal detection via MegaDetector v6 (Microsoft PytorchWildlife).

MegaDetector v6 is a YOLOv9-based model that classifies every box as one of:
  class 0 – animal
  class 1 – person
  class 2 – vehicle

We only forward *animal* boxes to the species classifier.

The model weights (~140 MB) are downloaded automatically from HuggingFace
on first use and cached in the configured `models_dir`.

Design notes
────────────
• Lazy-loaded so the GUI remains responsive at startup.
• Thread-safe: the single model instance is created once and reused.
• `detect()` accepts a BGR numpy array (direct OpenCV output) and returns
  a list of `AnimalBox` objects with normalised coordinates so callers
  never need to know the original image size.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_load_lock = threading.Lock()


# ── Public data class ──────────────────────────────────────────────────────────

# MegaDetector class IDs
_CLASS_ANIMAL = 0
_CLASS_PERSON = 1
# class 2 = vehicle — ignored


@dataclass
class AnimalBox:
    """A single MegaDetector detection, coordinates normalised to [0,1]."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "animal"   # 'animal' | 'person'

    def to_norm_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def is_person(self) -> bool:
        return self.label == "person"

    @property
    def is_animal(self) -> bool:
        return self.label == "animal"


# ── Detector class ────────────────────────────────────────────────────────────

class AnimalDetector:
    """
    Wraps MegaDetector v6 from Microsoft PytorchWildlife.

    Parameters
    ----------
    confidence_threshold : float
        Minimum MegaDetector confidence to keep a box (default 0.20).
        The official recommendation is 0.2; raise to 0.5 to reduce false
        positives at the cost of missing some real animals.
    device : str
        'cpu' or 'cuda'.  CPU is the intended deployment target.
    models_dir : Path | str
        Directory where PytorchWildlife caches downloaded weights.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.20,
        device: str = "cpu",
        models_dir: str | Path = "models",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._model = None   # lazy
        self._load_error: str | None = None  # set once on permanent failure

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load MegaDetector v6 (idempotent, thread-safe)."""
        if self._model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        with _load_lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                raise RuntimeError(self._load_error)

            logger.info(
                "Loading MegaDetector v6 … "
                "(first run downloads ~140 MB from HuggingFace)"
            )
            try:
                import os
                import torch  # noqa: F401
                os.environ.setdefault(
                    "TORCH_HOME", str(self.models_dir / "torch_hub")
                )
                from PytorchWildlife.models import detection as pw_det  # type: ignore

                self._model = self._load_megadetector(pw_det)
                # Warm-up pass to force JIT compilation on first call
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                self._run_inference(dummy)
                logger.info("MegaDetector v6 ready on %s", self.device.upper())
            except ImportError as exc:
                msg = (
                    "PytorchWildlife is not installed.  "
                    "Run:  pip install PytorchWildlife torch torchvision"
                )
                self._load_error = msg
                raise RuntimeError(msg) from exc
            except Exception as exc:
                msg = f"Failed to load MegaDetector v6: {exc}"
                self._load_error = msg
                raise RuntimeError(msg) from exc

    # ------------------------------------------------------------------
    def _load_megadetector(self, pw_det):
        """
        Load MDv6, using the locally cached weights file when available.

        PytorchWildlife has a bug where it checks for 'MDV6b-yolov9-c.pt'
        but wget saves the file as 'MDV6-yolov9-c.pt', so the cache check
        always fails and it re-downloads.  We find the file ourselves and
        pass it directly via the weights= parameter to bypass the download.
        """
        import os
        import torch

        checkpoints_dir = Path(torch.hub.get_dir()) / "checkpoints"
        # Look for any variant of the cached filename
        candidates = [
            checkpoints_dir / "MDV6-yolov9-c.pt",
            checkpoints_dir / "MDV6b-yolov9-c.pt",
            checkpoints_dir / "MDV6-yolov9-c (1).pt",
        ]
        cached = next((p for p in candidates if p.exists()), None)

        if cached:
            logger.info("Loading MegaDetector from cache: %s", cached.name)
            return pw_det.MegaDetectorV6(
                weights=str(cached),
                version="MDV6-yolov9-c",
                device=self.device,
                pretrained=False,
            )

        # First run — download from Zenodo (~50 MB)
        logger.info("Downloading MegaDetector weights (one-time, ~50 MB)…")
        return pw_det.MegaDetectorV6(
            version="MDV6-yolov9-c",
            device=self.device,
            pretrained=True,
        )

    # ------------------------------------------------------------------
    def _run_inference(self, bgr: np.ndarray) -> list[AnimalBox]:
        """Internal – run one frame through MDv6 and parse results."""
        # PytorchWildlife ≥1.3 expects an RGB numpy array
        rgb = bgr[:, :, ::-1].copy()

        try:
            result = self._model.single_image_detection(
                rgb,
                det_conf_thres=self.confidence_threshold,
            )
        except Exception as exc:
            logger.warning("MegaDetector inference error: %s", exc)
            return []

        boxes: list[AnimalBox] = []
        if not result:
            return boxes

        dets = result.get("detections")
        if dets is None:
            return boxes

        # PytorchWildlife returns a `supervision.Detections` object
        if not hasattr(dets, "xyxy") or len(dets.xyxy) == 0:
            return boxes

        h, w = bgr.shape[:2]
        for i, xyxy in enumerate(dets.xyxy):
            class_id   = int(dets.class_id[i]) if hasattr(dets, "class_id") else 0
            confidence = float(dets.confidence[i]) if hasattr(dets, "confidence") else 1.0

            # Keep animals (0) and people (1); ignore vehicles (2)
            if class_id == _CLASS_ANIMAL:
                label = "animal"
            elif class_id == _CLASS_PERSON:
                label = "person"
            else:
                continue

            if confidence < self.confidence_threshold:
                continue

            boxes.append(AnimalBox(
                x1=float(xyxy[0]) / w,
                y1=float(xyxy[1]) / h,
                x2=float(xyxy[2]) / w,
                y2=float(xyxy[3]) / h,
                confidence=confidence,
                label=label,
            ))

        return boxes

    # ------------------------------------------------------------------
    def detect(self, bgr: np.ndarray) -> list[AnimalBox]:
        """
        Detect animals in a BGR frame.

        Parameters
        ----------
        bgr : np.ndarray
            OpenCV BGR image, uint8, any resolution.

        Returns
        -------
        list[AnimalBox]
            Detected animal boxes with normalised coordinates.
            Empty list when no animals are found or confidence is too low.
        """
        self._load()
        return self._run_inference(bgr)

    # ------------------------------------------------------------------
    def crop(self, bgr: np.ndarray, box: AnimalBox) -> np.ndarray:
        """
        Return a tight crop of the animal region.

        Adds a small padding (5 % of box side) so the classifier sees a
        little context around the animal.
        """
        h, w = bgr.shape[:2]
        pad_x = (box.x2 - box.x1) * 0.05
        pad_y = (box.y2 - box.y1) * 0.05

        x1 = max(0, int((box.x1 - pad_x) * w))
        y1 = max(0, int((box.y1 - pad_y) * h))
        x2 = min(w, int((box.x2 + pad_x) * w))
        y2 = min(h, int((box.y2 + pad_y) * h))

        if x2 <= x1 or y2 <= y1:
            return bgr  # degenerate – return full frame
        return bgr[y1:y2, x1:x2]
