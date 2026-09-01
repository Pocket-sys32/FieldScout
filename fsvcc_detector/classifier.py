"""
Species classifier — three backends, selected automatically.

Priority order
──────────────
1. Phase-2 ONNX  – fine-tuned EfficientNet-B0 (best accuracy, requires training)
2. SpeciesNet    – Google's camera-trap classifier (great IR support, requires
                   a free Kaggle API token on first download)
3. CLIP          – OpenAI zero-shot fallback (no setup, decent accuracy)

SpeciesNet is specifically trained on Wildlife Insights camera-trap data,
including millions of night-IR frames, making it much better suited to
Bushnell footage than CLIP.

SpeciesNet setup (one-time, free)
──────────────────────────────────
1. Create a free account at https://www.kaggle.com
2. Go to Account → Settings → API → Create New Token
3. Download kaggle.json and save it to  C:\\Users\\<you>\\.kaggle\\kaggle.json
4. The model (~500 MB) downloads automatically on first run and is cached.

If kaggle.json is not present the app falls back to CLIP automatically
and logs a warning.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .species import SPECIES_LIST, HUMAN_ENTRY, _BIRD_CLASS_HINTS, all_prompts, by_key

logger = logging.getLogger(__name__)

_load_lock = threading.Lock()

# ── Kaggle model identifier for SpeciesNet ────────────────────────────────────
_SPECIESNET_MODEL = "google/speciesnet/pyTorch/v4.0.2a/1"


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    species_key: str
    common_name: str
    scientific_name: str
    confidence: float
    top3: list[tuple[str, float]]
    backend: str       # 'speciesnet' | 'clip' | 'onnx' | 'none'


# ── SpeciesNet backend ────────────────────────────────────────────────────────

class _SpeciesNetClassifier:
    """
    Google SpeciesNet classifier — trained specifically on camera-trap imagery
    (day + night IR).  Downloads ~500 MB from Kaggle on first use.

    Designed to receive the full video frame plus the MegaDetector bounding
    box so SpeciesNet can do its own crop/preprocess (same pipeline it was
    trained with).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model     = None
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_error:
            raise RuntimeError(self._load_error)
        with _load_lock:
            if self._model is not None:
                return
            if self._load_error:
                raise RuntimeError(self._load_error)
            logger.info(
                "Loading SpeciesNet … "
                "(first run downloads ~500 MB from Kaggle)"
            )
            try:
                from speciesnet import SpeciesNetClassifier  # type: ignore
                import kagglehub  # type: ignore  (installed with speciesnet)

                model_dir = kagglehub.model_download(_SPECIESNET_MODEL)
                self._model = SpeciesNetClassifier(
                    model_name=model_dir,
                    device=self.device,
                )
                logger.info("SpeciesNet ready on %s", self.device.upper())
            except Exception as exc:
                msg = str(exc)
                # Friendly message for the most common failure
                if "kaggle" in msg.lower() or "credentials" in msg.lower() or "401" in msg:
                    msg = (
                        "SpeciesNet requires a Kaggle API token.  "
                        "See Settings → 'SpeciesNet setup' in the README, or "
                        "place kaggle.json in C:\\Users\\<you>\\.kaggle\\  "
                        "Falling back to CLIP."
                    )
                self._load_error = msg
                raise RuntimeError(msg) from exc

    def classify(
        self,
        bgr_crop: np.ndarray,
        full_frame: np.ndarray | None = None,
        box_norm: tuple[float, float, float, float] | None = None,
    ) -> ClassificationResult:
        """
        Classify an animal.

        Parameters
        ----------
        bgr_crop   : tight crop from MegaDetector (always used as fallback)
        full_frame : full video frame (BGR) — preferred input for SpeciesNet
        box_norm   : (x1, y1, x2, y2) normalised bounding box from MegaDetector
        """
        from PIL import Image  # type: ignore

        self._load()

        # Build PIL image and optional BBox for SpeciesNet
        if full_frame is not None and box_norm is not None:
            rgb_full = full_frame[:, :, ::-1].copy()
            pil_img  = Image.fromarray(rgb_full)
            from speciesnet import BBox  # type: ignore
            x1, y1, x2, y2 = box_norm
            bbox = BBox(
                xmin=x1,
                ymin=y1,
                width=max(0.0, x2 - x1),
                height=max(0.0, y2 - y1),
            )
            preprocessed = self._model.preprocess(pil_img, bboxes=[bbox])
        else:
            rgb_crop = bgr_crop[:, :, ::-1].copy()
            pil_img  = Image.fromarray(rgb_crop)
            preprocessed = self._model.preprocess(pil_img)

        result = self._model.predict("frame", preprocessed)

        if "failures" in result or "classifications" not in result:
            return _unknown_result("speciesnet")

        classes = result["classifications"]["classes"]
        scores  = result["classifications"]["scores"]

        # SpeciesNet label format: "uuid;class;order;family;genus;species;common"
        # Parse each label into (scientific_name, common_name)
        parsed = [_parse_speciesnet_label(c) for c in classes]
        logger.debug("SpeciesNet top-5: %s", [(p, round(s,3)) for p, s in zip(parsed, scores)])

        species_key, confidence = _match_taxa_parsed(parsed, scores)
        sp = by_key(species_key)

        top3 = [
            (sp_name or common or raw, float(s))
            for (sp_name, common, raw), s in zip(parsed[:3], scores[:3])
        ]

        return ClassificationResult(
            species_key    = species_key,
            common_name    = sp["common_name"]    if sp else (parsed[0][1] or parsed[0][0] or classes[0]),
            scientific_name= sp["scientific_name"] if sp else (parsed[0][0] or classes[0]),
            confidence     = confidence,
            top3           = top3,
            backend        = "speciesnet",
        )


# ── CLIP zero-shot fallback ───────────────────────────────────────────────────

class _ClipClassifier:
    """CLIP ViT-B/32 zero-shot species classifier (fallback when no Kaggle token)."""

    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, device: str = "cpu", models_dir: str | Path = "models") -> None:
        self.device = device
        self.models_dir = Path(models_dir)
        self._model      = None
        self._processor  = None
        self._text_feats = None

    def _load(self) -> None:
        if self._model is not None:
            return
        with _load_lock:
            if self._model is not None:
                return
            logger.info(
                "Loading CLIP ViT-B/32 … "
                "(first run downloads ~340 MB from HuggingFace)"
            )
            try:
                import os
                import torch  # noqa: F401
                from transformers import CLIPModel, CLIPProcessor  # type: ignore

                cache = str(self.models_dir / "hf_cache")

                def _load_clip(offline: bool):
                    orig = os.environ.get("HF_HUB_OFFLINE")
                    try:
                        if offline:
                            os.environ["HF_HUB_OFFLINE"] = "1"
                        else:
                            os.environ.pop("HF_HUB_OFFLINE", None)
                        m = CLIPModel.from_pretrained(
                            self.MODEL_ID, cache_dir=cache,
                            local_files_only=offline,
                        ).to(self.device)
                        p = CLIPProcessor.from_pretrained(
                            self.MODEL_ID, cache_dir=cache,
                            local_files_only=offline,
                        )
                        return m, p
                    finally:
                        if orig is None:
                            os.environ.pop("HF_HUB_OFFLINE", None)
                        else:
                            os.environ["HF_HUB_OFFLINE"] = orig

                try:
                    self._model, self._processor = _load_clip(offline=True)
                except Exception:
                    logger.info("CLIP not in cache — downloading (~340 MB)…")
                    self._model, self._processor = _load_clip(offline=False)

                self._model.eval()
                self._precompute_text()
                logger.info("CLIP ready on %s", self.device.upper())
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is not installed.  Run: pip install transformers"
                ) from exc

    def _precompute_text(self) -> None:
        import torch
        prompts, indices = all_prompts()
        inputs = self._processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
            feats = _extract_tensor(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        n = len(SPECIES_LIST)
        averaged = torch.zeros(n, feats.shape[-1], device=self.device)
        counts   = torch.zeros(n, device=self.device)
        for i, sp_idx in enumerate(indices):
            averaged[sp_idx] += feats[i]
            counts[sp_idx]   += 1
        self._text_feats = averaged / counts.unsqueeze(1)

    def classify(
        self,
        bgr_crop: np.ndarray,
        full_frame: np.ndarray | None = None,
        box_norm: tuple | None = None,
    ) -> ClassificationResult:
        import torch
        from PIL import Image  # type: ignore

        self._load()
        rgb = bgr_crop[:, :, ::-1].copy()
        pil = Image.fromarray(rgb)
        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            img_feat = self._model.get_image_features(**inputs)
            img_feat = _extract_tensor(img_feat)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (100.0 * img_feat @ self._text_feats.T).softmax(dim=-1)
        probs = sims[0].cpu().numpy()
        order = probs.argsort()[::-1]
        top_idx = order[0]
        sp = SPECIES_LIST[top_idx]
        top3 = [
            (SPECIES_LIST[i]["common_name"], float(probs[i]))
            for i in order[:3]
        ]
        return ClassificationResult(
            species_key    = sp["key"],
            common_name    = sp["common_name"],
            scientific_name= sp["scientific_name"],
            confidence     = float(probs[top_idx]),
            top3           = top3,
            backend        = "clip",
        )


# ── Phase-2 custom ONNX backend ───────────────────────────────────────────────

class _OnnxClassifier:
    """Fine-tuned EfficientNet-B0 exported to ONNX after Phase-2 training."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _SIZE = 224

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._session   = None

    def _load(self) -> None:
        if self._session is not None:
            return
        with _load_lock:
            if self._session is not None:
                return
            import onnxruntime as ort  # type: ignore
            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
            logger.info("Custom ONNX classifier loaded: %s", self.model_path.name)

    def classify(
        self,
        bgr_crop: np.ndarray,
        full_frame: np.ndarray | None = None,
        box_norm: tuple | None = None,
    ) -> ClassificationResult:
        import cv2
        self._load()
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self._SIZE, self._SIZE))
        x   = (rgb.astype(np.float32) / 255.0 - self._MEAN) / self._STD
        inp = x.transpose(2, 0, 1)[None]
        logits = self._session.run(None, {"input": inp})[0][0]
        probs  = _softmax(logits)
        order  = probs.argsort()[::-1]
        sp = SPECIES_LIST[order[0]]
        top3 = [
            (SPECIES_LIST[i]["common_name"], float(probs[i]))
            for i in order[:3]
        ]
        return ClassificationResult(
            species_key    = sp["key"],
            common_name    = sp["common_name"],
            scientific_name= sp["scientific_name"],
            confidence     = float(probs[order[0]]),
            top3           = top3,
            backend        = "onnx",
        )


# ── Public factory ─────────────────────────────────────────────────────────────

class SpeciesClassifier:
    """
    Auto-selects the best available backend:
      Phase-2 ONNX  (if custom_classifier_path is set and file exists)
      → SpeciesNet  (if Kaggle token is present)
      → CLIP        (always available, no setup needed)
    """

    def __init__(
        self,
        device: str = "cpu",
        models_dir: str | Path = "models",
        custom_classifier_path: str | Path = "",
    ) -> None:
        self.device    = device
        self.models_dir = Path(models_dir)

        onnx_path = Path(custom_classifier_path) if custom_classifier_path else None
        if onnx_path and onnx_path.exists():
            logger.info("Using Phase-2 ONNX classifier: %s", onnx_path)
            self._primary: _OnnxClassifier | _SpeciesNetClassifier | _ClipClassifier = _OnnxClassifier(onnx_path)
            self._fallback = None
        else:
            # Try SpeciesNet; fall back to CLIP if Kaggle isn't configured
            self._primary  = _SpeciesNetClassifier(device=device)
            self._fallback = _ClipClassifier(device=device, models_dir=self.models_dir)

    def classify(
        self,
        bgr_crop: np.ndarray,
        full_frame: np.ndarray | None = None,
        box_norm: tuple[float, float, float, float] | None = None,
    ) -> ClassificationResult:
        """
        Classify an animal detection.

        Parameters
        ----------
        bgr_crop   : cropped animal image from MegaDetector (required)
        full_frame : full video frame — passed to SpeciesNet for best accuracy
        box_norm   : (x1, y1, x2, y2) normalised bounding box
        """
        h, w = bgr_crop.shape[:2]
        if h < 10 or w < 10:
            return _unknown_result("none")

        try:
            return self._primary.classify(bgr_crop, full_frame=full_frame, box_norm=box_norm)
        except Exception as exc:
            if self._fallback is not None:
                logger.warning(
                    "Primary classifier failed (%s) — using CLIP fallback.", exc
                )
                return self._fallback.classify(bgr_crop)
            raise


# ── Helpers ───────────────────────────────────────────────────────────────────

# Build reverse map: lower-case taxon string -> species key (built once at import)
_TAXON_TO_KEY: dict[str, str] = {}
for _sp in SPECIES_LIST:
    for _taxon in _sp["speciesnet_taxa"]:
        _TAXON_TO_KEY[_taxon.lower()] = _sp["key"]
_TAXON_TO_KEY["homo sapiens"] = "human"


def _parse_speciesnet_label(label: str) -> tuple[str, str, str]:
    """
    Parse a SpeciesNet label string into (scientific_name, common_name, raw).

    SpeciesNet format: "uuid;class;order;family;genus;species;common_name"
    Fields may be empty (e.g. blank/vehicle entries).
    Returns (scientific_name, common_name, raw_label).
    """
    parts = label.split(";")
    # parts: [uuid, class, order, family, genus, species, common]
    genus   = parts[4].strip() if len(parts) > 4 else ""
    species = parts[5].strip() if len(parts) > 5 else ""
    common  = parts[6].strip() if len(parts) > 6 else ""
    sci = f"{genus} {species}".strip() if genus else ""
    return sci, common, label


def _match_taxa_parsed(
    parsed: list[tuple[str, str, str]], scores: list[float]
) -> tuple[str, float]:
    """Map parsed SpeciesNet labels to one of our species keys."""
    for (sci, common, raw), score in zip(parsed, scores):
        raw_lower = raw.lower()

        if not sci and not common:
            continue
        if any(x in raw_lower for x in ("blank", "vehicle", "empty")):
            continue

        # Human — map directly instead of skipping
        if any(x in raw_lower for x in ("human", "homo sapiens", "homo")):
            return "human", float(score)

        # Try scientific name match first (most reliable)
        if sci:
            c = sci.lower()
            if c in _TAXON_TO_KEY:
                return _TAXON_TO_KEY[c], float(score)
            for taxon, key in _TAXON_TO_KEY.items():
                if c.startswith(taxon) or taxon.startswith(c.split()[0]):
                    return key, float(score)

        # Generic bird fallback for unmatched avian taxa
        if any(hint in raw_lower or hint in (sci or "").lower() or hint in (common or "").lower()
               for hint in _BIRD_CLASS_HINTS):
            if "callipepla" not in raw_lower and "zonotrichia" not in raw_lower:
                return "bird", float(score)

        # Try common name match as fallback
        if common:
            c = common.lower()
            for sp in SPECIES_LIST:
                if sp["common_name"].lower() in c or c in sp["common_name"].lower():
                    return sp["key"], float(score)

    return "unknown", float(scores[0]) if scores else 0.0


def _taxa_to_common(taxon: str) -> str:
    """Best-effort scientific name → common name lookup."""
    t = taxon.lower().strip()
    if t in _TAXON_TO_KEY:
        sp = by_key(_TAXON_TO_KEY[t])
        if sp:
            return sp["common_name"]
    return taxon.title()


def _unknown_result(backend: str) -> ClassificationResult:
    return ClassificationResult(
        species_key="unknown", common_name="Unknown",
        scientific_name="Unknown", confidence=0.0,
        top3=[], backend=backend,
    )


def _extract_tensor(output):
    """Handle newer transformers returning ModelOutput instead of a plain tensor."""
    import torch
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    return output[0]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()
