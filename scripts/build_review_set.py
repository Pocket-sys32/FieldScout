"""
scripts/build_review_set.py
════════════════════════════
Extract animal crops from a folder of .mov files and organise them into
per-species subdirectories so volunteers can verify (or correct) the AI's
guesses using any image viewer.

The resulting folder structure is suitable for direct use as a PyTorch
ImageFolder dataset in scripts/train_classifier.py.

Output structure
────────────────
review_crops/
  raccoon/             ← AI predicted "Raccoon"
    IMG_0012_f0003.jpg
    IMG_0015_f0009.jpg
  coyote/
    IMG_0018_f0002.jpg
  _needs_review/       ← low-confidence or tied predictions
    IMG_0020_f0005.jpg
    ...

Volunteer workflow
──────────────────
1. Run this script on the archive of .mov files.
2. Open review_crops/ in Windows Explorer or an image viewer.
3. Move any mis-labelled images to the correct species folder
   (or a 'delete' folder for non-animals / bad crops).
4. Once enough images are verified, run scripts/train_classifier.py
   to fine-tune the classifier (Phase 2).

Usage
─────
    python scripts/build_review_set.py  --videos /path/to/movs
                                        --output  review_crops
                                        [--threshold 0.65]
                                        [--sample-rate 2.0]
                                        [--max-frames 60]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from fsvcc_detector.aggregate import Aggregator, FrameRecord
from fsvcc_detector.classifier import SpeciesClassifier
from fsvcc_detector.config import cfg
from fsvcc_detector.detector import AnimalDetector
from fsvcc_detector.species import by_key
from fsvcc_detector.video import extract_frames, get_video_meta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and organise animal crops for volunteer review."
    )
    parser.add_argument(
        "--videos", required=True, metavar="DIR",
        help="Folder containing .mov files.",
    )
    parser.add_argument(
        "--output", default="review_crops", metavar="DIR",
        help="Output root directory for crops (default: review_crops/).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65, metavar="FLOAT",
        help="Confidence below this value → _needs_review/ folder (default 0.65).",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=2.0, metavar="FPS",
        help="Frames to extract per second (default 2.0 – more coverage for review).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=60, metavar="N",
        help="Max frames per video (default 60).",
    )
    parser.add_argument(
        "--min-box-area", type=float, default=0.005, metavar="FRAC",
        help="Minimum normalised bounding-box area to save (default 0.005 = 0.5%%).",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _EXTS = ("*.mov", "*.MOV", "*.avi", "*.AVI", "*.mp4", "*.MP4")
    mov_files: list[Path] = []
    for pat in _EXTS:
        mov_files.extend(videos_dir.glob(pat))
    mov_files = sorted(set(mov_files))

    if not mov_files:
        logger.error("No supported video files (.mov/.avi/.mp4) found in %s", videos_dir)
        sys.exit(1)

    logger.info("Found %d video(s) in %s", len(mov_files), videos_dir)
    logger.info("Output → %s", output_dir)

    detector   = AnimalDetector(confidence_threshold=cfg.megadetector_confidence,
                                device=cfg.device, models_dir=cfg.models_dir_resolved)
    classifier = SpeciesClassifier(device=cfg.device, models_dir=cfg.models_dir_resolved,
                                   custom_classifier_path=cfg.custom_classifier_path)

    total_crops = 0

    for v_idx, video_path in enumerate(mov_files, 1):
        logger.info("[%d/%d]  %s", v_idx, len(mov_files), video_path.name)

        try:
            for vframe in extract_frames(video_path,
                                         sample_rate=args.sample_rate,
                                         max_frames=args.max_frames):
                boxes = detector.detect(vframe.image)
                for box in boxes:
                    if box.area < args.min_box_area:
                        continue  # skip tiny detections

                    crop = detector.crop(vframe.image, box)
                    clf  = classifier.classify(crop)

                    # Determine output subdirectory
                    low_conf = clf.confidence < args.threshold
                    if low_conf or clf.species_key == "unknown":
                        subdir = output_dir / "_needs_review"
                    else:
                        sp = by_key(clf.species_key)
                        subdir = output_dir / (sp["key"] if sp else "unknown")

                    subdir.mkdir(parents=True, exist_ok=True)
                    fname = (
                        f"{video_path.stem}"
                        f"_f{vframe.index:04d}"
                        f"_{clf.species_key}"
                        f"_{int(clf.confidence * 100):02d}pct"
                        ".jpg"
                    )
                    out_path = subdir / fname
                    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    total_crops += 1

        except Exception as exc:
            logger.error("Error processing %s: %s", video_path.name, exc)
            continue

    logger.info("─" * 60)
    logger.info("Done.  %d crop(s) saved to %s", total_crops, output_dir)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Open %s in File Explorer.", output_dir)
    logger.info("  2. Move misidentified images to the correct species folder.")
    logger.info("  3. Move junk / non-animals to a 'delete' folder (then delete it).")
    logger.info("  4. Run: python scripts/train_classifier.py --data %s", output_dir)
    logger.info("")
    _print_class_counts(output_dir)


def _print_class_counts(root: Path) -> None:
    """Print a quick summary of how many crops are in each folder."""
    logger.info("Current crop counts per species:")
    for d in sorted(root.iterdir()):
        if d.is_dir():
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            bar = "█" * min(n // 5, 40)
            logger.info("  %-30s  %4d  %s", d.name, n, bar)


if __name__ == "__main__":
    main()
