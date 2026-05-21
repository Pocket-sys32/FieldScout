"""
scripts/train_classifier.py  ─  Phase-2 fine-tuning
═════════════════════════════════════════════════════
Fine-tune EfficientNet-B0 on the verified crops produced by
build_review_set.py, then export an ONNX model that the main app can
use in place of the CLIP zero-shot backend.

Recommended hardware
────────────────────
• Google Colab (free tier, T4 GPU) – training ~200 imgs/class ≈ 5-10 min
• Kaggle Notebooks (P100 GPU) – same dataset ≈ 3-5 min
• Local NVIDIA GPU – fastest option

This script does NOT require a GPU (it falls back to CPU) but training
will be slow (~30-60 min) on a modern laptop CPU for a typical dataset.

Usage
─────
    # Install Phase-2 extras first (not in the main requirements.txt):
    pip install timm onnx onnxruntime

    python scripts/train_classifier.py \
        --data  review_crops/ \
        --epochs 30 \
        --batch  32 \
        --output models/fsvcc_classifier.onnx

After training, open the app's Settings and set
"Custom ONNX classifier path" to  models/fsvcc_classifier.onnx
The app will switch from CLIP to your trained model automatically.

Expected accuracy (Phase 2)
────────────────────────────
Dataset size     Top-1 accuracy (estimate)
─────────────────────────────────────────
 50 crops/class  ~82–88 %
200 crops/class  ~88–93 %
500 crops/class  ~91–96 %

Hard cases (coyote vs gray fox, IR night footage) may still need
volunteer review flags even at high overall accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Class order must match SPECIES_LIST ──────────────────────────────────────
SPECIES_CLASSES = [
    "beaver", "bobcat", "coyote", "striped_skunk", "opossum",
    "deer", "gray_fox", "raccoon", "desert_cottontail",
    "squirrel", "california_quail", "golden_crowned_sparrow", "river_otter",
]

# ImageNet normalisation (EfficientNet pre-trained)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_INPUT_SIZE    = 224


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B0 on verified wildlife crops."
    )
    parser.add_argument(
        "--data", required=True, metavar="DIR",
        help="Root of the verified crop folder (ImageFolder structure).",
    )
    parser.add_argument(
        "--output", default="models/fsvcc_classifier.onnx", metavar="PATH",
        help="Where to write the ONNX model (default: models/fsvcc_classifier.onnx).",
    )
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data to use for validation (default 0.15).")
    parser.add_argument("--workers",   type=int,   default=2)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation (useful for tiny datasets).")
    args = parser.parse_args()

    data_dir   = Path(args.data)
    output_path = Path(args.output)

    # Validate data directory
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, models, transforms  # type: ignore
    except ImportError:
        logger.error("PyTorch / torchvision not installed.  Run: pip install torch torchvision")
        sys.exit(1)

    try:
        import timm  # type: ignore
    except ImportError:
        logger.error("timm not installed.  Run: pip install timm")
        sys.exit(1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

    # ── Data transforms ──────────────────────────────────────────────
    if args.no_augment:
        train_tf = _base_transform()
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(_INPUT_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            transforms.RandomGrayscale(p=0.15),   # simulate night IR
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    val_tf = _base_transform()

    # ── Dataset ──────────────────────────────────────────────────────
    full_ds = datasets.ImageFolder(str(data_dir))
    _log_class_distribution(full_ds)

    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds_raw, val_ds_raw = random_split(full_ds, [n_train, n_val],
                                             generator=torch.Generator().manual_seed(args.seed))

    train_ds = _TransformDataset(train_ds_raw, train_tf)
    val_ds   = _TransformDataset(val_ds_raw,   val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    n_classes = len(full_ds.classes)
    logger.info("Classes: %s", full_ds.classes)
    logger.info("Train: %d  Val: %d  Classes: %d", n_train, n_val, n_classes)

    # ── Model ────────────────────────────────────────────────────────
    model = timm.create_model(
        "efficientnet_b0", pretrained=True, num_classes=n_classes
    )
    model = model.to(device)

    # Freeze backbone for first 5 epochs, then unfreeze for fine-tuning
    _set_frozen(model, frozen=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 10,   # warm-up head only
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc  = 0.0
    best_ckpt     = output_path.with_suffix(".pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after epoch 5
        if epoch == 6:
            _set_frozen(model, frozen=False)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            logger.info("Backbone unfrozen — full fine-tuning begins.")

        t0 = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion,
                                           optimizer, device, training=True)
        val_loss,   val_acc   = _run_epoch(model, val_loader,   criterion,
                                           None,      device, training=False)
        scheduler.step()

        logger.info(
            "Epoch %3d/%d  train loss %.4f  acc %.1f%%  │  "
            "val loss %.4f  acc %.1f%%  (%.1fs)",
            epoch, args.epochs,
            train_loss, train_acc * 100,
            val_loss,   val_acc   * 100,
            time.time() - t0,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "classes":    full_ds.classes,
                "val_acc":    val_acc,
            }, best_ckpt)
            logger.info("  ✓ New best model saved (val_acc=%.2f%%)", val_acc * 100)

    logger.info("Training complete.  Best val accuracy: %.2f%%", best_val_acc * 100)

    # ── Export ONNX ──────────────────────────────────────────────────
    logger.info("Exporting ONNX model to %s …", output_path)
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, 3, _INPUT_SIZE, _INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    # Save class mapping alongside the ONNX so the app can verify alignment
    meta_path = output_path.with_suffix(".classes.json")
    with meta_path.open("w") as f:
        json.dump({"classes": ckpt["classes"], "val_acc": ckpt["val_acc"]}, f, indent=2)

    logger.info("ONNX export complete: %s", output_path)
    logger.info("Class mapping: %s", meta_path)
    logger.info("")
    logger.info("Next: open app Settings → set 'Custom ONNX classifier path' to:")
    logger.info("  %s", output_path.resolve())


# ── Training utilities ────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, device, training: bool):
    import torch
    model.train() if training else model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    ctx = torch.enable_grad if training else torch.no_grad
    with ctx():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(images)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += len(images)

    return total_loss / max(total, 1), correct / max(total, 1)


def _set_frozen(model, *, frozen: bool) -> None:
    """Freeze or unfreeze all layers except the final classifier head."""
    for name, param in model.named_parameters():
        is_head = "classifier" in name or "head" in name
        param.requires_grad = (not frozen) or is_head


def _base_transform():
    from torchvision import transforms  # type: ignore
    return transforms.Compose([
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _log_class_distribution(dataset) -> None:
    from collections import Counter
    counts = Counter(dataset.targets)
    logger.info("Class distribution:")
    for cls, idx in sorted(dataset.class_to_idx.items(), key=lambda x: x[1]):
        n = counts.get(idx, 0)
        bar = "█" * min(n // 5, 40)
        logger.info("  %-32s  %4d  %s", cls, n, bar)


class _TransformDataset:
    """Wrap a Subset so we can apply per-split transforms."""

    def __init__(self, subset, transform):
        self._subset    = subset
        self._transform = transform

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        img, label = self._subset[idx]
        if not hasattr(img, "convert"):
            from PIL import Image  # type: ignore
            import numpy as np
            img = Image.fromarray(img)
        return self._transform(img), label


if __name__ == "__main__":
    main()
