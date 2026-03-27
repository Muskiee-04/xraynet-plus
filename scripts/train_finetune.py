#!/usr/bin/env python3
"""
Fine-tune XRAYNET+ 4-class EfficientNet-B0 on folder-organized chest X-rays.

Layout::

    DATA_ROOT/
      train/<ClassFolder>/*.png
      val/<ClassFolder>/*.png   # optional

If `val/` is missing, a stratified split is taken from `train/` (see --val-split).

Examples::

    python scripts/train_finetune.py --data-dir data/xray_finetune
    python scripts/train_finetune.py --data-dir /path/to/cxr --epochs 30 --freeze-backbone 5

Checkpoint is written for `TorchCXRInference`: models/saved/xraynet_plus.pth (or --output).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from src.data.finetune_dataset import (  # noqa: E402
    FinetunePathDataset,
    build_train_val_lists,
    class_weights_from_samples,
)
from src.models.cxr_classifier import CLASS_NAMES, build_efficientnet_cxr  # noqa: E402
from src.training.loops import evaluate, set_backbone_requires_grad, train_one_epoch  # noqa: E402


def _load_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XRAYNET+ 4-class CXR model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root with train/ and optional val/ subfolders",
    )
    parser.add_argument("--config", type=str, default=os.path.join(_ROOT, "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--val-split", type=float, default=None, help="If no val/, fraction for stratified val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-backbone",
        type=int,
        default=0,
        metavar="N",
        help="For first N epochs, train classifier head only (backbone frozen)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional .pth to load (e.g. models/saved/xraynet_plus.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output checkpoint path (default: models/saved/xraynet_plus.pth)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest on macOS/Windows)",
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use inverse-frequency class weights in loss (imbalanced data)",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random backbone init (no ImageNet download). Use if offline or SSL/cert issues.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    tcfg = cfg.get("training", {})
    dcfg = cfg.get("data", {})

    data_dir = args.data_dir or os.environ.get("XRAYNET_DATA_DIR") or dcfg.get("finetune_data_root")
    if not data_dir:
        data_dir = os.path.join(_ROOT, "data", "xray_finetune")
    data_dir = os.path.abspath(data_dir)

    epochs = args.epochs if args.epochs is not None else int(tcfg.get("epochs", 40))
    batch_size = args.batch_size if args.batch_size is not None else int(tcfg.get("batch_size", 16))
    lr = args.lr if args.lr is not None else float(tcfg.get("learning_rate", 1e-4))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(tcfg.get("weight_decay", 0.01))
    val_split = args.val_split if args.val_split is not None else float(dcfg.get("val_split", 0.15))

    output_path = args.output or os.path.join(_ROOT, "models", "saved", "xraynet_plus.pth")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    train_samples, val_samples = build_train_val_lists(data_dir, val_split=val_split, seed=args.seed)
    if not train_samples:
        print(f"No training samples under {os.path.join(data_dir, 'train')}.")
        print("Expected: DATA_ROOT/train/<ClassName>/*.png  (classes: Tuberculosis, Pneumonia, COVID-19, No Findings)")
        sys.exit(1)
    if not val_samples:
        print("Warning: empty validation set; metrics will not be meaningful. Add val/ or lower --val-split.")
        sys.exit(1)

    num_classes = len(CLASS_NAMES)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}  Classes: {CLASS_NAMES}")

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8, fill=0),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = FinetunePathDataset(train_samples, train_tf)
    val_ds = FinetunePathDataset(val_samples, val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pretrained = bool(cfg.get("model", {}).get("pretrained", True)) and not args.no_pretrained
    model = build_efficientnet_cxr(num_classes=num_classes, pretrained_backbone=use_pretrained)
    model.to(device)
    if not use_pretrained:
        print("Using randomly initialized backbone (--no-pretrained or config model.pretrained: false).")

    if args.resume and os.path.isfile(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Loaded weights from {args.resume}")

    if args.freeze_backbone > 0:
        set_backbone_requires_grad(model, False)
        for p in model.classifier.parameters():
            p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    if args.class_weights:
        w = class_weights_from_samples(train_samples, num_classes)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
        print("Class weights:", dict(zip(CLASS_NAMES, w)))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_acc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        if args.freeze_backbone > 0 and epoch == args.freeze_backbone + 1:
            set_backbone_requires_grad(model, True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr * 0.5,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - epoch + 1, 1))
            print(f"Epoch {epoch}: unfrozen full model, lr scaled to {lr * 0.5}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, corr, tot = evaluate(model, val_loader, criterion, device, num_classes)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch}/{epochs}  train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
        )
        for i, name in enumerate(CLASS_NAMES):
            if tot[i] > 0:
                print(f"    {name}: {corr[i]}/{tot[i]} correct")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            payload = {
                "model_state_dict": model.state_dict(),
                "class_names": list(CLASS_NAMES),
                "val_acc": val_acc,
                "epoch": epoch,
                "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "data_dir": data_dir,
            }
            torch.save(payload, output_path)
            print(f"  -> saved best checkpoint ({val_acc:.4f}) to {output_path}")

    # Save training log next to checkpoint
    log_path = output_path.replace(".pth", "_history.yaml")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            yaml.dump({"history": history, "best_val_acc": best_val_acc}, f)
        print(f"Wrote {log_path}")
    except OSError:
        pass

    print(f"Done. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
