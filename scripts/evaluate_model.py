#!/usr/bin/env python3
"""
Validation metrics on a labeled folder tree (same layout as fine-tuning val/).

  python scripts/evaluate_model.py --data-dir data/xray_finetune --split val
  python scripts/evaluate_model.py --weights models/saved/xraynet_plus.pth --data-dir data/xray_finetune --split val

Prints accuracy, per-class precision/recall/F1, confusion matrix, and macro F1.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from src.data.finetune_dataset import FinetunePathDataset, collect_samples  # noqa: E402
from src.models.cxr_classifier import CLASS_NAMES, build_efficientnet_cxr  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--weights", type=str, default=None, help="Default: try xraynet_plus.pth")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="Random backbone if weights load fails")
    args = p.parse_args()

    split_dir = os.path.join(os.path.abspath(args.data_dir), args.split)
    samples = collect_samples(split_dir)
    if not samples:
        print(f"No images under {split_dir}")
        sys.exit(1)

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = FinetunePathDataset(samples, tfm)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet_cxr(
        num_classes=len(CLASS_NAMES),
        pretrained_backbone=not args.no_pretrained,
    )
    model.to(device)
    model.eval()

    wpath = args.weights or os.path.join(_ROOT, "models", "saved", "xraynet_plus.pth")
    if os.path.isfile(wpath):
        try:
            ckpt = torch.load(wpath, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(wpath, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state, strict=True)
            print(f"Loaded {wpath}")
        except RuntimeError as e:
            print(f"Could not load {wpath}: {e}")
            sys.exit(1)
    else:
        print(f"No weights at {wpath}; using initialized backbone only.")

    y_true: list[int] = []
    y_pred: list[int] = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = 0
    for x, y in tqdm(loader, desc="eval"):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        pred = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\nSamples: {n}  Loss: {total_loss / max(n, 1):.4f}  Accuracy: {acc:.4f}\n")
    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    print(cm)
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
