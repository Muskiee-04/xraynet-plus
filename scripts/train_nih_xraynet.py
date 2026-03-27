#!/usr/bin/env python3
"""
Fine-tune XRAYNET+ 4-class head on NIH ChestX-ray14 (Kaggle: nih-chest-xrays/data).

Download data first — see data/nih_chest_xray/KAGGLE_SETUP.txt

Example::

    python scripts/train_nih_xraynet.py --nih-root data/nih_chest_xray --epochs 5 --max-per-class 2000

Label mapping (see src/data/nih_xraynet_dataset.py):
  Pneumonia → Pneumonia
  No Finding (only) → No Findings
  Infiltration (no Pneumonia) → Tuberculosis *proxy* (not true TB)
  Optional --covid-proxy consolidation → COVID-19 *proxy* (NIH predates COVID)

COVID-19 has no real labels in NIH; default is --covid-proxy none (class may stay weak).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
import yaml
from torchvision import transforms

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from src.data.finetune_dataset import (  # noqa: E402
    FinetunePathDataset,
    class_weights_from_samples,
)
from src.data.nih_xraynet_dataset import (  # noqa: E402
    build_samples_from_nih,
    print_class_histogram,
)
from src.models.cxr_classifier import CLASS_NAMES, build_efficientnet_cxr  # noqa: E402
from src.training.loops import evaluate, set_backbone_requires_grad, train_one_epoch  # noqa: E402


def _load_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    p = argparse.ArgumentParser(description="NIH ChestX-ray14 → XRAYNET+ fine-tune")
    p.add_argument(
        "--nih-root",
        type=str,
        default=None,
        help="Folder with Data_Entry_2017.csv, images/, list files (Kaggle unzip root)",
    )
    p.add_argument("--config", type=str, default=os.path.join(_ROOT, "config.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument(
        "--official-test-as-val",
        action="store_true",
        help="Train on train_val list, validate on official test_list (no random val split)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-backbone", type=int, default=0, metavar="N")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Default: models/saved/xraynet_nih_finetuned.pth",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap images per class (faster experiments; omit for full NIH)",
    )
    p.add_argument(
        "--tb-proxy",
        choices=("none", "infiltration"),
        default="infiltration",
        help="Map Infiltration→Tuberculosis index (weak proxy; none = skip those rows)",
    )
    p.add_argument(
        "--covid-proxy",
        choices=("none", "consolidation"),
        default="none",
        help="Map Consolidation→COVID-19 (weak; NIH has no true COVID labels)",
    )
    args = p.parse_args()

    cfg = _load_config(args.config)
    tcfg = cfg.get("training", {})
    dcfg = cfg.get("data", {})

    nih_root = args.nih_root or dcfg.get("nih_chest_xray_root") or os.path.join(_ROOT, "data", "nih_chest_xray")
    nih_root = os.path.abspath(nih_root)

    epochs = args.epochs if args.epochs is not None else int(tcfg.get("epochs", 15))
    batch_size = args.batch_size if args.batch_size is not None else int(tcfg.get("batch_size", 32))
    lr = args.lr if args.lr is not None else float(tcfg.get("learning_rate", 3e-4))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(tcfg.get("weight_decay", 0.01))

    output_path = args.output or os.path.join(_ROOT, "models", "saved", "xraynet_nih_finetuned.pth")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tb_proxy = args.tb_proxy  # type: ignore[assignment]
    covid_proxy = args.covid_proxy  # type: ignore[assignment]

    train_samples, val_samples = build_samples_from_nih(
        nih_root,
        val_fraction=args.val_fraction,
        seed=args.seed,
        tb_proxy=tb_proxy,
        covid_proxy=covid_proxy,
        max_per_class=args.max_per_class,
        official_test_as_val=args.official_test_as_val,
    )

    print_class_histogram(train_samples, "Train")
    print_class_histogram(val_samples, "Val")

    num_classes = len(CLASS_NAMES)
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
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
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

    if args.resume and os.path.isfile(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"Loaded {args.resume}")

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
        filter(lambda x: x.requires_grad, model.parameters()),
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - epoch + 1, 1)
            )
            print(f"Epoch {epoch}: full model unfrozen, lr={lr * 0.5}")

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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "nih_root": nih_root,
                    "tb_proxy": args.tb_proxy,
                    "covid_proxy": args.covid_proxy,
                },
                output_path,
            )
            print(f"  -> saved best to {output_path}")

    log_path = output_path.replace(".pth", "_history.yaml")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            yaml.dump({"history": history, "best_val_acc": best_val_acc}, f)
        print(f"Wrote {log_path}")
    except OSError:
        pass

    print(f"Done. Best val acc: {best_val_acc:.4f}")
    print("Copy to models/saved/xraynet_plus.pth to use as default in the Streamlit app, or set XRAYNET_WEIGHTS.")


if __name__ == "__main__":
    main()
