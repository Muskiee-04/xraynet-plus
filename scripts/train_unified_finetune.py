#!/usr/bin/env python3
"""
Unified fine-tune: NIH ChestX-ray14 + optional Kaggle TB + optional Kaggle COVID +
optional Mooney chest X-ray pneumonia (paultimothymooney/chest-xray-pneumonia).

Recommended (all three):
  python scripts/train_unified_finetune.py \\
    --nih-root data/nih_chest_xray \\
    --tb-root data/kaggle_tb_chest \\
    --covid-root data/kaggle_covid_chest \\
    --epochs 10 --class-weights --cap-per-class 8000

Mooney + COVID folder (no NIH), e.g. after unzipping Kaggle downloads:

  python scripts/train_unified_finetune.py \\
    --pneumonia-mooney-root data/mooney_chest_xray \\
    --covid-root data/alif_covid/dataset

See data/UNIFIED_FINETUNE_SETUP.txt for Kaggle download commands.

Output: models/saved/xraynet_unified_finetuned.pth
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

from src.data.finetune_dataset import FinetunePathDataset, class_weights_from_samples  # noqa: E402
from src.data.kaggle_folder_sources import print_kaggle_scan_hint  # noqa: E402
from src.data.unified_cxr_samples import (  # noqa: E402
    build_unified_train_val,
    print_class_histogram,
)
from src.models.cxr_classifier import CLASS_NAMES, build_efficientnet_cxr  # noqa: E402
from src.training.loops import evaluate, set_backbone_requires_grad, train_one_epoch  # noqa: E402


def _load_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _nih_root_if_ready(path: str | None) -> str | None:
    """Use NIH only when Data_Entry_2017.csv is actually present (skip empty config path)."""
    if not path:
        return None
    path = os.path.abspath(path)
    try:
        from src.data.nih_xraynet_dataset import _discover_nih_paths

        _discover_nih_paths(path)
        return path
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description="Unified NIH + TB + COVID fine-tune for XRAYNET+")
    p.add_argument("--nih-root", type=str, default=None)
    p.add_argument("--tb-root", type=str, default=None)
    p.add_argument("--covid-root", type=str, default=None)
    p.add_argument(
        "--pneumonia-mooney-root",
        type=str,
        default=None,
        help="Unzip root for paultimothymooney/chest-xray-pneumonia (train/test NORMAL vs PNEUMONIA)",
    )
    p.add_argument("--config", type=str, default=os.path.join(_ROOT, "config.yaml"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-backbone", type=int, default=0, metavar="N")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Default: models/saved/xraynet_unified_finetuned.pth",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument(
        "--nih-max-per-class",
        type=int,
        default=None,
        help="Cap NIH-only contribution per class before merge",
    )
    p.add_argument(
        "--cap-per-class",
        type=int,
        default=None,
        help="After merge, cap each class to this many images (balance)",
    )
    p.add_argument(
        "--nih-tb-proxy",
        choices=("auto", "none", "infiltration"),
        default="auto",
        help="auto: disable NIH TB proxy when --tb-root is set",
    )
    p.add_argument(
        "--nih-covid-proxy",
        choices=("auto", "none", "consolidation"),
        default="auto",
        help="auto: none; set consolidation to use NIH consolidation→COVID when no covid folder set",
    )
    p.add_argument("--dry-run", action="store_true", help="Print counts and exit")
    args = p.parse_args()

    cfg = _load_config(args.config)
    tcfg = cfg.get("training", {})
    dcfg = cfg.get("data", {})

    if args.nih_root:
        nih_root = _nih_root_if_ready(args.nih_root)
        if not nih_root:
            print(f"NIH data not found under {args.nih_root!r} (need Data_Entry_2017.csv).")
            sys.exit(1)
    else:
        nih_root = _nih_root_if_ready(dcfg.get("nih_chest_xray_root"))

    def _config_dir(p: str | None) -> str | None:
        if not p:
            return None
        p = os.path.abspath(p)
        if not os.path.isdir(p):
            return None
        # Ignore empty placeholder dirs (e.g. data/kaggle_tb_chest with no unzip yet)
        visible = [x for x in os.listdir(p) if not x.startswith(".")]
        return p if visible else None

    if args.tb_root:
        tb_root = os.path.abspath(args.tb_root)
        if not os.path.isdir(tb_root):
            print(f"TB root is not a directory: {tb_root!r}")
            sys.exit(1)
    else:
        tb_root = _config_dir(dcfg.get("kaggle_tb_chest_root"))

    if args.covid_root:
        covid_root = os.path.abspath(args.covid_root)
        if not os.path.isdir(covid_root):
            print(f"COVID root is not a directory: {covid_root!r}")
            sys.exit(1)
    else:
        covid_root = _config_dir(dcfg.get("kaggle_covid_chest_root"))

    if args.pneumonia_mooney_root:
        pneumonia_mooney_root = os.path.abspath(args.pneumonia_mooney_root)
        if not os.path.isdir(pneumonia_mooney_root):
            print(f"Pneumonia (Mooney) root is not a directory: {pneumonia_mooney_root!r}")
            sys.exit(1)
    else:
        pneumonia_mooney_root = _config_dir(dcfg.get("mooney_chest_xray_root"))

    if not nih_root and not tb_root and not covid_root and not pneumonia_mooney_root:
        print(
            "Set at least one of --nih-root, --tb-root, --covid-root, --pneumonia-mooney-root "
            "(or config.yaml data.*_root)."
        )
        print("See data/UNIFIED_FINETUNE_SETUP.txt")
        sys.exit(1)

    nih_tb = "auto" if args.nih_tb_proxy == "auto" else args.nih_tb_proxy  # type: ignore[assignment]
    nih_cov = "auto" if args.nih_covid_proxy == "auto" else args.nih_covid_proxy  # type: ignore[assignment]

    if args.dry_run:
        print("Dry run — scanning paths:")
        if nih_root and os.path.isdir(nih_root):
            print(f"  NIH: {nih_root}")
        elif nih_root:
            print(f"  NIH: MISSING {nih_root}")
        if tb_root:
            print_kaggle_scan_hint(tb_root, "TB")
        if covid_root:
            print_kaggle_scan_hint(covid_root, "COVID")
        if pneumonia_mooney_root:
            print_kaggle_scan_hint(pneumonia_mooney_root, "Mooney pneumonia")
        try:
            tr, va = build_unified_train_val(
                nih_root=nih_root,
                tb_root=tb_root,
                covid_root=covid_root,
                pneumonia_mooney_root=pneumonia_mooney_root,
                val_fraction=args.val_fraction,
                seed=args.seed,
                nih_max_per_class=args.nih_max_per_class,
                nih_tb_proxy=nih_tb,
                nih_covid_proxy=nih_cov,
                cap_per_class_after_merge=args.cap_per_class,
            )
            print_class_histogram(tr, "Train (dry)")
            print_class_histogram(va, "Val (dry)")
        except Exception as e:
            print(f"Build failed: {e}")
            sys.exit(1)
        print("OK")
        return

    epochs = args.epochs if args.epochs is not None else int(tcfg.get("epochs", 20))
    batch_size = args.batch_size if args.batch_size is not None else int(tcfg.get("batch_size", 24))
    lr = args.lr if args.lr is not None else float(tcfg.get("learning_rate", 2e-4))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(tcfg.get("weight_decay", 0.01))

    output_path = args.output or os.path.join(_ROOT, "models", "saved", "xraynet_unified_finetuned.pth")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    train_samples, val_samples = build_unified_train_val(
        nih_root=nih_root,
        tb_root=tb_root,
        covid_root=covid_root,
        pneumonia_mooney_root=pneumonia_mooney_root,
        val_fraction=args.val_fraction,
        seed=args.seed,
        nih_max_per_class=args.nih_max_per_class,
        nih_tb_proxy=nih_tb,
        nih_covid_proxy=nih_cov,
        cap_per_class_after_merge=args.cap_per_class,
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
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
        print(f"Loaded {args.resume}")

    if args.freeze_backbone > 0:
        set_backbone_requires_grad(model, False)
        for par in model.classifier.parameters():
            par.requires_grad = True

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
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.5, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - epoch + 1, 1)
            )
            print(f"Epoch {epoch}: backbone unfrozen, lr={lr * 0.5}")

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
            if i < len(tot) and tot[i] > 0:
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
                    "sources": {"nih": nih_root, "tb": tb_root, "covid": covid_root},
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
    print("Use in app: cp models/saved/xraynet_unified_finetuned.pth models/saved/xraynet_plus.pth")
    print("  or export XRAYNET_WEIGHTS=...")


if __name__ == "__main__":
    main()
