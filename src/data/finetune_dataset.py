"""
Build labeled file lists for 4-class XRAYNET+ fine-tuning from directory trees.

Expected layout (recommended)::

    DATA_ROOT/
      train/
        Tuberculosis/*.png
        Pneumonia/*.jpg
        ...
      val/          # optional; if omitted, a stratified split is made from train/
        ...

Folder names are matched case-insensitively to canonical classes or common aliases
(tb, normal, covid19, etc.).
"""
from __future__ import annotations

import os
import random
from typing import Sequence

import torch
from torch.utils.data import Dataset

from src.models.cxr_classifier import CLASS_NAMES

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# (canonical_index, acceptable folder name tokens after normalization)
_FOLDER_ALIASES: list[tuple[int, tuple[str, ...]]] = [
    (0, ("tuberculosis", "tb")),
    (1, ("pneumonia",)),
    (2, ("covid19", "covid_19", "covid")),
    (
        3,
        ("nofindings", "no_findings", "normal", "healthy", "negative", "clear"),
    ),
]


def _normalize_folder_name(name: str) -> str:
    s = name.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def folder_to_class_index(folder_name: str) -> int | None:
    """Map a subdirectory name to 0..3, or None if unknown."""
    key = _normalize_folder_name(folder_name)
    # Exact match to canonical folder-style names
    for i, c in enumerate(CLASS_NAMES):
        if key == _normalize_folder_name(c):
            return i
    for idx, aliases in _FOLDER_ALIASES:
        if key in aliases:
            return idx
    return None


def collect_samples(split_dir: str) -> list[tuple[str, int]]:
    """All (path, label) under split_dir/<class_folder>/."""
    out: list[tuple[str, int]] = []
    if not os.path.isdir(split_dir):
        return out
    for sub in sorted(os.listdir(split_dir)):
        sub_path = os.path.join(split_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        idx = folder_to_class_index(sub)
        if idx is None:
            raise ValueError(
                f"Unknown class folder '{sub}' in {split_dir}. "
                f"Use one of: {list(CLASS_NAMES)} or aliases (tb, normal, covid19, …)."
            )
        for root, _, files in os.walk(sub_path):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in _IMAGE_EXT:
                    continue
                out.append((os.path.join(root, fn), idx))
    return out


def stratified_split(
    samples: list[tuple[str, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if not samples:
        return [], []
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {}
    for path, y in samples:
        by_label.setdefault(y, []).append(path)
    train: list[tuple[str, int]] = []
    val: list[tuple[str, int]] = []
    for y, paths in by_label.items():
        paths = paths.copy()
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_ratio))) if len(paths) > 1 else 0
        if len(paths) == 1:
            n_val = 0
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]
        if not train_paths and val_paths:
            # keep at least one sample in train
            train_paths = [val_paths.pop()]
        for p in train_paths:
            train.append((p, y))
        for p in val_paths:
            val.append((p, y))
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_train_val_lists(
    data_root: str,
    val_split: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """
    If data_root/val exists, use train/ and val/.
    Else use all of train/ and stratified split into train/val.
    """
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    if os.path.isdir(val_root):
        train_samples = collect_samples(train_root)
        val_samples = collect_samples(val_root)
        return train_samples, val_samples
    train_all = collect_samples(train_root)
    if not train_all:
        raise FileNotFoundError(
            f"No images found under {train_root}. "
            "Create class subfolders (e.g. Tuberculosis/, Pneumonia/, …)."
        )
    train_samples, val_samples = stratified_split(train_all, val_ratio=val_split, seed=seed)
    if not val_samples and len(train_all) > 1:
        rng = random.Random(seed)
        pool = train_all.copy()
        rng.shuffle(pool)
        n_val = max(1, int(round(len(pool) * val_split)))
        n_val = min(n_val, len(pool) - 1)
        val_samples = pool[:n_val]
        train_samples = pool[n_val:]
    return train_samples, val_samples


def class_weights_from_samples(samples: Sequence[tuple[str, int]], num_classes: int) -> list[float]:
    counts = [0] * num_classes
    for _, y in samples:
        counts[y] += 1
    total = sum(counts)
    if total == 0:
        return [1.0] * num_classes
    # inverse frequency, normalized
    w = [total / (c * num_classes) if c > 0 else 0.0 for c in counts]
    m = max(w) or 1.0
    return [x / m if x > 0 else 1.0 for x in w]


class FinetunePathDataset(Dataset):
    """Loads (path, label) with torchvision transforms."""

    def __init__(self, samples: list[tuple[str, int]], transform):
        from torchvision.datasets.folder import default_loader

        self.samples = samples
        self.transform = transform
        self.loader = default_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, y
