"""
Map NIH ChestX-ray14 CSV rows to XRAYNET+ single-label indices.

Class indices match `CLASS_NAMES` in `src.models.cxr_classifier`:
  0 Tuberculosis   — NOT in NIH labels; optional *weak* proxy only (see below).
  1 Pneumonia      — rows where "Pneumonia" appears in Finding Labels.
  2 COVID-19       — NOT in NIH (dataset predates COVID); optional *weak* proxy only.
  3 No Findings    — Finding Labels is exactly "No Finding".

Multi-label priority (first match wins):
  Pneumonia > optional COVID proxy > optional TB proxy > pure No Finding.

Rows that do not match any rule are skipped (keeps labels cleaner).

**Clinical caveat:** Infiltration ≠ TB; Consolidation ≠ COVID. Proxies are for coursework /
transfer-learning bootstrap only; add real TB/COVID sets for clinical use.
"""
from __future__ import annotations

import os
import random
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.cxr_classifier import CLASS_NAMES

TBProxy = Literal["none", "infiltration"]
CovidProxy = Literal["none", "consolidation"]


def map_finding_to_class(
    finding_labels: str,
    *,
    tb_proxy: TBProxy = "infiltration",
    covid_proxy: CovidProxy = "none",
) -> int | None:
    parts = {p.strip() for p in str(finding_labels).split("|") if p.strip()}

    if "Pneumonia" in parts:
        return 1

    if covid_proxy == "consolidation" and "Consolidation" in parts and "Pneumonia" not in parts:
        return 2

    if tb_proxy == "infiltration" and "Infiltration" in parts and "Pneumonia" not in parts:
        return 0

    if parts == {"No Finding"}:
        return 3

    return None


def _discover_nih_paths(nih_root: str) -> tuple[str, str, str | None, str | None]:
    """Return (csv_path, image_dir, train_list_path, test_list_path)."""
    nih_root = os.path.abspath(nih_root)
    candidates = [
        nih_root,
        os.path.join(nih_root, "archive"),
        os.path.join(nih_root, "data"),
        os.path.join(nih_root, "nih-chest-xrays"),
    ]
    csv_path = None
    base = None
    for c in candidates:
        p = os.path.join(c, "Data_Entry_2017.csv")
        if os.path.isfile(p):
            csv_path = p
            base = c
            break
    if not csv_path:
        raise FileNotFoundError(
            f"Data_Entry_2017.csv not found under {nih_root}. "
            "Download from Kaggle: nih-chest-xrays/data and unzip."
        )

    image_dir = None
    for sub in ("images", "images_224", "resized", "png"):
        d = os.path.join(base, sub)
        if os.path.isdir(d):
            image_dir = d
            break
    if image_dir is None:
        raise FileNotFoundError(f"No images/ folder next to CSV under {base}")

    import glob

    train_list = None
    test_list = None
    for p in sorted(glob.glob(os.path.join(base, "train_val_list*.txt"))):
        train_list = p
        break
    if not train_list:
        for p in sorted(glob.glob(os.path.join(base, "train_list*.txt"))):
            train_list = p
            break
    for p in sorted(glob.glob(os.path.join(base, "test_list*.txt"))):
        test_list = p
        break
    return csv_path, image_dir, train_list, test_list


def load_image_list(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return {line.strip() for line in f if line.strip()}


def _collect_for_allowed(
    df: pd.DataFrame,
    image_dir: str,
    allowed: set[str],
    tb_proxy: TBProxy,
    covid_proxy: CovidProxy,
) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for _, row in df.iterrows():
        name = row["Image Index"]
        if name not in allowed:
            continue
        path = os.path.join(image_dir, name)
        if not os.path.isfile(path):
            continue
        y = map_finding_to_class(
            row["Finding Labels"], tb_proxy=tb_proxy, covid_proxy=covid_proxy
        )
        if y is None:
            continue
        out.append((os.path.abspath(path), y))
    return out


def _cap_per_class(samples: list[tuple[str, int]], max_per_class: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    by_c: dict[int, list[tuple[str, int]]] = {}
    for p, y in samples:
        by_c.setdefault(y, []).append((p, y))
    capped = []
    for lst in by_c.values():
        rng.shuffle(lst)
        capped.extend(lst[:max_per_class])
    rng.shuffle(capped)
    return capped


def build_samples_from_nih(
    nih_root: str,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
    tb_proxy: TBProxy = "infiltration",
    covid_proxy: CovidProxy = "none",
    max_per_class: int | None = None,
    official_test_as_val: bool = False,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Build (train_samples, val_samples) as lists of (absolute_image_path, class_index).

    - Default: NIH ``train_val_list.txt`` → stratified random train/val split.
    - ``official_test_as_val=True``: train on train_val list, validate on ``test_list.txt``
      (matches common ChestX-ray14 benchmarking; val distribution may shift).
    """
    csv_path, image_dir, train_list_p, test_list_p = _discover_nih_paths(nih_root)
    df = pd.read_csv(csv_path)

    if not train_list_p:
        raise FileNotFoundError(
            "train_val_list.txt (or train_list.txt) not found next to CSV. "
            "Use the official NIH file lists from the dataset."
        )
    train_allowed = load_image_list(train_list_p)

    if official_test_as_val:
        if not test_list_p:
            raise FileNotFoundError("test_list.txt not found; cannot use official test as val.")
        test_allowed = load_image_list(test_list_p)
        train_samples = _collect_for_allowed(df, image_dir, train_allowed, tb_proxy, covid_proxy)
        val_samples = _collect_for_allowed(df, image_dir, test_allowed, tb_proxy, covid_proxy)
    else:
        samples = _collect_for_allowed(df, image_dir, train_allowed, tb_proxy, covid_proxy)
        labels = [y for _, y in samples]
        paths = [p for p, _ in samples]
        try:
            pt, pv, yt, yv = train_test_split(
                paths,
                labels,
                test_size=val_fraction,
                random_state=seed,
                stratify=labels,
            )
        except ValueError:
            pt, pv, yt, yv = train_test_split(
                paths, labels, test_size=val_fraction, random_state=seed
            )
        train_samples = list(zip(pt, yt))
        val_samples = list(zip(pv, yv))

    if not train_samples:
        raise RuntimeError(
            "No training samples. Check paths, CSV, and that images exist (e.g. images/*.png)."
        )
    if not val_samples:
        raise RuntimeError("No validation samples.")

    if max_per_class is not None:
        train_samples = _cap_per_class(train_samples, max_per_class, seed)
        val_samples = _cap_per_class(val_samples, max_per_class, seed + 1)

    return train_samples, val_samples


def print_class_histogram(samples: list[tuple[str, int]], title: str) -> None:
    from collections import Counter

    c = Counter(y for _, y in samples)
    print(f"{title} (n={len(samples)}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {c.get(i, 0)}")
