"""
Merge NIH ChestX-ray14 with optional Kaggle TB and COVID folder datasets,
then produce a single stratified train/val split for XRAYNET+ (4 classes).

When ``tb_root`` is set, NIH ``tb_proxy`` defaults to ``none`` (real TB images).
When ``covid_root`` is set, NIH ``covid_proxy`` defaults to ``none`` (real COVID images).
"""
from __future__ import annotations

import os
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.kaggle_folder_sources import (
    collect_covid19_chest_kaggle,
    collect_pneumonia_mooney_chest_xray,
    collect_tb_chest_kaggle,
    dedupe_samples,
)
from src.data.nih_xraynet_dataset import (
    CovidProxy,
    TBProxy,
    _cap_per_class,
    _collect_for_allowed,
    _discover_nih_paths,
    load_image_list,
)
from src.models.cxr_classifier import CLASS_NAMES


def collect_nih_trainval_flat(
    nih_root: str,
    *,
    tb_proxy: TBProxy = "infiltration",
    covid_proxy: CovidProxy = "none",
    max_per_class: int | None = None,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """All mapped samples from NIH train_val list (no train/val split)."""
    csv_path, image_dir, train_list_p, _test = _discover_nih_paths(nih_root)
    if not train_list_p:
        raise FileNotFoundError("NIH train_val_list*.txt not found.")
    df = pd.read_csv(csv_path)
    allowed = load_image_list(train_list_p)
    out = _collect_for_allowed(df, image_dir, allowed, tb_proxy, covid_proxy)
    if max_per_class is not None:
        out = _cap_per_class(out, max_per_class, seed)
    return out


def build_unified_train_val(
    *,
    nih_root: str | None = None,
    tb_root: str | None = None,
    covid_root: str | None = None,
    pneumonia_mooney_root: str | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    nih_max_per_class: int | None = None,
    nih_tb_proxy: TBProxy | Literal["auto"] = "auto",
    nih_covid_proxy: CovidProxy | Literal["auto"] = "auto",
    cap_per_class_after_merge: int | None = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Combine data sources; single global stratified split into train/val.

    At least one of nih_root, tb_root, covid_root, pneumonia_mooney_root must be provided.
    """
    if not nih_root and not tb_root and not covid_root and not pneumonia_mooney_root:
        raise ValueError(
            "Provide at least one of: nih_root, tb_root, covid_root, pneumonia_mooney_root"
        )

    if nih_tb_proxy == "auto":
        tb_p: TBProxy = "none" if tb_root else "infiltration"
    else:
        tb_p = nih_tb_proxy  # type: ignore[assignment]

    if nih_covid_proxy == "auto":
        cov_p: CovidProxy = "none"
    else:
        cov_p = nih_covid_proxy  # type: ignore[assignment]

    all_samples: list[tuple[str, int]] = []

    if nih_root:
        nih_root = os.path.abspath(nih_root)
        all_samples.extend(
            collect_nih_trainval_flat(
                nih_root,
                tb_proxy=tb_p,
                covid_proxy=cov_p,
                max_per_class=nih_max_per_class,
                seed=seed,
            )
        )

    if tb_root:
        tb_root = os.path.abspath(tb_root)
        tbs = collect_tb_chest_kaggle(tb_root)
        if not tbs:
            raise RuntimeError(
                f"No TB dataset images found under {tb_root}. "
                "Check unzip layout; subfolders should be named like Tuberculosis/ and Normal/."
            )
        all_samples.extend(tbs)

    if covid_root:
        covid_root = os.path.abspath(covid_root)
        cvs = collect_covid19_chest_kaggle(covid_root)
        if not cvs:
            raise RuntimeError(
                f"No COVID dataset images found under {covid_root}. "
                "Check unzip; subfolders like COVID/, Normal/, Pneumonia/ expected."
            )
        all_samples.extend(cvs)

    if pneumonia_mooney_root:
        pneumonia_mooney_root = os.path.abspath(pneumonia_mooney_root)
        pms = collect_pneumonia_mooney_chest_xray(pneumonia_mooney_root)
        if not pms:
            raise RuntimeError(
                f"No Mooney chest X-ray pneumonia images found under {pneumonia_mooney_root}. "
                "Expected nested folders like chest_xray/train/NORMAL and .../PNEUMONIA."
            )
        all_samples.extend(pms)

    all_samples = dedupe_samples(all_samples)
    if not all_samples:
        raise RuntimeError("No samples after merge.")

    if cap_per_class_after_merge is not None:
        all_samples = _cap_per_class(all_samples, cap_per_class_after_merge, seed + 7)

    labels = [y for _, y in all_samples]
    paths = [p for p, _ in all_samples]
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

    return list(zip(pt, yt)), list(zip(pv, yv))


def print_class_histogram(samples: list[tuple[str, int]], title: str) -> None:
    from collections import Counter

    c = Counter(y for _, y in samples)
    print(f"{title} (n={len(samples)}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {c.get(i, 0)}")
