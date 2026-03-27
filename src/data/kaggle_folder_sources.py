"""
Scan Kaggle-style folder datasets (class name = subdirectory) for XRAYNET+ labels.

Datasets:
  - Tuberculosis (TB) Chest X-ray: tawsifurrahman/tuberculosis-tb-chest-xray-dataset
  - COVID-19 Chest X-ray: alifrahman/covid19-chest-xray-image-dataset
  - Chest X-ray Pneumonia (Mooney): paultimothymooney/chest-xray-pneumonia
    (``chest_xray/train/NORMAL``, ``.../PNEUMONIA``, etc.)

Unzip layouts vary; we try direct class folders under ``root`` or one extra nesting level,
and a deeper walk for Mooney-style trees.
"""
from __future__ import annotations

import os

from src.models.cxr_classifier import CLASS_NAMES

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _normalize(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def _class_from_folder_name(name: str, class_to_aliases: dict[int, frozenset[str]]) -> int | None:
    key = _normalize(name)
    if not key:
        return None
    for cls_idx, aliases in class_to_aliases.items():
        if key in aliases:
            return cls_idx
    for cls_idx, aliases in class_to_aliases.items():
        for a in aliases:
            if len(a) >= 4 and (key == a or key.startswith(a + "_") or key.endswith("_" + a)):
                return cls_idx
    return None


def _walk_images(root_dir: str, label: int, out: list[tuple[str, int]]) -> None:
    for r, _, files in os.walk(root_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _IMAGE_EXT:
                continue
            p = os.path.abspath(os.path.join(r, fn))
            if os.path.isfile(p):
                out.append((p, label))


def _recursive_collect_mapped(
    root: str,
    mapping: dict[int, frozenset[str]],
    *,
    max_depth: int = 14,
) -> list[tuple[str, int]]:
    """
    Walk ``root`` until a directory is found whose immediate subdirs are class folders
    (e.g. ``train/NORMAL``, ``train/PNEUMONIA``). Skips ``__MACOSX`` and dot dirs.
    """
    root = os.path.abspath(root)
    out: list[tuple[str, int]] = []

    def visit(d: str, depth: int) -> None:
        if depth > max_depth:
            return
        direct = _try_collect_direct_class_folders(d, mapping)
        if direct:
            out.extend(direct)
            return
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return
        for name in names:
            if name.startswith(".") or name == "__MACOSX":
                continue
            sub = os.path.join(d, name)
            if os.path.isdir(sub):
                visit(sub, depth + 1)

    visit(root, 0)
    return out


def _try_collect_direct_class_folders(root: str, mapping: dict[int, frozenset[str]]) -> list[tuple[str, int]]:
    root = os.path.abspath(root)
    out: list[tuple[str, int]] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if not os.path.isdir(sub):
            continue
        cls = _class_from_folder_name(name, mapping)
        if cls is not None:
            _walk_images(sub, cls, out)
    return out


def collect_tb_chest_kaggle(root: str) -> list[tuple[str, int]]:
    """
    TB vs normal style folders → class 0 (Tuberculosis) and 3 (No Findings).

    Expected folder names (any similar): Tuberculosis, TB, Normal, Healthy, Negative.
    """
    mapping: dict[int, frozenset[str]] = {
        0: frozenset(
            {
                "tuberculosis",
                "tb",
                "tuberculoses",
                "tuberculosiss",
                "positive",
                "ptb",
                "tuberculosis_image",
                "tb_positive",
            }
        ),
        3: frozenset(
            {
                "normal",
                "healthy",
                "negative",
                "no_finding",
                "nofinding",
                "no_findings",
                "control",
                "non_tb",
                "not_tb",
            }
        ),
    }
    samples = _try_collect_direct_class_folders(root, mapping)
    if samples:
        return samples
    for name in os.listdir(os.path.abspath(root)):
        sub = os.path.join(os.path.abspath(root), name)
        if not os.path.isdir(sub):
            continue
        inner = _try_collect_direct_class_folders(sub, mapping)
        if inner:
            return inner
    return []


def collect_pneumonia_mooney_chest_xray(root: str) -> list[tuple[str, int]]:
    """
    Paul Mooney chest X-ray pneumonia dataset → Pneumonia (1) and No Findings (3).

    Typical layout: ``chest_xray/train/{NORMAL,PNEUMONIA}``, ``chest_xray/test/...``.
    """
    mapping: dict[int, frozenset[str]] = {
        1: frozenset(
            {
                "pneumonia",
                "bacterial_pneumonia",
                "viral_pneumonia",
                "virus",
                "bacteria",
            }
        ),
        3: frozenset(
            {
                "normal",
                "healthy",
                "negative",
                "no_finding",
                "nofinding",
                "no_findings",
            }
        ),
    }
    samples = _try_collect_direct_class_folders(root, mapping)
    if samples:
        return samples
    for name in os.listdir(os.path.abspath(root)):
        sub = os.path.join(os.path.abspath(root), name)
        if not os.path.isdir(sub) or name == "__MACOSX":
            continue
        inner = _try_collect_direct_class_folders(sub, mapping)
        if inner:
            return inner
    return _recursive_collect_mapped(root, mapping)


def collect_covid19_chest_kaggle(root: str) -> list[tuple[str, int]]:
    """
    COVID / pneumonia / normal style folders → classes 2, 1, 3.

    Typical: COVID-19, COVID, Normal, Viral Pneumonia, Lung Opacity, etc.
    """
    mapping: dict[int, frozenset[str]] = {
        2: frozenset(
            {
                "covid",
                "covid19",
                "covid_19",
                "covid-19",
                "sars_cov_2",
                "sarscov2",
                "positive_covid",
                "corona",
                "corona_positive",
                "positive_covid19",
            }
        ),
        1: frozenset(
            {
                "pneumonia",
                "viral_pneumonia",
                "bacterial_pneumonia",
                "viral",
                "bacterial",
                "lung_opacity",
                "lungopacity",
                "opacity",
            }
        ),
        3: frozenset(
            {
                "normal",
                "healthy",
                "negative",
                "non_covid",
                "regular",
                "no_finding",
                "nofinding",
                "no_findings",
                "corona_negative",
            }
        ),
    }
    samples = _try_collect_direct_class_folders(root, mapping)
    if samples:
        return samples
    for name in os.listdir(os.path.abspath(root)):
        sub = os.path.join(os.path.abspath(root), name)
        if not os.path.isdir(sub) or name == "__MACOSX":
            continue
        inner = _try_collect_direct_class_folders(sub, mapping)
        if inner:
            return inner
    deep = _recursive_collect_mapped(root, mapping)
    return deep


def dedupe_samples(samples: list[tuple[str, int]]) -> list[tuple[str, int]]:
    seen: set[str] = set()
    out: list[tuple[str, int]] = []
    for p, y in samples:
        if p in seen:
            continue
        seen.add(p)
        out.append((p, y))
    return out


def print_kaggle_scan_hint(root: str, kind: str) -> None:
    if not os.path.isdir(root):
        print(f"  [{kind}] Missing directory: {root}")
        return
    subs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    print(f"  [{kind}] {root} — subfolders: {subs[:20]}{'...' if len(subs) > 20 else ''}")
