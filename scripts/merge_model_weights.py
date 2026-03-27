#!/usr/bin/env python3
"""
Toy federated-style merge: element-wise mean of two compatible checkpoints.

  python scripts/merge_model_weights.py models/saved/site_a.pth models/saved/site_b.pth \\
      --output models/saved/xraynet_plus_merged.pth

Use only for same-architecture models; this is not secure aggregation or DP.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from src.models.cxr_classifier import CLASS_NAMES  # noqa: E402


def load_state(path: str) -> dict:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("model_state_dict", ckpt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_a", type=str)
    p.add_argument("checkpoint_b", type=str)
    p.add_argument("--output", type=str, default=os.path.join(_ROOT, "models", "saved", "xraynet_plus_merged.pth"))
    args = p.parse_args()

    sa = load_state(args.checkpoint_a)
    sb = load_state(args.checkpoint_b)
    if set(sa.keys()) != set(sb.keys()):
        raise SystemExit("State dict keys differ; cannot merge.")
    merged = {}
    for k in sa:
        a, b = sa[k], sb[k]
        if not torch.is_floating_point(a):
            merged[k] = a.clone()
        else:
            merged[k] = (0.5 * (a.float() + b.float())).to(dtype=a.dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": merged,
            "class_names": list(CLASS_NAMES),
            "merged_from": [os.path.abspath(args.checkpoint_a), os.path.abspath(args.checkpoint_b)],
        },
        args.output,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
