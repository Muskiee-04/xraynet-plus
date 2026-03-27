"""
Create `models/saved/xraynet_plus.pth`: ImageNet-pretrained EfficientNet-B0 + 4-class head.
Use your own fine-tuned weights for clinical deployment; this checkpoint makes the app runnable offline.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch

from src.models.cxr_classifier import CLASS_NAMES, build_efficientnet_cxr


def main():
    out_dir = os.path.join(_ROOT, "models", "saved")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "xraynet_plus.pth")

    model = build_efficientnet_cxr(num_classes=len(CLASS_NAMES), pretrained_backbone=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
        },
        path,
    )
    print(f"Saved demo checkpoint to {path}")


if __name__ == "__main__":
    main()
