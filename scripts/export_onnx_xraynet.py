"""Export `models/saved/xraynet_plus.pth` (4-class XRAYNET+ head) to ONNX for edge deployment."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
from torchvision import models

from src.models.cxr_classifier import CLASS_NAMES


def main():
    ckpt_path = os.path.join(_ROOT, "models", "saved", "xraynet_plus.pth")
    if not os.path.isfile(ckpt_path):
        print("Missing checkpoint. Run: python scripts/init_demo_model.py")
        return

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASS_NAMES))
    m.load_state_dict(state, strict=True)
    m.eval()

    out = os.path.join(_ROOT, "models", "saved", "xraynet_plus.onnx")
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        m,
        dummy,
        out,
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Wrote {out} ({len(CLASS_NAMES)} classes: {CLASS_NAMES})")


if __name__ == "__main__":
    main()
