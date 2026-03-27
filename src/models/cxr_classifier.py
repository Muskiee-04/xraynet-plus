import warnings

import torch.nn as nn

from src.utils.ssl_setup import apply_ssl_compatibility

apply_ssl_compatibility()

from torchvision import models

CLASS_NAMES = [
    "Tuberculosis",
    "Pneumonia",
    "COVID-19",
    "No Findings",
]


def _efficientnet_b0_no_weights() -> nn.Module:
    try:
        return models.efficientnet_b0(weights=None)
    except Exception:
        return models.efficientnet_b0(pretrained=False)


def build_efficientnet_cxr(num_classes: int = 4, pretrained_backbone: bool = True) -> nn.Module:
    """
    If ImageNet weights cannot be downloaded (e.g. SSL CERTIFICATE_VERIFY_FAILED on macOS),
    falls back to a randomly initialized backbone — no network required.
    """
    m: nn.Module
    if pretrained_backbone:
        try:
            w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            m = models.efficientnet_b0(weights=w)
        except Exception as e:
            warnings.warn(
                f"ImageNet weights unavailable ({type(e).__name__}: {e}). "
                "Using randomly initialized backbone (run on a machine with working SSL to use pretrained weights, "
                "or place models/saved/xraynet_plus.pth from init_demo_model.py).",
                UserWarning,
                stacklevel=2,
            )
            m = _efficientnet_b0_no_weights()
    else:
        m = _efficientnet_b0_no_weights()
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m
