"""
ONNX Runtime inference for 4-class XRAYNET+ (exported via scripts/export_onnx_xraynet.py).
Faster CPU path for edge deployment; no Grad-CAM (use PyTorch `/predict` for explainability).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

from src.models.cxr_classifier import CLASS_NAMES
from src.utils.cxr_recommendations import get_recommendation_detail, get_recommendation_line


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


class OnnxCXRInference:
    def __init__(self, onnx_path: Optional[str] = None, providers: Optional[list[str]] = None):
        import onnxruntime as ort

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = (
            onnx_path
            or os.environ.get("XRAYNET_ONNX_PATH")
            or os.path.join(root, "models", "saved", "xraynet_plus.onnx")
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ONNX model not found at {path}. Run: python scripts/export_onnx_xraynet.py"
            )
        if providers is not None:
            prov = providers
        else:
            mode = (os.environ.get("XRAYNET_ORT_PROVIDERS") or "cpu").strip().lower()
            if mode == "cuda" and torch.cuda.is_available():
                prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                prov = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=prov)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = list(CLASS_NAMES)

    def predict_logits(self, x_nchw: np.ndarray) -> np.ndarray:
        """x_nchw: float32 (1,3,H,W) normalized like training."""
        out = self.session.run([self.output_name], {self.input_name: x_nchw})[0]
        return out[0]

    def get_prediction_dict(self, image_tensor_torch: torch.Tensor) -> dict[str, Any]:
        """Match keys used by PDF/reporting (no heatmap)."""
        x = image_tensor_torch.detach().cpu().numpy().astype(np.float32)
        logits = self.predict_logits(x)
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        name = self.class_names[idx]
        rec = get_recommendation_detail(name)
        return {
            "class_name": name,
            "class_index": idx,
            "confidence": float(probs[idx]),
            "probabilities": {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))},
            "recommendation": get_recommendation_line(name),
            "clinical_steps": list(rec["clinical_steps"]),
            "prevention": list(rec["prevention"]),
            "description": f"Model assigns highest probability to {name} (ONNX backend).",
        }
