from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.explainability.gradcam import GradCAMPlusPlusTorch
from src.models.cxr_classifier import CLASS_NAMES, build_efficientnet_cxr
from src.utils.cxr_recommendations import get_recommendation_detail, get_recommendation_line
from src.utils.helpers import create_gradcam_visualization


class TorchCXRInference:
    """
    PyTorch EfficientNet-B0 4-class CXR classifier with Grad-CAM++ heatmaps.
    """

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        if weights_path is None and os.environ.get("XRAYNET_WEIGHTS"):
            weights_path = os.environ.get("XRAYNET_WEIGHTS")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = list(CLASS_NAMES)
        self.model = build_efficientnet_cxr(num_classes=len(self.class_names), pretrained_backbone=True)
        self.model.to(self.device)
        self.model.eval()

        self.weights_loaded_from: Optional[str] = None
        if weights_path:
            if not self._try_load_checkpoint(weights_path):
                raise RuntimeError(f"Could not load weights from {weights_path!r} (wrong architecture?).")
            self.weights_loaded_from = os.path.abspath(weights_path)
        else:
            for path in self._checkpoint_candidates():
                if self._try_load_checkpoint(path):
                    self.weights_loaded_from = os.path.abspath(path)
                    break
        self._target_layer = self.model.features[-1]
        self._cam: Optional[GradCAMPlusPlusTorch] = None

    @staticmethod
    def _repo_root() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    @classmethod
    def _checkpoint_candidates(cls) -> list[str]:
        root = cls._repo_root()
        return [
            os.path.join(root, "models", "saved", "xraynet_unified_finetuned.pth"),
            os.path.join(root, "models", "saved", "xraynet_plus.pth"),
            os.path.join(root, "models", "saved", "xraynet_nih_finetuned.pth"),
            os.path.join(root, "models", "saved", "xraynet_plus_merged.pth"),
            os.path.join(root, "models", "saved", "xraynet_finetuned_best.pth"),
            os.path.join(root, "models", "saved", "best_model.pth"),
        ]

    def _try_load_checkpoint(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False
        try:
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(path, map_location=self.device)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state, strict=True)
            return True
        except (RuntimeError, OSError, KeyError):
            return False

    def _ensure_cam(self) -> GradCAMPlusPlusTorch:
        if self._cam is None:
            self._cam = GradCAMPlusPlusTorch(self.model, self._target_layer)
        return self._cam

    def close_cam(self):
        if self._cam is not None:
            self._cam.remove()
            self._cam = None

    @torch.inference_mode()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float32)
        return self.model(x)

    def get_detailed_prediction(self, image_tensor: torch.Tensor, original_rgb: np.ndarray) -> dict:
        """
        image_tensor: (1,3,H,W) float32 CPU or CUDA
        original_rgb: HxWx3 uint8 for heatmap sizing
        """
        self.model.train(False)
        self.model.zero_grad(set_to_none=True)
        x = image_tensor.to(self.device, dtype=torch.float32).clone().detach().requires_grad_(True)

        cam_h = self._ensure_cam()
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        class_idx = int(idx.item())
        class_name = self.class_names[class_idx]
        confidence = float(conf.item())

        prob_map = {self.class_names[i]: float(probs[i].item()) for i in range(len(self.class_names))}

        logits[0, class_idx].backward(retain_graph=False)
        cam = cam_h.compute_heatmap()
        heatmap_rgb = create_gradcam_visualization(cam, original_rgb)

        rec = get_recommendation_detail(class_name)
        clinical = list(rec["clinical_steps"])
        prevention = list(rec["prevention"])

        return {
            "class_name": class_name,
            "class_index": class_idx,
            "confidence": confidence,
            "probabilities": prob_map,
            "recommendation": get_recommendation_line(class_name),
            "clinical_steps": clinical,
            "prevention": prevention,
            "description": f"Model assigns highest probability to {class_name}.",
            "heatmap_rgb": heatmap_rgb,
        }
