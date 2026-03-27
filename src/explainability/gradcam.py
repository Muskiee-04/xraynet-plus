"""
Class-discriminative localization: Grad-CAM (Selvaraju et al.) and Grad-CAM++
(Chattopadhyay et al.) — default path is Grad-CAM++ channel weighting without extra deps.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAMPlusPlusTorch:
    """Grad-CAM++ weights (see Chattopadhyay et al., WACV 2018)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _m, _inp, out):
        self.activations = out.detach()

    def _backward_hook(self, _m, _gi, go):
        self.gradients = go[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def compute_heatmap(self) -> np.ndarray:
        """
        After one forward + backward on the hooked model, build a 2D map in [0, 1].
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Run model forward and target-class backward before compute_heatmap().")

        acts = self.activations[0]
        grads = self.gradients[0]
        eps = 1e-8

        grad_2 = grads * grads
        grad_3 = grad_2 * grads
        denom = 2.0 * grad_2 + (acts * grad_3).sum(dim=(1, 2), keepdim=True) + eps
        alphas = grad_2 / denom
        weights = (alphas * F.relu(grads)).sum(dim=(1, 2), keepdim=True)

        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)
        return cam.detach().cpu().numpy().astype(np.float32)
