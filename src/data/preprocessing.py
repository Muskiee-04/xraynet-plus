from __future__ import annotations

import cv2
import numpy as np
import torch


class CXRPreprocessor:
    """Chest X-ray preprocessing: optional CLAHE, resize, ImageNet normalization."""

    def __init__(self, target_size: int = 224, use_clahe: bool = True):
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _to_rgb_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image.astype(np.uint8)
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0].astype(np.uint8)
        elif image.ndim == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return rgb

    def preprocess_for_inference(
        self, image: np.ndarray
    ) -> tuple[torch.Tensor | None, np.ndarray]:
        """
        Returns:
            tensor (1, 3, H, W) float32 on CPU, and original RGB uint8 for display/Grad-CAM.
        """
        try:
            rgb = self._to_rgb_uint8(image)
            original_rgb = rgb.copy()
            resized = cv2.resize(rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
            x = resized.astype(np.float32) / 255.0
            x = (x - self.mean) / self.std
            x = np.transpose(x, (2, 0, 1))
            tensor = torch.from_numpy(x).unsqueeze(0)
            return tensor, original_rgb
        except Exception:
            return None, np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
