import cv2
import numpy as np


def create_gradcam_visualization(cam2d: np.ndarray, original_rgb: np.ndarray) -> np.ndarray:
    """
    Resize cam to original image size, apply colormap, return uint8 RGB heatmap image.
    """
    h, w = original_rgb.shape[:2]
    cam = cv2.resize(cam2d, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_u8 = np.uint8(255 * np.clip(cam, 0.0, 1.0))
    heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return heat_rgb


def overlay_heatmap_on_image(original_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if original_rgb.shape[:2] != heatmap_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (original_rgb.shape[1], original_rgb.shape[0]))
    out = cv2.addWeighted(original_rgb.astype(np.float32), 1.0 - alpha, heatmap_rgb.astype(np.float32), alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)
