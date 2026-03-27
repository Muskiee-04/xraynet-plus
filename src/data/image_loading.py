"""
Load chest imaging from raw bytes: DICOM (pydicom) or common raster formats (Pillow).
Returns HxWx3 uint8 RGB for CXRPreprocessor.
"""
from __future__ import annotations

import io
from typing import BinaryIO

import numpy as np
from PIL import Image


def _is_dicom_magic(data: bytes) -> bool:
    return len(data) >= 132 and data[128:132] == b"DICM"


def _filename_suggests_dicom(name: str) -> bool:
    n = name.lower()
    return n.endswith(".dcm") or n.endswith(".dicom")


def dicom_bytes_to_rgb_uint8(data: bytes) -> np.ndarray:
    """Decode DICOM pixel data to uint8 RGB (3-channel) for CNN preprocessing."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float64, copy=False)
        lo, hi = np.percentile(rgb, (0.5, 99.5))
        if hi <= lo:
            lo, hi = float(rgb.min()), float(rgb.max())
            if hi <= lo:
                hi = lo + 1.0
        rgb = np.clip((rgb - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return rgb
    if arr.ndim > 2:
        arr = arr[0]  # multi-frame grayscale: first frame
    arr = arr.astype(np.float64, copy=False)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    try:
        arr = apply_voi_lut(arr, ds)
        arr = np.asarray(arr, dtype=np.float64)
    except Exception:
        pass

    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2") or "MONOCHROME2")
    if photometric.upper() == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = np.percentile(arr, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    u8 = np.clip(arr, 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def raster_bytes_to_rgb_uint8(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_image_bytes_to_rgb(data: bytes, filename: str = "") -> np.ndarray:
    """
    Auto-detect DICOM vs raster. `filename` hints .dcm extension when magic is missing.
    """
    if not data:
        raise ValueError("Empty file.")
    if _is_dicom_magic(data) or _filename_suggests_dicom(filename):
        try:
            return dicom_bytes_to_rgb_uint8(data)
        except Exception as e:
            if _filename_suggests_dicom(filename):
                raise ValueError(f"DICOM decode failed: {e}") from e
    return raster_bytes_to_rgb_uint8(data)


def load_image_from_stream(fp: BinaryIO, filename: str = "") -> np.ndarray:
    data = fp.read()
    return load_image_bytes_to_rgb(data, filename)
