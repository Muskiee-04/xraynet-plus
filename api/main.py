"""
XRAYNET+ REST API
-----------------
- POST /predict — PyTorch + Grad-CAM++ heatmap (PNG base64). Accepts PNG/JPEG/WebP and DICOM (.dcm).
- POST /predict/onnx — ONNX Runtime, class probabilities only (edge / no Grad-CAM).

Run from repo root:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Env:
  XRAYNET_WEIGHTS      — optional explicit .pth for PyTorch
  XRAYNET_ONNX_PATH    — optional path to .onnx (default models/saved/xraynet_plus.onnx)
  XRAYNET_ORT_PROVIDERS — cpu | cuda
"""
from __future__ import annotations

import base64
import io
import os
import sys
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

from src.data.image_loading import load_image_bytes_to_rgb
from src.data.preprocessing import CXRPreprocessor
from src.inference.onnx_cxr import OnnxCXRInference
from src.inference.torch_inference import TorchCXRInference

app = FastAPI(title="XRAYNET+", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_torch_engine: Optional[TorchCXRInference] = None
_onnx_engine: Optional[OnnxCXRInference] = None
_preprocessor = CXRPreprocessor()


def get_torch_engine() -> TorchCXRInference:
    global _torch_engine
    if _torch_engine is None:
        _torch_engine = TorchCXRInference()
    return _torch_engine


def get_onnx_engine() -> OnnxCXRInference:
    global _onnx_engine
    if _onnx_engine is None:
        _onnx_engine = OnnxCXRInference()
    return _onnx_engine


class PredictResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    recommendation: str
    clinical_steps: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)
    heatmap_png_base64: str
    backend: str = "pytorch"


class PredictOnnxResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    recommendation: str
    clinical_steps: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)
    backend: str = "onnx"
    explainability_note: str = Field(
        default="Grad-CAM++ is only available via POST /predict (PyTorch).",
        description="ONNX path is for fast edge inference without saliency maps.",
    )


def _allowed_upload(content_type: Optional[str], filename: Optional[str]) -> bool:
    ct = (content_type or "").lower()
    fn = (filename or "").lower()
    if ct.startswith("image/"):
        return True
    if ct in ("application/dicom", "application/octet-stream"):
        return True
    if fn.endswith((".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
        return True
    return False


def _decode_image_bytes(raw: bytes, filename: str) -> np.ndarray:
    try:
        return load_image_bytes_to_rgb(raw, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image/DICOM: {e}") from e


@app.get("/health")
def health():
    return {"status": "ok", "service": "xraynet+"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> Any:
    if not _allowed_upload(file.content_type, file.filename):
        raise HTTPException(
            status_code=400,
            detail="Upload an image (PNG/JPEG/…) or DICOM (.dcm).",
        )
    raw = await file.read()
    arr = _decode_image_bytes(raw, file.filename or "")
    tensor, original_rgb = _preprocessor.preprocess_for_inference(arr)
    if tensor is None:
        raise HTTPException(status_code=400, detail="Preprocessing failed.")

    eng = get_torch_engine()
    out = eng.get_detailed_prediction(tensor, original_rgb)
    heat = out.pop("heatmap_rgb", None)
    if heat is None:
        raise HTTPException(status_code=500, detail="Heatmap generation failed.")

    buf = io.BytesIO()
    Image.fromarray(heat).save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")

    return PredictResponse(
        class_name=out["class_name"],
        confidence=float(out["confidence"]),
        probabilities={k: float(v) for k, v in out["probabilities"].items()},
        recommendation=str(out.get("recommendation", "")),
        clinical_steps=list(out.get("clinical_steps") or []),
        prevention=list(out.get("prevention") or []),
        heatmap_png_base64=b64,
        backend="pytorch",
    )


@app.post("/predict/onnx", response_model=PredictOnnxResponse)
async def predict_onnx(file: UploadFile = File(...)) -> Any:
    if not _allowed_upload(file.content_type, file.filename):
        raise HTTPException(
            status_code=400,
            detail="Upload an image or DICOM (.dcm).",
        )
    try:
        eng = get_onnx_engine()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e

    raw = await file.read()
    arr = _decode_image_bytes(raw, file.filename or "")
    tensor, _orig = _preprocessor.preprocess_for_inference(arr)
    if tensor is None:
        raise HTTPException(status_code=400, detail="Preprocessing failed.")

    out = eng.get_prediction_dict(tensor)
    return PredictOnnxResponse(
        class_name=out["class_name"],
        confidence=float(out["confidence"]),
        probabilities={k: float(v) for k, v in out["probabilities"].items()},
        recommendation=str(out.get("recommendation", "")),
        clinical_steps=list(out.get("clinical_steps") or []),
        prevention=list(out.get("prevention") or []),
        backend="onnx",
    )
