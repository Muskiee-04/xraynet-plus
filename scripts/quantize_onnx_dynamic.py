#!/usr/bin/env python3
"""
Dynamic INT8 weight quantization for smaller/faster CPU ONNX (edge deployment).

Requires: models/saved/xraynet_plus.onnx from scripts/export_onnx_xraynet.py

Output: models/saved/xraynet_plus_int8.onnx

Use with OnnxCXRInference(onnx_path='...xraynet_plus_int8.onnx') or set env XRAYNET_ONNX_PATH.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("Install onnxruntime with quantization support: pip install onnxruntime")
        sys.exit(1)

    inp = os.path.join(_ROOT, "models", "saved", "xraynet_plus.onnx")
    out = os.path.join(_ROOT, "models", "saved", "xraynet_plus_int8.onnx")
    if not os.path.isfile(inp):
        print(f"Missing {inp}. Run: python scripts/export_onnx_xraynet.py")
        sys.exit(1)

    quantize_dynamic(inp, out, weight_type=QuantType.QUInt8)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
