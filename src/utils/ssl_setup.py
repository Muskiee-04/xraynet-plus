"""
Fix urllib HTTPS for torchvision weight downloads on macOS/conda setups where
certificate verification fails. Tries certifi, then system defaults, then
TLS without verification (local dev only — not for untrusted networks).
"""
from __future__ import annotations

import ssl
import urllib.request

_DONE = False

# PyTorch hosts EfficientNet-B0 weights here; lightweight HEAD probe.
_PROBE_URL = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"


def _head_ok(ctx: ssl.SSLContext) -> None:
    req = urllib.request.Request(_PROBE_URL, method="HEAD")
    urllib.request.urlopen(req, context=ctx, timeout=30)


def apply_ssl_compatibility() -> None:
    global _DONE
    if _DONE:
        return
    _DONE = True

    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        _head_ok(ctx)
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
        return
    except Exception:
        pass

    try:
        ctx = ssl.create_default_context()
        _head_ok(ctx)
        return
    except Exception:
        pass

    ssl._create_default_https_context = ssl._create_unverified_context
