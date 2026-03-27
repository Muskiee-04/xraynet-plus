# XrayNet+

Offline-first, explainable AI for thoracic disease screening from chest X-rays (TB, pneumonia, COVID-19, no findings).

## Features

- **Streamlit UI** — Upload **PNG/JPEG/WebP/TIFF/BMP or DICOM (`.dcm`)**, view predictions, confidence, Grad-CAM++ overlays, PDF reports, SQLite admin.
- **FastAPI** — `POST /predict` (PyTorch + heatmap), `POST /predict/onnx` (ONNX, fast CPU edge path without saliency).
- **Training** — `scripts/train_finetune.py` for folder-based fine-tuning; `scripts/evaluate_model.py` for validation metrics.
- **Deployment** — ONNX export, optional INT8 quantization, Docker, env overrides (`XRAYNET_WEIGHTS`, `XRAYNET_ONNX_PATH`).

## Quick start

```bash
pip install -r requirements.txt
python scripts/init_demo_model.py   # optional: save starter .pth
streamlit run app/main.py
```

**SSL errors on macOS/conda:** the app runs `src/utils/ssl_setup.py` before downloading ImageNet weights (tries certifi, then falls back to unverified TLS for local dev only so PyTorch model URLs work).

API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Data layout (fine-tune / evaluation)

See `data/xray_finetune/DATA_LAYOUT.txt`. Evaluate on a labeled `val/` tree:

```bash
python scripts/evaluate_model.py --data-dir data/xray_finetune --split val
```

## NIH Chest X-ray14 (Kaggle)

Download instructions: `data/nih_chest_xray/KAGGLE_SETUP.txt`.

```bash
python scripts/train_nih_xraynet.py --nih-root data/nih_chest_xray --epochs 5 --max-per-class 2000 --class-weights
```

NIH does not contain true TB or COVID labels; the script maps findings to the 4 app classes using documented proxies (see `src/data/nih_xraynet_dataset.py`).

**Unified fine-tune (NIH + TB + COVID Kaggle):** see `data/UNIFIED_FINETUNE_SETUP.txt`.

```bash
python scripts/train_unified_finetune.py --nih-root data/nih_chest_xray --tb-root data/kaggle_tb_chest --covid-root data/kaggle_covid_chest --dry-run
python scripts/train_unified_finetune.py --nih-root data/nih_chest_xray --tb-root data/kaggle_tb_chest --covid-root data/kaggle_covid_chest --epochs 10 --class-weights --cap-per-class 8000
```

## Disclaimer

For research and decision-support demos only — not a certified medical device. Clinical use requires validation, regulatory clearance, and qualified oversight.
