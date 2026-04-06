# ChestRay Gemini

Explainable chest X-ray screening (TB, pneumonia, COVID-19, no findings) with **local EfficientNet + Grad-CAM++**, plus an optional **Google Gemini** copilot for educational narratives, chat, clinician-style question prompts, and (with explicit consent) vision-assisted commentary.

This line of work is a **separate product** from the original [xraynet-](https://github.com/Muskiee-04/xraynet-) repository: same core model stack, but Gemini integration, rebranded UI/PDFs, and extra API routes.

## Features

- **Streamlit** — Upload PNG/JPEG/WebP/TIFF/BMP or DICOM; heatmaps; PDF reports; SQLite admin; Gemini tabs (narrative, questions, heatmap explainer, chat, optional vision).
- **FastAPI** — `POST /predict`, `POST /predict/onnx`, plus `GET /gemini/status`, `POST /gemini/interpret`, `POST /gemini/chat` when `GEMINI_API_KEY` is set on the server.
- **Training / ONNX** — Same scripts as the upstream project (`scripts/train_finetune.py`, export/quantize, Docker).

## Quick start

```bash
pip install -r requirements.txt
python scripts/init_demo_model.py   # optional: starter .pth
export GEMINI_API_KEY="your-key"    # optional; or use sidebar / Streamlit secrets
streamlit run app/main.py
```

**Gemini API key (free tier):** create one at [Google AI Studio](https://aistudio.google.com/apikey).

- **Environment:** `export GEMINI_API_KEY=...`
- **Streamlit secrets:** `.streamlit/secrets.toml` with `GEMINI_API_KEY = "..."`
- **UI:** Sidebar → “Google Gemini (optional)” (session only; do not commit keys)

**Model name:** override with `GEMINI_MODEL` (default `gemini-2.0-flash`).

**SSL on macOS/conda:** `src/utils/ssl_setup.py` runs before ImageNet weight downloads (certifi, then dev fallback).

### API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Gemini endpoints require `GEMINI_API_KEY` in the server environment.

## Publish as a **new** GitHub repository

1. On GitHub, create an empty repository (e.g. `chestray-gemini`). Do **not** add a README if you will push an existing branch.
2. From this project folder (e.g. on branch `geminiapi`):

```bash
git remote add chestray https://github.com/YOUR_USER/chestray-gemini.git
git push -u chestray geminiapi:main
```

Or rename the default branch and set a single `origin` if you no longer need the old remote:

```bash
git remote remove origin   # only if you intend to detach from xraynet-
git remote add origin https://github.com/YOUR_USER/chestray-gemini.git
git push -u origin geminiapi:main
```

## Data layout (fine-tune / evaluation)

See `data/xray_finetune/DATA_LAYOUT.txt`.

```bash
python scripts/evaluate_model.py --data-dir data/xray_finetune --split val
```

## NIH Chest X-ray14 (Kaggle)

`data/nih_chest_xray/KAGGLE_SETUP.txt`.

```bash
python scripts/train_nih_xraynet.py --nih-root data/nih_chest_xray --epochs 5 --max-per-class 2000 --class-weights
```

**Unified fine-tune:** `data/UNIFIED_FINETUNE_SETUP.txt`.

## Privacy note (Gemini)

Text features send **model outputs and optional user-entered clinical notes** to Google. **Vision** sends **image pixels** when you opt in. Do not use identifiable patient data without appropriate authorization and policy review.

## Disclaimer

For research and decision-support demos only — not a certified medical device. Gemini outputs are educational, not a radiology report. Clinical use requires validation, regulatory clearance, and qualified oversight.
