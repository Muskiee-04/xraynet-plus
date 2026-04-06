"""
Google Gemini API integration for educational narratives and case chat.

Uses the Gemini Developer API (Google AI Studio). Set GEMINI_API_KEY.
Optional: GEMINI_MODEL (default gemini-2.0-flash).

Not for clinical diagnosis; outputs are decision-support / education only.
"""
from __future__ import annotations

import os
from typing import Any, Optional

_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

_MEDICAL_PREFIX = """You are assisting with an educational radiology decision-support demo.
Rules:
- Do NOT state a definitive medical diagnosis. Frame everything as possibilities aligned with the provided model probabilities and context.
- Emphasize uncertainty, need for qualified clinician review, and correlation with history/exam/labs.
- If the user asks for treatment prescriptions or dosing, refuse and redirect to a licensed clinician.
- Be concise but thorough. Use clear sections when asked for structured output.
"""


def _client_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def configure_from_key(api_key: Optional[str]) -> None:
    """If api_key is set, configure the SDK (also sets process env for this session)."""
    if not api_key or not str(api_key).strip():
        return
    key = str(api_key).strip()
    os.environ["GEMINI_API_KEY"] = key
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
    except ImportError as e:
        raise RuntimeError("Install google-generativeai: pip install google-generativeai") from e


def _ensure_configured() -> None:
    if not _client_available():
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Add it in the sidebar, a .env file, or your environment."
        )
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"].strip())


def build_case_context_text(
    patient_data: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    """Structured text from local model outputs only (no pixel data)."""
    lines = [
        "=== Case context (from local CNN + Grad-CAM++ demo) ===",
        f"Patient ID: {patient_data.get('patient_id', 'N/A')}",
        f"Age: {patient_data.get('age', 'N/A')}, Gender: {patient_data.get('gender', 'N/A')}",
        f"Clinical notes (user-entered): {patient_data.get('clinical_notes', 'N/A')}",
        "",
        "Per-image model outputs:",
    ]
    for i, r in enumerate(results, 1):
        pred = r.get("prediction") or {}
        probs = pred.get("probabilities") or {}
        prob_str = ", ".join(f"{k}: {float(v):.3f}" for k, v in sorted(probs.items(), key=lambda x: -x[1]))
        lines.append(
            f"  Image {i} ({r.get('filename', 'unknown')}): "
            f"top_class={pred.get('class_name')}, confidence={float(pred.get('confidence', 0)):.3f}; "
            f"probabilities: {prob_str}"
        )
        lines.append(f"    Summary line: {pred.get('recommendation', '')}")
    return "\n".join(lines)


def generate_case_interpretation(
    patient_data: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    extra_instructions: str = "",
) -> str:
    """
    Single-shot narrative: differential-style discussion, limitations, and suggested follow-up themes.
    """
    _ensure_configured()
    import google.generativeai as genai

    ctx = build_case_context_text(patient_data, results)
    prompt = f"""{_MEDICAL_PREFIX}

{ctx}

Task: Write a structured response with these headings (markdown):
## Plain-language summary
## How to read the model probabilities (uncertainty)
## Differential considerations (non-diagnostic)
## What to discuss with a radiologist or treating clinician
## Limitations of AI screening on chest X-rays

Keep total length under ~900 words unless the case is multi-image with conflicting tops.
{extra_instructions}
"""
    model = genai.GenerativeModel(_DEFAULT_MODEL)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip() or "(Empty response from model.)"


def generate_radiology_question_suggestions(
    patient_data: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    """Short bullet list of questions a patient or trainee might ask a clinician."""
    _ensure_configured()
    import google.generativeai as genai

    ctx = build_case_context_text(patient_data, results)
    prompt = f"""{_MEDICAL_PREFIX}

{ctx}

Task: Produce 8–12 bullet questions (for a patient or trainee) to ask a qualified clinician or radiologist.
No diagnosis. One short bullet per line, markdown list.
"""
    model = genai.GenerativeModel(_DEFAULT_MODEL)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip() or "(Empty response.)"


def generate_heatmap_explanation_plain_language(
    patient_data: dict[str, Any],
    result: dict[str, Any],
) -> str:
    """Explain what Grad-CAM-style heatmaps mean for this single image."""
    _ensure_configured()
    import google.generativeai as genai

    mini = build_case_context_text(patient_data, [result])
    prompt = f"""{_MEDICAL_PREFIX}

{mini}

Task: In 2–4 short paragraphs, explain what a class-activation heatmap shows on a chest X-ray in this demo,
why it is NOT the same as a radiologist's read, and how to avoid over-interpreting bright regions.
"""
    model = genai.GenerativeModel(_DEFAULT_MODEL)
    resp = model.generate_content(prompt)
    return (resp.text or "").strip() or "(Empty response.)"


def chat_reply_with_history(
    user_message: str,
    patient_data: dict[str, Any],
    results: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> str:
    """
    Multi-turn chat using google-generativeai chat session.
    history items: {"role": "user"|"model", "parts": str}
    """
    _ensure_configured()
    import google.generativeai as genai

    ctx = build_case_context_text(patient_data, results)
    model = genai.GenerativeModel(_DEFAULT_MODEL)

    starter = [
        {"role": "user", "parts": f"{_MEDICAL_PREFIX}\n\nCase context:\n{ctx}"},
        {
            "role": "model",
            "parts": "Understood. I will provide educational support only, emphasize uncertainty, "
            "and avoid definitive diagnosis or prescribing.",
        },
    ]
    gemini_history = starter + [{"role": h["role"], "parts": h["parts"]} for h in history if h.get("parts")]
    chat = model.start_chat(history=gemini_history)
    resp = chat.send_message(user_message)
    return (resp.text or "").strip() or "(Empty response.)"


def multimodal_compare_image(
    image_bytes: bytes,
    _mime_type: str,
    patient_data: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    user_prompt: str = "Comment on this chest imaging study in an educational way, "
    "consistent with the provided model output. Do not give a definitive diagnosis.",
) -> str:
    """
    Sends image bytes to Gemini (user must accept privacy implications).
    mime_type: e.g. image/png, image/jpeg
    """
    _ensure_configured()
    import google.generativeai as genai

    ctx = build_case_context_text(patient_data, results)
    import PIL.Image
    import io

    img = PIL.Image.open(io.BytesIO(image_bytes))
    model = genai.GenerativeModel(_DEFAULT_MODEL)
    prompt = f"""{_MEDICAL_PREFIX}

{ctx}

User request: {user_prompt}

Remember: correlate any visual impressions with the numeric class probabilities above; note limitations.
"""
    resp = model.generate_content([prompt, img])
    return (resp.text or "").strip() or "(Empty response.)"


def is_configured() -> bool:
    return _client_available()
