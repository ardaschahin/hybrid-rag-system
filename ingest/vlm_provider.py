import os
import json
import time
import base64
from typing import Any, Dict, Optional, Tuple, List

import httpx


class VLMError(RuntimeError):
    pass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _extract_json_block(s: str) -> str:
    s = (s or "").strip()
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return s
    return s[i : j + 1]


def _normalize_blacklist() -> List[str]:
    raw = os.getenv("VLM_BLACKLIST", "red amber green|rag system|traffic light|retrieval augmented")
    return [p.strip().lower() for p in raw.split("|") if p.strip()]


def _matches_blacklist(text: str) -> bool:
    t = (text or "").lower()
    for term in _normalize_blacklist():
        if term and term in t:
            return True
    return False


def caption_image(
    png_bytes: bytes,
    *,
    pdf_name: str,
    page_no: int,
    page_text_hint: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Returns (caption_text_or_None, debug_meta)
      - None => do NOT store caption (no_figure / low_conf / blacklisted / invalid_json)
    """
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_VLM_MODEL", "llava:13b")
    timeout_s = _env_float("VLM_TIMEOUT", 240.0)
    min_conf = _env_float("VLM_MIN_CONFIDENCE", 0.60)

    temperature = _env_float("VLM_TEMPERATURE", 0.1)
    num_predict = _env_int("VLM_NUM_PREDICT", 220)

    url = f"{base}/api/generate"
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    hint = ""
    if page_text_hint:
        hint = page_text_hint.strip().replace("\n", " ")
        hint = " ".join(hint.split())[:500]

    # IMPORTANT: general prompt that works for many engineering PDFs:
    # - extract legend, line styles, labels, restrictions
    # - avoid "RAG system" hallucinations etc.
    prompt = (
        "You are extracting information from a PDF PAGE IMAGE for a RAG system.\n"
        "RULES:\n"
        "- Do NOT guess. Only describe what you can SEE.\n"
        "- Focus on diagrams, figures, tables, charts, measurement drawings, labels, arrows, dimensions.\n"
        "- If there is a LEGEND / KEY (colors, hatch patterns), list it.\n"
        "- If there are LINE STYLES (dashed/solid/boundary lines), state what they represent if labeled.\n"
        "- Extract any important on-figure words verbatim when possible (e.g., 'Highway', 'Principal elevation').\n"
        "- If the page has NO meaningful figure/table/diagram (only running text), set has_figure=false.\n"
        "- If uncertain, lower confidence.\n"
        "\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  \"has_figure\": true/false,\n'
        '  \"confidence\": number between 0 and 1,\n'
        '  \"bullets\": [\"...\"]\n'
        "}\n"
        "\n"
        f"PDF: {pdf_name} page {page_no}\n"
        + (f"TEXT_HINT (may help align labels): {hint}\n" if hint else "")
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": num_predict},
    }

    t0 = time.time()
    try:
        r = httpx.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
    except Exception as e:
        raise VLMError(f"VLM request failed: {e}") from e
    dt = time.time() - t0

    raw = (r.json().get("response") or "").strip()
    candidate = _extract_json_block(raw)

    debug: Dict[str, Any] = {
        "pdf": pdf_name,
        "page": page_no,
        "vlm_model": model,
        "latency_s": round(dt, 2),
        "raw_json": candidate,
        "reason": "",
        "confidence": 0.0,
        "has_figure": False,
        "bullets_n": 0,
    }

    try:
        parsed = json.loads(candidate)
    except Exception:
        debug["reason"] = "invalid_json"
        return None, debug

    has_figure = bool(parsed.get("has_figure", False))
    conf = parsed.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0

    bullets = parsed.get("bullets") or []
    if not isinstance(bullets, list):
        bullets = []

    cleaned: List[str] = []
    for b in bullets:
        if isinstance(b, str):
            s = " ".join(b.strip().split())
            if s:
                cleaned.append(s)

    caption_text = "\n".join([f"- {b}" for b in cleaned]).strip()

    debug["has_figure"] = has_figure
    debug["confidence"] = conf
    debug["bullets_n"] = len(cleaned)

    if not has_figure:
        debug["reason"] = "no_figure"
        return None, debug

    if conf < min_conf:
        debug["reason"] = f"low_confidence<{min_conf}"
        return None, debug

    if _matches_blacklist(caption_text):
        debug["reason"] = "blacklisted_terms"
        return None, debug

    if len(caption_text) < 40:
        debug["reason"] = "too_short"
        return None, debug

    debug["reason"] = "accepted"
    return caption_text, debug
