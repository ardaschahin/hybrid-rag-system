from typing import Dict, Any, List, Optional
import re

PROMPT_EXCERPT_LEN = 700
MAX_QUOTE_LEN = 180
MAX_EVIDENCE = 2

# Strict output intent detection (match prompt.py)
YESNO_ONLY_RE = re.compile(
    r"\b(answer\s+yes\/no|answer\s+(yes|no)|reply\s+with\s+only\s+(yes|no)|yes\/no|evet\/hayir|evet\s*mi\s*hayir\s*mi)\b",
    re.IGNORECASE,
)
NUMBER_ONLY_RE = re.compile(
    r"\b(reply\s+with\s+only\s+the\s+number|only\s+the\s+number|sadece\s+sayı|yalnızca\s+sayı|yalnızca\s+rakam)\b",
    re.IGNORECASE,
)


def _clean_ws(s: str) -> str:
    """Normalize whitespace to make substring checks stable."""
    return " ".join((s or "").replace("\n", " ").split()).strip()


def build_source_maps(retrieved: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    m: Dict[int, Dict[str, Any]] = {}
    for i, h in enumerate(retrieved, start=1):
        excerpt = _clean_ws(h.get("text") or "")
        # IMPORTANT: truncate WITHOUT adding "..." to avoid substring surprises
        if len(excerpt) > PROMPT_EXCERPT_LEN:
            excerpt = excerpt[:PROMPT_EXCERPT_LEN]

        md = h.get("metadata") or {}
        m[i] = {
            "chunk_id": h.get("chunk_id"),
            "excerpt": excerpt,
            "kind": md.get("kind", "text"),
            "page": md.get("page"),
        }
    return m


def _is_yesno_only(question: str) -> bool:
    return YESNO_ONLY_RE.search(question or "") is not None


def _is_number_only(question: str) -> bool:
    return NUMBER_ONLY_RE.search(question or "") is not None


def _enforce_strict_answer(payload: Dict[str, Any], question: Optional[str]) -> Dict[str, Any]:
    """
    If question asks for strict YES/NO or number-only, enforce it.
    If not compliant, return fallback payload.
    """
    q = question or ""
    ans = _clean_ws(payload.get("answer") or "")

    if _is_yesno_only(q):
        # Must be exactly YES or NO
        if ans.upper() not in {"YES", "NO"}:
            return {"answer": "I don't have enough information in the provided excerpts.", "evidence": []}
        payload["answer"] = ans.upper()
        return payload

    if _is_number_only(q):
        # Must be only digits
        if not re.fullmatch(r"\d+", ans):
            return {"answer": "I don't have enough information in the provided excerpts.", "evidence": []}
        payload["answer"] = ans
        return payload

    return payload


def validate_evidence(
    payload: Dict[str, Any],
    retrieved: List[Dict[str, Any]],
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Strict evidence validator + strict answer mode enforcement (optional question):
    - If question requests YES/NO only -> answer must be exactly YES or NO
    - If question requests number-only -> answer must be digits only
    - source_id must exist
    - chunk_id must match that SOURCE chunk_id
    - quote <= MAX_QUOTE_LEN
    - quote must be exact substring of excerpt shown to model (after whitespace normalization)
    - dedupe identical quotes
    - keep at most MAX_EVIDENCE
    """
    if not isinstance(payload, dict):
        return {"answer": "I don't have enough information in the provided excerpts.", "evidence": []}

    # Enforce strict output modes first (so we can reject malformed answers early)
    payload = _enforce_strict_answer(payload, question)

    src_map = build_source_maps(retrieved)

    evidence = payload.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = []

    valid: List[Dict[str, Any]] = []
    seen_quotes = set()

    for ev in evidence:
        if not isinstance(ev, dict):
            continue

        try:
            sid = int(ev.get("source_id"))
        except Exception:
            continue

        src = src_map.get(sid)
        if not src:
            continue

        expected_chunk = src.get("chunk_id")
        excerpt = src.get("excerpt") or ""

        if ev.get("chunk_id") != expected_chunk:
            continue

        quote = _clean_ws(ev.get("quote") or "")
        if not quote:
            continue

        if len(quote) > MAX_QUOTE_LEN:
            continue

        # strict substring check
        if quote not in excerpt:
            continue

        if quote in seen_quotes:
            continue
        seen_quotes.add(quote)

        valid.append({
            "source_id": sid,
            "chunk_id": expected_chunk,
            "quote": quote,
        })

        if len(valid) >= MAX_EVIDENCE:
            break

    payload["evidence"] = valid
    return payload
