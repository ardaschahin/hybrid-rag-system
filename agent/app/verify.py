from typing import Dict, Any, List

PROMPT_EXCERPT_LEN = 700
MAX_QUOTE_LEN = 180
MAX_EVIDENCE = 2


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
        }
    return m


def validate_evidence(payload: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Strict evidence validator:
    - source_id must exist
    - chunk_id must match that SOURCE chunk_id
    - quote <= MAX_QUOTE_LEN
    - quote must be exact substring of excerpt shown to model (after whitespace normalization)
    - dedupe identical quotes
    - keep at most MAX_EVIDENCE
    """
    if not isinstance(payload, dict):
        return {"answer": "I don't have enough information in the provided excerpts.", "evidence": []}

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
