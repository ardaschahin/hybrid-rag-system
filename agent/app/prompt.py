import json
import re
from typing import List, Dict, Any

VISUAL_INTENT_RE = re.compile(
    r"\b(diagram|figure|table|chart|flowchart|illustration|şekil|tablo)\b",
    re.IGNORECASE,
)

PAGE_RE = re.compile(r"\b(?:page|p\.?|sayfa)\s*#?\s*(\d{1,3})\b", re.IGNORECASE)

# NEW: strict output-intent detection
YESNO_ONLY_RE = re.compile(
    r"\b(answer\s+yes\/no|answer\s+(yes|no)|reply\s+with\s+only\s+(yes|no)|yes\/no|evet\/hayir|evet\s*mi\s*hayir\s*mi)\b",
    re.IGNORECASE,
)
NUMBER_ONLY_RE = re.compile(
    r"\b(reply\s+with\s+only\s+the\s+number|only\s+the\s+number|sadece\s+sayı|yalnızca\s+sayı|yalnızca\s+rakam)\b",
    re.IGNORECASE,
)

MAX_EXCERPT = 700
MAX_QUOTE = 180
MAX_CANDIDATES = 8


def _clean_ws(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").split()).strip()


def _asked_page(question: str) -> int | None:
    m = PAGE_RE.search(question or "")
    if not m:
        return None
    try:
        p = int(m.group(1))
        if 1 <= p <= 999:
            return p
    except Exception:
        return None
    return None


def _anchor_spans(ex: str, anchors: List[str]) -> List[str]:
    out: List[str] = []
    low = ex.lower()
    for a in anchors:
        idx = low.find(a.lower())
        if idx == -1:
            continue
        start = max(0, idx - 40)
        end = min(len(ex), idx + len(a) + 120)
        cand = _clean_ws(ex[start:end])
        if len(cand) > MAX_QUOTE:
            cand = cand[:MAX_QUOTE]
        if cand:
            out.append(cand)
    return out


def _make_quote_candidates(excerpt: str) -> List[str]:
    ex = _clean_ws(excerpt)
    if not ex:
        return []

    cands: List[str] = []

    # 1) caption bulletlarına göre ayır
    if " - " in ex:
        parts = ex.split(" - ")
        for p in parts:
            p = _clean_ws(p)
            if not p:
                continue
            if p.lower().startswith("figure/table caption"):
                continue
            if len(p) > MAX_QUOTE:
                p = p[:MAX_QUOTE]
            cands.append(p)
            if len(cands) >= MAX_CANDIDATES:
                break

    # 2) Anchor'lar (restriction/measurement gibi yerlerde işe yarıyor)
    if len(cands) < 3:
        anchors = [
            "This restriction means that",
            "will require an application for planning permission",
            "Eaves height is measured",
            "obscure glazed to minimum of level 3",
            "There will only be one principal elevation",
            "the principal elevation will be what is understood",
        ]
        cands.extend(_anchor_spans(ex, anchors))

    # 3) yoksa sentence split
    if not cands:
        parts = re.split(r"(?<=[\.\!\?])\s+|;\s+", ex)
        for p in parts:
            p = _clean_ws(p)
            if not p:
                continue
            if len(p) > MAX_QUOTE:
                p = p[:MAX_QUOTE]
            cands.append(p)
            if len(cands) >= MAX_CANDIDATES:
                break

    # de-dupe
    out = []
    seen = set()
    for c in cands:
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= MAX_CANDIDATES:
            break

    return out[:MAX_CANDIDATES]


def build_prompt(question: str, object_summary: Dict[str, Any], retrieved: List[Dict[str, Any]]) -> str:
    ctx_blocks: list[str] = []
    has_caption = False

    for i, h in enumerate(retrieved, start=1):
        md = h.get("metadata") or {}
        filename = md.get("filename", "unknown")
        page = md.get("page", "unknown")
        section = md.get("section", None)
        kind = md.get("kind", "text")
        if kind == "caption":
            has_caption = True

        chunk_id = h.get("chunk_id", f"chunk_{i}")

        # IMPORTANT: truncate WITHOUT adding "..."
        text = _clean_ws(h.get("text") or "")
        if len(text) > MAX_EXCERPT:
            text = text[:MAX_EXCERPT]

        quote_candidates = _make_quote_candidates(text)
        qc_lines = "\n".join([f"  - {c}" for c in quote_candidates]) if quote_candidates else "  - (none)"

        ctx_blocks.append(
            f"SOURCE {i}\n"
            f"kind: {kind}\n"
            f"chunk_id: {chunk_id}\n"
            f"file: {filename}\n"
            f"page: {page}\n"
            f"section: {section}\n"
            f"excerpt: {text}\n"
            f"quote_candidates (copy EXACTLY one of these as evidence.quote):\n"
            f"{qc_lines}"
        )

    context = "\n\n".join(ctx_blocks) if ctx_blocks else "NO_EXCERPTS_FOUND"

    # Allow evidence to be empty for purely object-based direct answers,
    # but for any document claim we need evidence.
    schema = {
        "answer": "string (max 3 sentences, unless strict YES/NO or number-only is requested)",
        "evidence": [{"source_id": 1, "chunk_id": "string", "quote": "string (<=180 chars)"}],
    }

    obj_json = json.dumps(object_summary or {}, ensure_ascii=False)
    visual_intent = VISUAL_INTENT_RE.search(question or "") is not None
    asked_page = _asked_page(question)

    yesno_only = YESNO_ONLY_RE.search(question or "") is not None
    number_only = NUMBER_ONLY_RE.search(question or "") is not None

    return f"""You are a hybrid RAG QA agent.

Return ONLY valid JSON (no markdown, no backticks).

OBJECT_SUMMARY (session-only; OK to use it for object counting/presence, NOT for document facts):
{obj_json}

SOURCE KIND:
- kind=text    => extracted PDF text
- kind=caption => generated from a page image (diagram/table/figure)

STRICT OUTPUT MODES:
- If the question requests YES/NO only (yesno_only={str(yesno_only).lower()}), then answer MUST be exactly "YES" or "NO" (no extra words).
- If the question requests number-only (number_only={str(number_only).lower()}), then answer MUST be only digits (e.g., "3"), nothing else.

DOCUMENT RULES:
- Use ONLY the SOURCES for factual claims about the document.
- If asked_page={asked_page} is not None, prefer SOURCES from that page.
- If visual_intent=true and there is at least one kind=caption source:
  - Answer MUST primarily summarize caption content.
- Use kind=caption for "what is shown"; use kind=text for rules/measurements/definitions.

OBJECT RULES:
- You may use OBJECT_SUMMARY for: counts, presence/absence per layer/type, simple aggregations.
- If you mention object facts, they MUST be consistent with OBJECT_SUMMARY.

EVIDENCE RULES:
- Evidence is required for any document-derived claim.
- evidence.quote MUST be copied EXACTLY from quote_candidates (preferred) OR an exact substring of excerpt.
- quote length <= 180 chars.
- Max 2 evidence items.
- evidence.source_id must match SOURCE number.
- evidence.chunk_id must match that SOURCE chunk_id.
- If the question explicitly asks for a quote from page X, provide at least one evidence item from that page if available.

IMPORTANT (visual questions):
- If the question asks about a diagram/figure/table AND there is NO kind=caption source, output EXACTLY:
  {{ "answer": "I don't have enough information in the provided excerpts.", "evidence": [] }}

If you cannot support any document claim with an exact quote, output EXACTLY:
{{ "answer": "I don't have enough information in the provided excerpts.", "evidence": [] }}

Question visual_intent={str(visual_intent).lower()} has_caption={str(has_caption).lower()}

JSON schema:
{json.dumps(schema, indent=2)}

SOURCES:
{context}

Question:
{question}

Return JSON only:
"""
