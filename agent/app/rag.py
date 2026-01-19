import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from fastembed import TextEmbedding

# --- Optional telemetry disable (version-safe) ---
Settings = None
try:
    from chromadb.config import Settings as _Settings  # most common
    Settings = _Settings
except Exception:
    try:
        from chromadb.config import Settings as _Settings  # same path, keep
        Settings = _Settings
    except Exception:
        Settings = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docs")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_EMBEDDER = TextEmbedding(model_name=EMBED_MODEL)

# --- Lazy singletons (safe with uvicorn startup/reload) ---
_CLIENT = None
_COLLECTION = None

def _get_collection():
    global _CLIENT, _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION

    if Settings is not None:
        _CLIENT = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    else:
        # fallback: no settings
        _CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)

    _COLLECTION = _CLIENT.get_or_create_collection(name=COLLECTION_NAME)
    return _COLLECTION


CLASS_Q_RE = re.compile(r"\bClass\s+([A-Z])\b", re.IGNORECASE)
CLASS_COVERS_RE = re.compile(r"\bClass\s+([A-Z])\s+covers\b", re.IGNORECASE)
CLASS_HEADING_RE = re.compile(r"\bClass\s+([A-Z])\s*[–-]\s*", re.IGNORECASE)

PAGE_RE = re.compile(r"\b(?:page|p\.?|sayfa)\s*#?\s*(\d{1,3})\b", re.IGNORECASE)

VISUAL_INTENT_RE = re.compile(
    r"\b(diagram|figure|table|chart|flowchart|illustration|şekil|tablo)\b",
    re.IGNORECASE,
)

REF_PAGE_RE = re.compile(r"\bsee\s+page\s+(\d{1,3})\b", re.IGNORECASE)

ENABLE_KEYWORD_BOOST = os.getenv("ENABLE_KEYWORD_BOOST", "1").strip() == "1"
ENABLE_FOLLOW_LINKS = os.getenv("ENABLE_FOLLOW_LINKS", "0").strip() == "1"
ENABLE_LEXICAL_BONUS = os.getenv("ENABLE_LEXICAL_BONUS", "1").strip() == "1"

STOPWORDS = {
    "the", "and", "or", "for", "with", "from", "that", "this", "into", "over",
    "page", "sayfa", "p", "figure", "diagram", "table", "chart", "flowchart",
    "what", "does", "show", "summarize", "in", "on", "of", "to", "a", "an",
    "is", "are", "it", "as", "at", "by", "be", "give", "sentences", "sentence",
    "ve", "ile", "bir", "bu", "şu", "icin", "için", "olarak", "mi", "mı", "mu", "mü",
    "ne", "nedir", "acikla", "açıkla",
}

def asked_class_from_question(question: str) -> Optional[str]:
    m = CLASS_Q_RE.search(question or "")
    if not m:
        return None
    return f"Class {m.group(1).upper()}"

def asked_page_from_question(question: str) -> Optional[int]:
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

def has_visual_intent(question: str) -> bool:
    return VISUAL_INTENT_RE.search(question or "") is not None

def normalize_query(query: str) -> str:
    asked = asked_class_from_question(query)
    if not asked:
        return query
    ql = query.lower()
    if asked.lower() in ql:
        return query
    return f"{asked}. {query}"

def _boost_definition_like_chunks(asked_class: str, text: str) -> float:
    t = (text or "").strip()
    m = CLASS_HEADING_RE.search(t)
    if m:
        mentioned = f"Class {m.group(1).upper()}"
        if mentioned == asked_class:
            return 0.12
    return 0.0

def _penalize_cross_class_chunks(asked_class: str, text: str) -> float:
    t = (text or "").lower()
    m = CLASS_COVERS_RE.search(text or "")
    if m:
        mentioned = f"Class {m.group(1).upper()}"
        if mentioned != asked_class:
            return 0.35
    if asked_class.lower() in t:
        return 0.0
    if "roof" in t and ("class b" in t or "class c" in t):
        return 0.25
    return 0.0

def _and_where(*conds: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conds = [c for c in conds if c]
    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}

def _query(collection, q_vec, n: int, where: Optional[Dict[str, Any]]):
    return collection.query(
        query_embeddings=[q_vec],
        n_results=n,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

def _to_candidates(res) -> List[Dict[str, Any]]:
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        dist = dists[i] if i < len(dists) else None
        base_score = None if dist is None else float(1.0 / (1.0 + dist))
        out.append({
            "chunk_id": ids[i],
            "text": docs[i] or "",
            "metadata": metas[i] or {},
            "score": base_score,
        })
    return out

def _query_terms(query: str) -> List[str]:
    q = (query or "").lower()
    words = re.findall(r"[a-zA-ZğüşöçıİĞÜŞÖÇ0-9]+", q)
    terms = []
    for w in words:
        w = w.strip().lower()
        if len(w) < 4:
            continue
        if w in STOPWORDS:
            continue
        terms.append(w)
    seen = set()
    out = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:12]

def _keyword_boost_score(query: str, text: str) -> float:
    if not ENABLE_KEYWORD_BOOST:
        return 0.0
    terms = _query_terms(query)
    if not terms:
        return 0.0
    tl = (text or "").lower()
    hits = 0
    for t in terms:
        if t in tl:
            hits += 1
    return min(0.18, 0.03 * hits)

def _tokenize_lex(s: str) -> List[str]:
    s = (s or "").lower()
    toks = re.findall(r"[a-z0-9ğüşöçı]+", s)
    out = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in STOPWORDS:
            continue
        out.append(t)
    return out[:24]

def _lexical_bonus_score(query: str, text: str) -> float:
    if not ENABLE_LEXICAL_BONUS:
        return 0.0
    qt = _tokenize_lex(query)
    if not qt:
        return 0.0
    tl = (text or "").lower()
    hits = 0
    for t in qt:
        if t in tl:
            hits += 1
    ratio = hits / max(1, len(qt))
    return 0.25 * ratio

def _apply_keyword_rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates

    if not (ENABLE_KEYWORD_BOOST or ENABLE_LEXICAL_BONUS):
        candidates.sort(key=lambda x: (x["score"] is not None, x["score"]), reverse=True)
        return candidates

    for c in candidates:
        s = c.get("score")
        base = float(s) if s is not None else 0.0
        kb = _keyword_boost_score(query, c.get("text") or "")
        lb = _lexical_bonus_score(query, c.get("text") or "")
        c["score"] = base + kb + lb

    candidates.sort(key=lambda x: (x["score"] is not None, x["score"]), reverse=True)
    return candidates

def _follow_link_pages(text: str) -> List[int]:
    if not ENABLE_FOLLOW_LINKS:
        return []
    out = []
    for m in REF_PAGE_RE.finditer(text or ""):
        try:
            p = int(m.group(1))
            if 1 <= p <= 999:
                out.append(p)
        except Exception:
            continue
    seen = set()
    res = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        res.append(p)
    return res[:2]

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    collection = _get_collection()

    page_intent = asked_page_from_question(query)
    visual_intent = has_visual_intent(query)
    asked_class = asked_class_from_question(query)

    query_norm = query
    if asked_class and page_intent is None:
        query_norm = normalize_query(query)

    q_vec = list(_EMBEDDER.embed([query_norm]))[0]
    raw_k = max(top_k * 6, 18)

    if page_intent is not None:
        caption_where = _and_where({"page": page_intent}, {"kind": "caption"})
        text_where = _and_where({"page": page_intent}, {"kind": "text"})
        any_where = {"page": page_intent}

        caps = _to_candidates(_query(collection, q_vec, raw_k, caption_where)) if visual_intent else []
        txts = _to_candidates(_query(collection, q_vec, raw_k, text_where))

        caps = _apply_keyword_rerank(query, caps)
        txts = _apply_keyword_rerank(query, txts)

        if visual_intent and not caps:
            mix = _to_candidates(_query(collection, q_vec, raw_k, any_where))
            mix = _apply_keyword_rerank(query, mix)
            return mix[:top_k]

        merged: List[Dict[str, Any]] = []
        if caps:
            merged.append(caps[0])
        if txts:
            merged.append(txts[0])

        seen = {c["chunk_id"] for c in merged}
        pool = [*caps[1:], *txts[1:]]
        pool = _apply_keyword_rerank(query, pool)
        for c in pool:
            if c["chunk_id"] in seen:
                continue
            merged.append(c)
            seen.add(c["chunk_id"])
            if len(merged) >= top_k:
                break

        if ENABLE_FOLLOW_LINKS and merged:
            ref_pages = []
            for c in merged[:2]:
                ref_pages.extend(_follow_link_pages(c.get("text") or ""))
            for rp in ref_pages:
                extra = _to_candidates(_query(collection, q_vec, max(4, top_k), {"page": rp}))
                extra = _apply_keyword_rerank(query, extra)
                for e in extra:
                    if e["chunk_id"] in seen:
                        continue
                    merged.append(e)
                    seen.add(e["chunk_id"])
                    if len(merged) >= max(top_k, 6):
                        break

        merged = _apply_keyword_rerank(query, merged)
        return merged[:top_k]

    where = None
    if asked_class:
        where = _and_where({"section": asked_class}, {"kind": "text"})

    res = _query(collection, q_vec, raw_k, where)
    candidates = _to_candidates(res)

    if asked_class:
        for c in candidates:
            text = c.get("text") or ""
            penalty = _penalize_cross_class_chunks(asked_class, text)
            boost = _boost_definition_like_chunks(asked_class, text)
            s = c.get("score")
            if s is not None:
                c["score"] = max(0.0, float(s) - penalty + boost)

    candidates = _apply_keyword_rerank(query, candidates)
    return candidates[:top_k]
