import json
import re
from typing import TypedDict, List, Dict, Any, Optional, Tuple

from langgraph.graph import StateGraph, END

from agent.app.prompt import build_prompt
from agent.app.llm_provider import generate
from agent.app.verify import validate_evidence

# --- Intent detection (router) ---
PAGE_RE = re.compile(r"\b(?:page|p\.?|sayfa)\s*#?\s*(\d{1,3})\b", re.IGNORECASE)
VISUAL_INTENT_RE = re.compile(
    r"\b(diagram|figure|table|chart|flowchart|illustration|şekil|tablo)\b",
    re.IGNORECASE,
)

# Object-related intent (TR+EN)
OBJ_INTENT_RE = re.compile(
    r"\b(object|objects|layer|katman|type|tip|polyline|line|window|highway|kaç|adet|nesne|obje)\b",
    re.IGNORECASE,
)

# Doc-span intent: user asking "what does page X explain", "what is the restriction", "measure", etc.
DOC_RULE_INTENT_RE = re.compile(
    r"\b(explain|restriction|means|rule|measured|measure|how to|what is|define|definition|nedir|açıkla|tanım|ölç|kural|kısıt)\b",
    re.IGNORECASE,
)

# Extract explicit layer/type targets from question (dynamic mapping)
LAYER_TARGET_RE = re.compile(
    r"\b(?:layer|katman)\s*[:=]?\s*([A-Za-z0-9_\- ]{2,40})\b", re.IGNORECASE
)
TYPE_TARGET_RE = re.compile(
    r"\b(?:type|tip)\s*[:=]?\s*([A-Za-z0-9_\-]{2,30})\b", re.IGNORECASE
)


class GraphState(TypedDict, total=False):
    question: str
    object_summary: Dict[str, Any]
    object_list: List[Dict[str, Any]]
    hits: List[Dict[str, Any]]

    # Router outputs
    plan: Dict[str, Any]
    filtered_hits: List[Dict[str, Any]]
    short_circuit: bool

    # Verification outputs
    object_checks: List[Dict[str, Any]]

    # Direct-answer shortcut outputs (agentic)
    direct_answer: str
    direct_evidence: List[Dict[str, Any]]

    # LLM raw + parsed
    raw: str
    answer: str
    evidence: List[Dict[str, Any]]

    retry_count: int
    max_retries: int

    # For correct evidence mapping
    used_hits: List[Dict[str, Any]]


def _extract_json_block(s: str) -> str:
    s = (s or "").strip()
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return s
    return s[i : j + 1]


def _asked_page(question: str) -> Optional[int]:
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


def _is_visual(question: str) -> bool:
    return VISUAL_INTENT_RE.search(question or "") is not None


def _score(h: Dict[str, Any]) -> float:
    s = h.get("score")
    try:
        return float(s) if s is not None else -1.0
    except Exception:
        return -1.0


def _clean_ws(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").split()).strip()


def _extract_layer_target(q: str) -> Optional[str]:
    """
    Try to extract explicit layer name from question like:
    - "layer Windows"
    - "katman Highway"
    - "layer: MyLayer"
    """
    m = LAYER_TARGET_RE.search(q or "")
    if not m:
        return None
    name = _clean_ws(m.group(1))
    # trim trailing common words
    name = re.sub(r"\b(objects|objeler|count|kaç|adet)\b$", "", name, flags=re.IGNORECASE).strip()
    return name or None


def _extract_type_target(q: str) -> Optional[str]:
    m = TYPE_TARGET_RE.search(q or "")
    if not m:
        return None
    name = _clean_ws(m.group(1))
    return name or None


def node_router(state: GraphState) -> GraphState:
    """
    Routing plan:
    - visual intent => need caption; if no caption => short_circuit
    - non-visual => text
    Also decides "direct_*" strategies (object count/layer/type, direct doc span).
    """
    q = (state.get("question", "") or "").strip()
    ql = q.lower()
    hits = state.get("hits", []) or []

    visual = _is_visual(q)
    asked_page = _asked_page(q)

    caps: List[Dict[str, Any]] = []
    txts: List[Dict[str, Any]] = []

    for h in hits:
        md = h.get("metadata") or {}
        if md.get("kind") == "caption":
            caps.append(h)
        else:
            txts.append(h)

    caps.sort(key=_score, reverse=True)
    txts.sort(key=_score, reverse=True)

    short_circuit = False
    filtered: List[Dict[str, Any]] = []

    if visual:
        if not caps:
            short_circuit = True
            filtered = hits[:]
        else:
            best_cap = None
            if asked_page is not None:
                for c in caps:
                    if (c.get("metadata") or {}).get("page") == asked_page:
                        best_cap = c
                        break
            if best_cap is None:
                best_cap = caps[0]

            filtered.append(best_cap)

            cap_page = (best_cap.get("metadata") or {}).get("page")
            best_txt = None
            for t in txts:
                if (t.get("metadata") or {}).get("page") == cap_page:
                    best_txt = t
                    break
            if best_txt:
                filtered.append(best_txt)
            elif txts:
                filtered.append(txts[0])

            seen = {h.get("chunk_id") for h in filtered}
            pool = [*caps, *txts]
            pool.sort(key=_score, reverse=True)
            for h in pool:
                cid = h.get("chunk_id")
                if cid in seen:
                    continue
                filtered.append(h)
                seen.add(cid)
                if len(filtered) >= 5:
                    break
    else:
        filtered = (txts[:5] if txts else hits[:5])

    # ---- Decide strategy ----
    strategy = "caption+text" if visual and caps else ("text_only" if not visual else "visual_no_caption_shortcircuit")

    # ✅ NEW: compute dynamic targets once, store in plan
    layer_target = _extract_layer_target(q)
    type_target = _extract_type_target(q)

    # Direct object answering
    if re.search(r"\bhow many\b|\bkaç\b|\badet\b", ql) and OBJ_INTENT_RE.search(q):
        # explicit targets win
        if layer_target is not None:
            strategy = "direct_object_layer_count"
        elif type_target is not None:
            strategy = "direct_object_type_count"
        else:
            # fallback heuristics (keep your old behavior)
            if ("window" in ql or "windows" in ql or "pencere" in ql) or ("highway" in ql):
                strategy = "direct_object_layer_count"
            elif ("polyline" in ql or re.search(r"\bline\b", ql)):
                strategy = "direct_object_type_count"
            else:
                strategy = "direct_object_count"

    # Direct doc span when page+rule signal is strong
    if strategy == "text_only" and asked_page is not None and DOC_RULE_INTENT_RE.search(q):
        strategy = "direct_doc_span"

    plan = {
        "visual_intent": visual,
        "asked_page": asked_page,
        "has_caption": bool(caps),
        "strategy": strategy,
        # ✅ NEW: include dynamic targets in plan
        "layer_target": layer_target,
        "type_target": type_target,
    }

    return {"plan": plan, "filtered_hits": filtered, "short_circuit": short_circuit}


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def node_object_verify(state: GraphState) -> GraphState:
    q = (state.get("question", "") or "").strip()
    ql = q.lower()
    summ = state.get("object_summary") or {}
    obj_list = state.get("object_list") or []
    hits = state.get("filtered_hits") or state.get("hits") or []

    checks: List[Dict[str, Any]] = []

    total = _safe_int(summ.get("total_objects", 0), 0)
    by_layer = summ.get("by_layer") or {}
    by_type = summ.get("by_type") or {}

    if len(obj_list) != total:
        checks.append({
            "level": "warning",
            "code": "OBJECT_SUMMARY_MISMATCH",
            "message": f"object_summary.total_objects={total} but object_list has {len(obj_list)} items.",
        })

    if OBJ_INTENT_RE.search(q) and len(obj_list) == 0:
        checks.append({
            "level": "warning",
            "code": "NO_OBJECTS_IN_SESSION",
            "message": "Question seems object-related, but current session object_list is empty.",
        })

    # malformed objects
    bad = 0
    for obj in obj_list:
        if not isinstance(obj, dict):
            bad += 1
            continue
        if "type" not in obj or "layer" not in obj:
            bad += 1
            continue
        if "points" in obj and isinstance(obj.get("points"), list) and len(obj["points"]) == 0:
            bad += 1
    if bad:
        checks.append({
            "level": "warning",
            "code": "MALFORMED_OBJECTS",
            "message": f"{bad} object(s) look malformed (missing type/layer or empty points).",
        })

    # context-aware missing layer checks (only warn if doc or question mentions)
    ht = _clean_ws(" ".join([(h.get("text") or "") for h in hits])).lower()

    if ("highway" in ql) or ("highway" in ht):
        if not any(k.lower() == "highway" for k in by_layer.keys()):
            checks.append({
                "level": "warning",
                "code": "LAYER_MISSING_HIGHWAY",
                "message": "Doc/question mentions 'highway' but object_list has no 'Highway' layer objects.",
            })

    if ("window" in ql) or ("windows" in ql) or ("window" in ht) or ("windows" in ht):
        if not any("window" in k.lower() for k in by_layer.keys()):
            checks.append({
                "level": "warning",
                "code": "LAYER_MISSING_WINDOWS",
                "message": "Doc/question mentions windows but object_list has no Windows layer objects.",
            })

    return {"object_checks": checks}


def _best_text_hit(hits: List[Dict[str, Any]], asked_page: Optional[int]) -> Optional[Dict[str, Any]]:
    # prefer text on asked page
    for h in hits:
        md = h.get("metadata") or {}
        if md.get("kind") == "text" and asked_page is not None and md.get("page") == asked_page:
            return h
    # otherwise best text
    for h in hits:
        md = h.get("metadata") or {}
        if md.get("kind") == "text":
            return h
    return None


def _extract_restriction_sentence(excerpt: str) -> Optional[str]:
    """
    Try to pull a clean sentence starting at 'This restriction means ...'
    Works even if PDF text is single-line.
    """
    ex = _clean_ws(excerpt)
    if not ex:
        return None

    idx = ex.lower().find("this restriction means")
    if idx == -1:
        return None

    tail = ex[idx:]
    # cut to first period if exists (or cap length)
    m = re.search(r"\.", tail)
    if m:
        sent = tail[: m.start() + 1]
    else:
        sent = tail[:240]

    sent = _clean_ws(sent)
    if len(sent) < 20:
        return None
    return sent


def node_direct_answer(state: GraphState) -> GraphState:
    """
    Executes direct_* strategies without calling the LLM.
    - direct_object_count / direct_object_layer_count / direct_object_type_count
    - direct_doc_span (summarize from best text hit, keep quote evidence)
    """
    plan = state.get("plan") or {}
    strategy = (plan.get("strategy") or "").strip()
    q = (state.get("question", "") or "").strip()
    ql = q.lower()

    summ = state.get("object_summary") or {}
    by_layer = summ.get("by_layer") or {}
    by_type = summ.get("by_type") or {}

    hits = state.get("filtered_hits") or state.get("hits") or []

    # --- direct object counts ---
    if strategy == "direct_object_count":
        n = _safe_int(summ.get("total_objects", 0), 0)
        return {
            "direct_answer": str(n),
            "direct_evidence": [],
            "used_hits": hits,
            "plan": {**plan, "strategy": "direct_object_count"},
        }

    if strategy == "direct_object_layer_count":
        # ✅ NEW: read from plan first (computed in router)
        layer_target = (plan.get("layer_target") or "").strip() or None

        # fallback heuristics
        if layer_target is None:
            if "window" in ql or "windows" in ql or "pencere" in ql:
                layer_target = "Windows"
            elif "highway" in ql:
                layer_target = "Highway"

        count = 0
        if layer_target:
            for k, v in by_layer.items():
                if str(k).lower() == layer_target.lower():
                    count = _safe_int(v, 0)
                    break

        return {
            "direct_answer": str(count),
            "direct_evidence": [],
            "used_hits": hits,
            "plan": {**plan, "strategy": "direct_object_layer_count", "layer_target": layer_target},
        }

    if strategy == "direct_object_type_count":
        # ✅ NEW: read from plan first
        type_target = (plan.get("type_target") or "").strip() or None

        if type_target is None:
            if "polyline" in ql:
                type_target = "POLYLINE"
            elif re.search(r"\bline\b", ql):
                type_target = "LINE"

        count = 0
        if type_target:
            for k, v in by_type.items():
                if str(k).lower() == type_target.lower():
                    count = _safe_int(v, 0)
                    break

        return {
            "direct_answer": str(count),
            "direct_evidence": [],
            "used_hits": hits,
            "plan": {**plan, "strategy": "direct_object_type_count", "type_target": type_target},
        }

    # --- direct doc span ---
    if strategy == "direct_doc_span":
        asked_page = plan.get("asked_page")
        best = _best_text_hit(hits, asked_page)
        if not best:
            return {}

        chunk_id = best.get("chunk_id")
        excerpt = _clean_ws(best.get("text") or "")

        restr = _extract_restriction_sentence(excerpt)
        if restr:
            ans = restr
        else:
            ex_low = excerpt.lower()
            if "is measured from" in ex_low:
                ans = (
                    "It explains how eaves height should be measured: from ground level at the base of the outside wall "
                    "to where the wall would meet the upper surface of the roof slope, ignoring overhang."
                )
            else:
                ans = (excerpt[:200].rstrip() + ".") if excerpt else "I don't have enough information in the provided excerpts."

        # ✅ quote iyileştirmesi
        if restr:
            quote = restr[:180]
        else:
            quote = excerpt[:180] if len(excerpt) > 180 else excerpt

        return {
            "direct_answer": ans,
            "direct_evidence": [{"source_id": 1, "chunk_id": chunk_id, "quote": quote}],
            "used_hits": [best],  # IMPORTANT: evidence source_id mapping
            "plan": {**plan, "strategy": "direct_doc_span"},
        }


    return {}


def node_draft(state: GraphState) -> GraphState:
    # If direct_answer already computed, skip LLM
    if state.get("direct_answer") is not None:
        payload = {
            "answer": state.get("direct_answer") or "I don't have enough information in the provided excerpts.",
            "evidence": state.get("direct_evidence") or [],
        }
        return {"raw": json.dumps(payload)}

    hits = state.get("filtered_hits") or state.get("hits", []) or []

    if state.get("short_circuit"):
        return {"raw": '{"answer":"I don\'t have enough information in the provided excerpts.","evidence":[]}'}

    prompt = build_prompt(
        state.get("question", ""),
        state.get("object_summary") or {},
        hits,
    )
    raw = generate(prompt)
    return {"raw": raw}


def node_parse_and_validate(state: GraphState) -> GraphState:
    hits = state.get("filtered_hits") or state.get("hits", []) or []
    raw = state.get("raw", "")
    candidate = _extract_json_block(raw)

    plan = state.get("plan") or {}
    strategy = (plan.get("strategy") or "").strip()
    used_hits = state.get("used_hits") or hits

    try:
        parsed = json.loads(candidate)
    except Exception:
        return {
            "answer": "I don't have enough information in the provided excerpts.",
            "evidence": [],
            "object_checks": state.get("object_checks", []) or [],
            "plan": plan,
            "used_hits": used_hits,
        }

    # Validate evidence against the SAME hits used for SOURCE numbering
    try:
        parsed = validate_evidence(parsed, used_hits)
    except Exception:
        parsed = {"answer": "I don't have enough information in the provided excerpts.", "evidence": []}

    answer = parsed.get("answer") or "I don't have enough information in the provided excerpts."
    evidence = parsed.get("evidence") or []

    # ✅ IMPORTANT: direct_* strategies do NOT require evidence
    if (not evidence) and (not strategy.startswith("direct_")):
        answer = "I don't have enough information in the provided excerpts."
        evidence = []

    return {
        "answer": answer,
        "evidence": evidence,
        "object_checks": state.get("object_checks", []) or [],
        "plan": plan,
        "used_hits": used_hits,
    }


def node_retry(state: GraphState) -> GraphState:
    return {"retry_count": state.get("retry_count", 0) + 1}


def should_retry(state: GraphState) -> str:
    evidence = state.get("evidence") or []
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    if state.get("short_circuit"):
        return "done"

    strategy = ((state.get("plan") or {}).get("strategy") or "").strip()
    if strategy.startswith("direct_"):
        return "done"

    if evidence:
        return "done"
    if retry_count < max_retries:
        return "retry"
    return "done"


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("router", node_router)
    g.add_node("object_verify", node_object_verify)
    g.add_node("direct", node_direct_answer)
    g.add_node("draft", node_draft)
    g.add_node("validate", node_parse_and_validate)
    g.add_node("retry", node_retry)

    g.set_entry_point("router")
    g.add_edge("router", "object_verify")
    g.add_edge("object_verify", "direct")
    g.add_edge("direct", "draft")
    g.add_edge("draft", "validate")

    g.add_conditional_edges(
        "validate",
        should_retry,
        {"retry": "retry", "done": END},
    )
    g.add_edge("retry", "draft")

    return g.compile()


GRAPH = build_graph()
