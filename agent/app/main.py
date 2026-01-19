from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
from fastapi import FastAPI
from pydantic import BaseModel

from agent.app.rag import retrieve
from agent.app.prompt import build_prompt
from agent.app.llm_provider import generate
from agent.app.verify import validate_evidence
from agent.app.graph import GRAPH

import traceback
from fastapi.responses import PlainTextResponse

app = FastAPI(title="AICI Agent Service", version="0.8.2")


@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    return PlainTextResponse(traceback.format_exc(), status_code=500)


class AnswerRequest(BaseModel):
    question: str
    object_list: list[dict] | None = None
    top_k: int | None = 2
    evidence_mode: bool | None = True
    quote_mode: bool | None = True


def _extract_json_block(s: str) -> str:
    s = (s or "").strip()
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return s
    return s[i : j + 1]


def _meta() -> dict:
    """
    Return provider/model info from environment so responses stay accurate
    when you switch models/providers in docker-compose/.env.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip()
    # try multiple common env names
    model = (
        os.getenv("OLLAMA_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL_NAME")
        or "unknown"
    )
    return {"provider": provider, "model": model}


_PAGE_RE = re.compile(r"\b(?:page|p\.?|sayfa)\s*#?\s*(\d{1,3})\b", re.IGNORECASE)

# Heuristic: pure object counting questions -> no retrieval needed
_OBJ_COUNT_RE = re.compile(r"\b(how many|kaç|adet)\b", re.IGNORECASE)
_OBJ_HINT_RE = re.compile(
    r"\b(object|objects|layer|katman|type|tip|polyline|line|window|windows|highway|nesne|obje)\b",
    re.IGNORECASE,
)

# optional env toggle (default ON)
_DISABLE_RETRIEVAL_FOR_OBJECT_Q = (
    os.getenv("AICI_DISABLE_RETRIEVAL_FOR_OBJECT_Q", "1").strip() == "1"
)


def _is_object_count_question(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    return (_OBJ_COUNT_RE.search(q) is not None) and (_OBJ_HINT_RE.search(q) is not None)


@app.get("/health")
def health():
    return {"status": "ok", "service": "agent"}


@app.post("/answer")
def answer(req: AnswerRequest):
    obj_list = req.object_list or []

    # ---- Ephemeral object summary ----
    by_layer: dict[str, int] = {}
    by_type: dict[str, int] = {}

    for obj in obj_list:
        if not isinstance(obj, dict):
            continue
        layer = obj.get("layer", "UNKNOWN")
        by_layer[layer] = by_layer.get(layer, 0) + 1

        t = obj.get("type", "UNKNOWN")
        by_type[t] = by_type.get(t, 0) + 1

    object_summary = {
        "total_objects": len(obj_list),
        "by_layer": by_layer,
        "by_type": by_type,
    }

    q = (req.question or "").strip()

    # ---- Retrieval decision ----
    # If this is a pure object counting question, skip retrieval to avoid irrelevant sources.
    skip_retrieval = _DISABLE_RETRIEVAL_FOR_OBJECT_Q and _is_object_count_question(q)

    hits = []
    effective_top_k = req.top_k if req.top_k is not None else 2
    if not skip_retrieval:
        if _PAGE_RE.search(q):
            effective_top_k = max(int(effective_top_k), 5)
        hits = retrieve(req.question, top_k=effective_top_k)

    # ---- Sources ----
    sources = []
    if not skip_retrieval:
        for h in hits:
            md = h.get("metadata") or {}
            source_item = {
                "chunk_id": h.get("chunk_id"),
                "score": h.get("score"),
                "filename": md.get("filename"),
                "section": md.get("section"),
                "page": md.get("page"),
                "kind": md.get("kind", "text"),
            }

            if req.evidence_mode:
                txt = (h.get("text") or "").replace("\n", " ").strip()
                source_item["excerpt"] = " ".join(txt.split())[:500]

            sources.append(source_item)

    # ---- Quote mode (LangGraph) ----
    if req.quote_mode:
        result = (
            GRAPH.invoke(
                {
                    "question": req.question,
                    "object_summary": object_summary,
                    "object_list": obj_list,
                    "hits": hits,
                    "retry_count": 0,
                    "max_retries": 1,
                }
            )
            or {}
        )

        plan = result.get("plan") or {}
        strategy = (plan.get("strategy") or "").strip()

        used_hits = result.get("used_hits") or hits

        # ✅ direct_* stratejilerde evidence zorunlu değil
        if not strategy.startswith("direct_"):
            try:
                result = validate_evidence(result, used_hits)
            except Exception:
                result = {
                    "answer": "I don't have enough information in the provided excerpts.",
                    "evidence": [],
                }

            if not (result.get("evidence") or []):
                result["answer"] = "I don't have enough information in the provided excerpts."
                result["evidence"] = []
        else:
            result["evidence"] = result.get("evidence") or []

        return {
            "answer": result.get(
                "answer", "I don't have enough information in the provided excerpts."
            ),
            "evidence": result.get("evidence", []),
            "object_checks": result.get("object_checks", []),
            "object_summary": object_summary,
            "sources": sources,  # will be [] for pure object count qs
            "plan": plan,
            "meta": {**_meta(), "format": "langgraph_json_with_quotes"},
        }

    # ---- Non-quote mode ----
    if skip_retrieval:
        # deterministic object answer via graph (no LLM)
        result = (
            GRAPH.invoke(
                {
                    "question": req.question,
                    "object_summary": object_summary,
                    "object_list": obj_list,
                    "hits": [],
                    "retry_count": 0,
                    "max_retries": 0,
                }
            )
            or {}
        )
        return {
            "answer": result.get(
                "answer", "I don't have enough information in the provided excerpts."
            ),
            "evidence": result.get("evidence", []),
            "object_summary": object_summary,
            "sources": [],
            "meta": {**_meta(), "format": "plain_text"},
        }

    # ---- LLM-only fallback (non-quote mode) ----
    prompt = build_prompt(req.question, object_summary, hits)
    raw = generate(prompt)

    candidate = _extract_json_block(raw)
    try:
        parsed = json.loads(candidate)
        parsed = validate_evidence(parsed, hits)
        final_answer = parsed.get("answer") or raw
        evidence = parsed.get("evidence") or []
    except Exception:
        final_answer = raw
        evidence = []

    return {
        "answer": final_answer,
        "evidence": evidence,
        "object_summary": object_summary,
        "sources": sources,
        "meta": {**_meta(), "format": "plain_text"},
    }
