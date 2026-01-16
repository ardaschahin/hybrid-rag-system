from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

from agent.app.rag import retrieve

app = FastAPI(title="AICI Agent Service", version="0.2.0")

class AnswerRequest(BaseModel):
    question: str
    object_list: list[dict] | None = None
    top_k: int | None = 5

@app.get("/health")
def health():
    return {"status": "ok", "service": "agent"}

@app.post("/answer")
def answer(req: AnswerRequest):
    obj_list = req.object_list or []

    # Ephemeral summary (şimdilik basit)
    by_layer: dict[str, int] = {}
    for obj in obj_list:
        layer = obj.get("layer", "UNKNOWN")
        by_layer[layer] = by_layer.get(layer, 0) + 1

    # Persistent retrieval
    hits = retrieve(req.question, top_k=req.top_k or 5)

    sources = []
    for h in hits:
        md = h["metadata"]
        sources.append({
            "chunk_id": h["chunk_id"],
            "score": h["score"],
            "filename": md.get("filename"),
            "page": md.get("page"),
        })

    # Şimdilik LLM yok: sadece retrieval kanıtı + state kanıtı
    return {
        "answer": f"Echo: {req.question} | objects={len(obj_list)} | retrieved={len(hits)}",
        "object_summary": {"total_objects": len(obj_list), "by_layer": by_layer},
        "sources": sources
    }
