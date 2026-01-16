from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

app = FastAPI(title="AICI Agent Service", version="0.1.0")

class AnswerRequest(BaseModel):
    question: str
    object_list: list[dict] | None = None

@app.get("/health")
def health():
    return {"status": "ok", "service": "agent"}

@app.post("/answer")
def answer(req: AnswerRequest):
    obj_list = req.object_list or []
    by_layer: dict[str, int] = {}
    for obj in obj_list:
        layer = obj.get("layer", "UNKNOWN")
        by_layer[layer] = by_layer.get(layer, 0) + 1

    return {
        "answer": f"Echo: {req.question} | objects={len(obj_list)}",
        "object_summary": {"total_objects": len(obj_list), "by_layer": by_layer},
        "sources": []
    }
