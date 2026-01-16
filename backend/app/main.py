from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os
from typing import Any

app = FastAPI(title="AICI Backend API", version="0.1.0")

AGENT_URL = os.getenv("AGENT_URL", "http://127.0.0.1:8001")

# MVP session store (in-memory): user yok şimdilik, tek global session
SESSION: dict[str, Any] = {
    "object_list": None
}

class ObjectsUpdateRequest(BaseModel):
    object_list: list[dict]

class QARequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "backend"}

@app.put("/session/objects")
def update_objects(req: ObjectsUpdateRequest):
    # Minimum validation
    if not isinstance(req.object_list, list):
        raise HTTPException(status_code=400, detail="object_list must be a list")
    SESSION["object_list"] = req.object_list
    return {"message": "objects_updated", "object_count": len(req.object_list)}

@app.post("/qa")
async def qa(req: QARequest):
    object_list = SESSION.get("object_list")
    if object_list is None:
        raise HTTPException(status_code=400, detail="No object_list set. Call PUT /session/objects first.")

    payload = {"question": req.question, "object_list": object_list}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{AGENT_URL}/answer", json=payload)
        r.raise_for_status()
        return r.json()
