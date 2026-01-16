import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from fastembed import TextEmbedding

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docs")

# Embedder'ı tek kez oluşturmak performans için iyi.
_EMBEDDER = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Returns list of dicts:
    {
      "chunk_id": str,
      "text": str,
      "metadata": {...},
      "score": float
    }
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    q_vec = list(_EMBEDDER.embed([query]))[0]

    res = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i in range(len(ids)):
        # Chroma distance: smaller = more similar
        dist = dists[i] if i < len(dists) else None
        score = None if dist is None else float(1.0 / (1.0 + dist))
        hits.append({
            "chunk_id": ids[i],
            "text": docs[i],
            "metadata": metas[i] or {},
            "score": score,
        })
    return hits
