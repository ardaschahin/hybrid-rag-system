## Quickstart 

This repo ships with a **pre-built Chroma DB** in `./chroma_db` (includes **VLM captions**).
So **no ingestion is required** to run the demo.

### Prerequisites

* Docker + Docker Compose (v2)
* (Optional) `jq` for pretty-printing JSON

### Services

* `frontend` UI: [http://localhost:8080](http://localhost:8080)
* `backend` API: [http://localhost:8000](http://localhost:8000)
* `agent` (RAG + LLM): [http://localhost:8001](http://localhost:8001)
* `ollama` (LLM/VLM server) runs **inside the Docker network only** (not exposed on host)

---

## 1) Start the stack

```bash
docker compose up -d --build ollama agent backend frontend
docker compose ps
```

---

## 2) Download models (first run only)

Models are stored in a Docker volume (`ollama_data`).
This step is required once (unless volumes are deleted).

```bash
docker compose exec -T ollama ollama pull llama3.1:8b
docker compose exec -T ollama ollama pull llava:13b
docker compose exec -T ollama ollama list
```

---

## 3) Health checks

```bash
curl -s http://localhost:8001/health && echo
curl -s http://localhost:8000/health && echo
```

Expected:

* agent: `{"status":"ok","service":"agent"}`
* backend: `{"status":"ok","service":"backend"}`

---

## 4) Run sample queries

### Agent direct

```bash
curl -sS -X POST "http://localhost:8001/answer" \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize what the document is about.", "top_k": 3, "quote_mode": true}' | jq .
```

```bash
curl -sS -X POST "http://localhost:8001/answer" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the diagram on page 38 show?", "top_k": 20, "quote_mode": true}' | jq .
```

---

## 5) Open the UI

Frontend: [http://localhost:8080](http://localhost:8080)

---

## Ingestion (optional / reproducibility)

The demo uses the prebuilt `./chroma_db`.
If you want to fully regenerate the DB:

1. Start ollama and pull models:

```bash
docker compose up -d ollama
docker compose exec -T ollama ollama pull llama3.1:8b
docker compose exec -T ollama ollama pull llava:13b
```

2. Rebuild DB with VLM captions:

```bash
docker compose run --rm \
  -e RESET_COLLECTION=1 \
  -e ENABLE_VLM_CAPTIONS=1 \
  ingest
```

---

## Troubleshooting

### "model not found" or empty `ollama list`

```bash
docker compose exec -T ollama ollama pull llama3.1:8b
docker compose exec -T ollama ollama pull llava:13b
```

### Agent cannot reach Ollama

Check agent env:

```bash
docker compose exec -T agent sh -lc 'echo $OLLAMA_BASE_URL'
```

Expected: `http://ollama:11434`

Connectivity check from agent:

```bash
docker compose exec -T agent python -c "import urllib.request; print(urllib.request.urlopen('http://ollama:11434/api/tags',timeout=10).read()[:200])"
```

### Clean restart (keep models)

```bash
docker compose down
docker compose up -d --build ollama agent backend frontend
```

### Full reset (WARNING: deletes downloaded models + DB)

```bash
docker compose down -v
```

---

## Security notes

* `ollama` is **not published to the host** (no `11434:11434` port mapping), so it is only reachable by other containers on the Docker network.
