# Hybrid RAG (Persistent PDF + Ephemeral Session Objects)

This repository implements a **hybrid Retrieval-Augmented Generation (RAG)** system that answers questions using:

* **Persistent knowledge**: a pre-embedded PDF stored in **ChromaDB** (`./chroma_db`, committed in the repo)
* **Ephemeral knowledge**: a **session-specific JSON object list** (stored in backend session memory; **NOT** embedded)

Dockerized architecture:

* **frontend** (React + Nginx): UI → `http://localhost:8080`
* **backend** (FastAPI): auth + session + routing → `http://localhost:8000`
* **agent** (FastAPI + LangGraph): hybrid RAG reasoning → `http://localhost:8001`
* **ollama**: local LLM/VLM inside Docker network (**not exposed to host**)

---

## Prerequisites

* Docker + Docker Compose (v2)
* (Optional) `jq` for pretty-printing JSON in curl examples

Ports used on host:

* `8080` (frontend), `8000` (backend), `8001` (agent)

If any of these ports are already in use, stop the conflicting services or change the mappings in `docker-compose.yml`.

---

## Quickstart (Reviewer) — one command

From the repo root:

```bash
make up
```

This will build + start all services and (optionally) prompt to download Ollama models.

Then:

```bash
make health
make open
```

Optional end-to-end check:

```bash
make test
```

Stop services (keeps downloaded models):

```bash
make down
```

⚠️ Full reset (deletes Ollama models volume!):

```bash
make reset
```

---

## Quickstart (Reviewer) — raw Docker commands (no Makefile)

### 1) Start services

```bash
docker compose up -d --build ollama agent backend frontend
docker compose ps
```

Health checks:

```bash
curl -s http://localhost:8001/health && echo
curl -s http://localhost:8000/health && echo
```

Expected:

* agent: `{"status":"ok","service":"agent"}`
* backend: `{"status":"ok","service":"backend"}`

### 2) Download models (first run)

**Required for answering document questions (LLM).** Models are stored in a Docker volume (`ollama_data`).
If you run `docker compose down -v` (or `make reset`), models will be removed and must be re-downloaded.

**Minimal (recommended for reviewers):**

```bash
docker compose exec -T ollama ollama pull llama3.1:8b
```

**Optional (only needed if you re-run ingestion with VLM captions):**

```bash
docker compose exec -T ollama ollama pull llava:13b
```

Check:

```bash
docker compose exec -T ollama ollama list
```

Notes:

* Download size is large (multi-GB). CPU inference is supported but slower.
* `ollama` is intentionally not published to host (no `11434:11434` mapping). Containers reach it via `http://ollama:11434`.

---

## Demo via UI (recommended)

Open:

* `http://localhost:8080`

### 1) Register / Login

Example credentials:

* email: `demo@example.com`
* password: `demo123`

### 2) Paste sample objects

In the UI “Object List” textarea paste (same content as `./sample_objects.json`):

```json
{
  "object_list": [
    {"type":"LINE","layer":"Highway","start":[0,0],"end":[10,0]},
    {"type":"POLYLINE","layer":"Windows","points":[[1,1],[2,1]],"closed":false},
    {"type":"POLYLINE","layer":"Windows","points":[[3,1],[4,1]],"closed":false}
  ]
}
```

### 3) Ask these questions (copy/paste)

**A) Object-only (ephemeral)**

1. `How many objects are in my current session object_list? Reply with only the number.`
   Expected: `3`

Then update object list to ONLY Highway:

```json
{ "object_list": [ {"type":"LINE","layer":"Highway","start":[0,0],"end":[10,0]} ] }
```

Ask again → expected: `1`

**B) Doc-only (persistent RAG)**

2. `On page 14, what is the restriction about the principal elevation and highway? 1-2 sentences.`
   Expected: answer supported by a quote from page 14.

**C) Hybrid (doc + objects together)**

3. `Using the page 14 rule AND my session objects: do I have any object on the "Highway" layer? Answer YES/NO and include a short quote from page 14 as evidence.`

* With Highway present → expected: `YES` (+ quote from page 14)
* Remove Highway from object list (leave only Windows) → expected: `NO` (+ quote from page 14)

**D) Visual question (bonus: captions used in retrieval)**

4. `What does the diagram on page 38 show?`
   Expected: response shows at least one source with `kind=caption`.

> Note: captions are already in the committed `./chroma_db`, so this works out-of-the-box.
> The **llava** model is only needed if you want to **re-run ingestion** with captions.

---

## Demo via API (curl)

### 1) Register + Login

```bash
curl -sS -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"demo123"}' | jq .

TOKEN=$(curl -sS -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"demo123"}' \
| jq -r '.access_token')

echo "$TOKEN" | head -c 20; echo
```

### 2) Update session objects (PUT)

You can use the provided file `./sample_objects.json` directly:

```bash
curl -sS -X PUT "http://localhost:8000/session/objects" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @sample_objects.json | jq .
```

Verify:

```bash
curl -sS -X GET "http://localhost:8000/session/objects" \
  -H "Authorization: Bearer $TOKEN" | jq .
```

### 3) Ask questions (hybrid endpoint)

```bash
curl -sS -X POST "http://localhost:8000/qa" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"On page 14, what is the restriction about the principal elevation and highway? 1-2 sentences.","top_k":5,"quote_mode":true}' | jq .
```

Object count (should reflect latest session state):

```bash
curl -sS -X POST "http://localhost:8000/qa" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"How many objects are in my current session object_list? Reply with only the number.","top_k":3}' | jq -r '.answer'
```

Agent direct (doc-only, no backend/session):

```bash
curl -sS -X POST "http://localhost:8001/answer" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the diagram on page 38 show?","top_k":10,"quote_mode":true}' | jq .
```

---

## Data & Embeddings

### Prebuilt DB (default)

This repo includes a prebuilt Chroma DB at `./chroma_db`, so **ingestion is NOT required** for the demo.

### Optional: re-ingest documents (advanced)

Recreate embeddings (uses `EMBED_MODEL`, defaults in `.env.example`):

```bash
docker compose run --rm \
  -e RESET_COLLECTION=1 \
  ingest
```

Enable VLM captions during ingestion (slower; requires `llava:13b`):

```bash
docker compose run --rm \
  -e RESET_COLLECTION=1 \
  -e ENABLE_VLM_CAPTIONS=1 \
  ingest
```

---

## Troubleshooting

### Ollama models missing (`ollama list` shows empty)

Pull models again:

```bash
docker compose exec -T ollama ollama pull llama3.1:8b
docker compose exec -T ollama ollama list
```

(Optional for ingestion captions)

```bash
docker compose exec -T ollama ollama pull llava:13b
```

### Agent can’t reach Ollama

Check from inside agent:

```bash
docker compose exec -T agent python -c "import urllib.request; print(urllib.request.urlopen('http://ollama:11434/api/tags',timeout=10).read()[:200])"
```

Expected: JSON with `"models":[...]`.

### “Works for me, not for reviewer”

Most common causes:

* Ports 8080/8000/8001 already in use on reviewer machine
* Docker not installed / old compose version
* Not enough disk space for models
* Corporate proxy blocking large downloads

### Clean restart (keeps downloaded models)

```bash
docker compose down
docker compose up -d --build ollama agent backend frontend
```

### Full reset (WARNING: deletes models)

```bash
docker compose down -v
```

---


## Roadmap

* Makefile-based “one-command” reviewer flow (already included): `make up`, `make test`, `make ingest(-vlm)`
* (Optional) improved Makefile prompts for large model downloads / time+disk warnings
