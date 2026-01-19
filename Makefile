SHELL := /bin/sh

COMPOSE := docker compose

# Host URLs
FRONTEND_URL := http://localhost:8080
BACKEND_URL  := http://localhost:8000
AGENT_URL    := http://localhost:8001

# Ollama models (change if needed)
LLM_MODEL := llama3.1:8b
VLM_MODEL := llava:13b

# Demo creds for quick API test
DEMO_EMAIL := demo@example.com
DEMO_PASS  := demo123

.PHONY: help up down reset rebuild logs ps health open pull pull-force \
        test test-agent test-backend test-auth test-objects test-qa \
        ingest ingest-vlm

help:
	@echo ""
	@echo "Targets:"
	@echo "  make up           - Build + start (asks to download Ollama models)"
	@echo "  make down         - Stop containers (keeps volumes/models)"
	@echo "  make reset        - Stop + remove volumes (DELETES Ollama models)"
	@echo "  make rebuild      - Rebuild images (no cache) + start"
	@echo "  make ps           - Show running containers"
	@echo "  make logs         - Tail logs"
	@echo "  make health       - Health checks (agent/backend)"
	@echo "  make open         - Print UI/API URLs"
	@echo "  make pull         - Interactive: download Ollama models"
	@echo "  make pull-force   - Download models without prompting"
	@echo "  make test         - Run a minimal end-to-end API test flow"
	@echo "  make ingest       - Optional: re-ingest embeddings (RESET_COLLECTION=1)"
	@echo "  make ingest-vlm   - Optional: re-ingest with VLM captions (slow)"
	@echo ""

up:
	@$(COMPOSE) up -d --build ollama agent backend frontend
	@$(MAKE) health
	@$(MAKE) pull

down:
	@$(COMPOSE) down

reset:
	@echo "WARNING: This will remove volumes and delete downloaded Ollama models."
	@printf "Continue? [y/N] " ; read ans ; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	  $(COMPOSE) down -v ; \
	else \
	  echo "Cancelled." ; \
	fi

rebuild:
	@$(COMPOSE) build --no-cache
	@$(COMPOSE) up -d ollama agent backend frontend
	@$(MAKE) health

logs:
	@$(COMPOSE) logs -f --tail=200

ps:
	@$(COMPOSE) ps

health:
	@printf "Agent health:   " ; curl -fsS $(AGENT_URL)/health   || true ; echo ""
	@printf "Backend health: " ; curl -fsS $(BACKEND_URL)/health || true ; echo ""

open:
	@echo "Frontend: $(FRONTEND_URL)"
	@echo "Backend:  $(BACKEND_URL)/docs"
	@echo "Agent:    $(AGENT_URL)/docs"

# --- Ollama models ---
pull:
	@printf "Download Ollama models now? (LLM=$(LLM_MODEL), VLM=$(VLM_MODEL)) [y/N] " ; read ans ; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	  $(MAKE) pull-force ; \
	else \
	  echo "Skipping model download. If needed later: make pull-force" ; \
	fi

pull-force:
	@echo "Pulling $(LLM_MODEL) ..."
	@$(COMPOSE) exec -T ollama ollama pull $(LLM_MODEL)
	@echo "Pulling $(VLM_MODEL) ..."
	@$(COMPOSE) exec -T ollama ollama pull $(VLM_MODEL)
	@echo "Installed models:"
	@$(COMPOSE) exec -T ollama ollama list || true

# --- Optional ingestion ---
ingest:
	@$(COMPOSE) run --rm -e RESET_COLLECTION=1 ingest

ingest-vlm:
	@$(COMPOSE) run --rm -e RESET_COLLECTION=1 -e ENABLE_VLM_CAPTIONS=1 ingest

# --- Minimal end-to-end test flow (API) ---
test: test-backend test-agent test-auth test-objects test-qa
	@echo ""
	@echo "OK: end-to-end API test finished."

test-agent:
	@curl -fsS $(AGENT_URL)/health >/dev/null
	@echo "OK: agent health"

test-backend:
	@curl -fsS $(BACKEND_URL)/health >/dev/null
	@echo "OK: backend health"

test-auth:
	@echo "Register/login test user (safe if already exists)..."
	@curl -fsS -X POST "$(BACKEND_URL)/auth/register" \
	  -H "Content-Type: application/json" \
	  -d '{"email":"$(DEMO_EMAIL)","password":"$(DEMO_PASS)"}' >/dev/null || true
	@TOKEN=$$(curl -fsS -X POST "$(BACKEND_URL)/auth/login" \
	  -H "Content-Type: application/json" \
	  -d '{"email":"$(DEMO_EMAIL)","password":"$(DEMO_PASS)"}' | \
	  python -c "import sys,json; print(json.load(sys.stdin)['access_token'])"); \
	echo "$$TOKEN" | head -c 20 >/dev/null ; \
	echo "OK: auth"

test-objects:
	@echo "Update objects for demo user..."
	@TOKEN=$$(curl -fsS -X POST "$(BACKEND_URL)/auth/login" \
	  -H "Content-Type: application/json" \
	  -d '{"email":"$(DEMO_EMAIL)","password":"$(DEMO_PASS)"}' | \
	  python -c "import sys,json; print(json.load(sys.stdin)['access_token'])"); \
	cat > /tmp/sample_objects.json <<'JSON' ; \
{ \
  "object_list": [ \
    {"type":"LINE","layer":"Highway","start":[0,0],"end":[10,0]}, \
    {"type":"POLYLINE","layer":"Windows","points":[[1,1],[2,1]],"closed":false}, \
    {"type":"POLYLINE","layer":"Windows","points":[[3,1],[4,1]],"closed":false} \
  ] \
} \
JSON \
	curl -fsS -X PUT "$(BACKEND_URL)/session/objects" \
	  -H "Authorization: Bearer $$TOKEN" \
	  -H "Content-Type: application/json" \
	  -d @/tmp/sample_objects.json >/dev/null ; \
	echo "OK: objects updated"

test-qa:
	@echo "Ask a doc question (page 14)..."
	@TOKEN=$$(curl -fsS -X POST "$(BACKEND_URL)/auth/login" \
	  -H "Content-Type: application/json" \
	  -d '{"email":"$(DEMO_EMAIL)","password":"$(DEMO_PASS)"}' | \
	  python -c "import sys,json; print(json.load(sys.stdin)['access_token'])"); \
	curl -fsS -X POST "$(BACKEND_URL)/qa" \
	  -H "Authorization: Bearer $$TOKEN" \
	  -H "Content-Type: application/json" \
	  -d '{"question":"On page 14, what is the restriction about the principal elevation and highway? 1-2 sentences.","top_k":5,"quote_mode":true}' \
	  >/dev/null ; \
	echo "OK: QA call"
