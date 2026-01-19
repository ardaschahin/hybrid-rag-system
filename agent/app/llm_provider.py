import os
import httpx

class LLMError(RuntimeError):
    pass

_CLIENT: httpx.Client | None = None

def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        timeout = httpx.Timeout(
            connect=float(os.getenv("LLM_CONNECT_TIMEOUT", "5")),
            read=float(os.getenv("LLM_READ_TIMEOUT", "120")),
            write=float(os.getenv("LLM_WRITE_TIMEOUT", "30")),
            pool=float(os.getenv("LLM_POOL_TIMEOUT", "30")),
        )
        limits = httpx.Limits(
            max_keepalive_connections=int(os.getenv("LLM_MAX_KEEPALIVE", "5")),
            max_connections=int(os.getenv("LLM_MAX_CONNECTIONS", "10")),
            keepalive_expiry=float(os.getenv("LLM_KEEPALIVE_EXPIRY", "30")),
        )
        _CLIENT = httpx.Client(timeout=timeout, limits=limits, http2=False)
    return _CLIENT

def generate(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    if provider == "ollama":
        return _generate_ollama(prompt)
    if provider == "openai":
        return _generate_openai(prompt)
    raise LLMError(f"Unsupported LLM_PROVIDER={provider}")

def _generate_ollama(prompt: str) -> str:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    url = f"{base}/api/generate"

    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "180"))
    top_p = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    fmt = os.getenv("OLLAMA_FORMAT", "json").strip() or "json"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": fmt,
        "options": {"temperature": temperature, "num_predict": num_predict, "top_p": top_p},
    }

    try:
        r = _client().post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise LLMError(f"Ollama request failed: {e}") from e

    return (data.get("response") or "").strip()

def _generate_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set")

    # Chat Completions style endpoint (widely compatible)
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # IMPORTANT: sen JSON parse ettiğin için modelin JSON dönmesini istiyoruz
    # build_prompt zaten JSON schema istiyor; burada response_format ekleyebiliriz.
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "300"))

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "Return ONLY valid JSON as instructed."},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        r = _client().post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        raise LLMError(f"OpenAI request failed: {e}") from e
