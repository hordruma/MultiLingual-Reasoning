"""
Provider Adapters
=================
Unified async interface for calling every model in config.MODELS.

Three provider kinds:
  anthropic      – native Anthropic Messages API
  openai_compat  – any OpenAI-style POST {base_url}/chat/completions endpoint
                   (OpenAI, Gemini's OpenAI-compatible layer, DeepSeek,
                   Mistral, DashScope/Qwen, OpenRouter, Azure AI Foundry
                   serverless, vLLM, Ollama, ...)
  mock           – offline test double, returns a canned answer

Credentials and endpoints are read from environment variables named in the
model config (`api_key_env`, `base_url_env`).  `.env` is loaded by
run_experiment.py via python-dotenv.
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class LLMResponse:
    """Standardised response from any provider."""
    model_id: str
    content: str                # full text response (CoT + answer)
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str = ""     # "stop", "length"/"max_tokens", ...
    truncated: bool = False     # True when the output hit max_tokens
    raw: Optional[dict] = None  # provider-specific payload for debugging


class ProviderError(Exception):
    """Raised when an API call fails after retries (or is not retryable)."""


class ConfigError(Exception):
    """Raised when a model is missing its key / endpoint configuration."""


# ── Retry policy ─────────────────────────────────────────────────────────

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 5, 15, 30]          # seconds, plus jitter
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
TIMEOUT_SECONDS = 180

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """One shared connection pool for the whole run."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
    return _client


async def close_client():
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    return False


def _retry_after(exc: Exception) -> Optional[float]:
    if isinstance(exc, httpx.HTTPStatusError):
        ra = exc.response.headers.get("retry-after")
        if ra:
            try:
                return min(float(ra), 120.0)
            except ValueError:
                return None
    return None


async def _retry(coro_fn, *args, **kwargs):
    """Retry on transient failures only. 4xx auth/validation errors fail fast."""
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 – we classify below
            last_err = e
            if not _is_retryable(e):
                detail = _describe(e)
                raise ProviderError(f"Non-retryable error: {detail}") from e
            if attempt < MAX_RETRIES - 1:
                wait = _retry_after(e) or (RETRY_BACKOFF[attempt] + random.uniform(0, 1))
                print(f"  ⚠ Retry {attempt + 1}/{MAX_RETRIES} after {wait:.0f}s: {_describe(e)}")
                await asyncio.sleep(wait)
    raise ProviderError(f"Failed after {MAX_RETRIES} attempts: {_describe(last_err)}") from last_err


def _describe(exc: Optional[Exception]) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:300].replace("\n", " ")
        return f"HTTP {exc.response.status_code} from {exc.request.url.host}: {body}"
    return f"{type(exc).__name__}: {exc}"


# ── Config resolution ───────────────────────────────────────────────────

def resolve_model(model_cfg: dict) -> dict:
    """
    Turn a config.MODELS entry into concrete call parameters, reading env
    vars.  Raises ConfigError with an actionable message if something is
    missing, so a long run never starts with a broken model.
    """
    provider = model_cfg["provider"]
    resolved = {"provider": provider, "model_id": model_cfg["model_id"]}

    if provider == "mock":
        return resolved

    key_env = model_cfg.get("api_key_env")
    api_key = os.environ.get(key_env, "") if key_env else ""
    if not api_key or api_key.startswith(("sk-ant-...", "sk-...", "AIza...", "...")):
        raise ConfigError(f"{key_env} is not set (needed for {model_cfg['display']})")
    resolved["api_key"] = api_key

    if provider == "openai_compat":
        base_env = model_cfg.get("base_url_env")
        base_url = (os.environ.get(base_env) if base_env else None) or model_cfg.get("base_url")
        if not base_url:
            raise ConfigError(f"No base URL for {model_cfg['display']} (set {base_env})")
        resolved["base_url"] = base_url.rstrip("/")
        extra_headers_env = model_cfg.get("extra_headers_env")
        resolved["extra_headers"] = {}
        if extra_headers_env and os.environ.get(extra_headers_env):
            # format: "Header-A: value;Header-B: value"
            for pair in os.environ[extra_headers_env].split(";"):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    resolved["extra_headers"][k.strip()] = v.strip()

    model_id_env = model_cfg.get("model_id_env")
    if model_id_env and os.environ.get(model_id_env):
        resolved["model_id"] = os.environ[model_id_env]

    return resolved


# ── Anthropic (native Messages API) ─────────────────────────────────────

async def _call_anthropic(resolved: dict, system: str, user: str,
                          max_tokens: int, temperature: float) -> LLMResponse:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": resolved["api_key"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": resolved["model_id"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    t0 = time.monotonic()
    r = await _get_client().post(url, headers=headers, json=body)
    r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    stop = data.get("stop_reason", "") or ""
    usage = data.get("usage", {})
    return LLMResponse(
        model_id=resolved["model_id"], content=text,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        latency_ms=elapsed, finish_reason=stop,
        truncated=(stop == "max_tokens"), raw=data,
    )


# ── OpenAI-compatible chat completions ──────────────────────────────────

async def _call_openai_compat(resolved: dict, system: str, user: str,
                              max_tokens: int, temperature: float) -> LLMResponse:
    url = f"{resolved['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {resolved['api_key']}",
        "content-type": "application/json",
        **resolved.get("extra_headers", {}),
    }
    body = {
        "model": resolved["model_id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.monotonic()
    r = await _get_client().post(url, headers=headers, json=body)
    r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    if "choices" not in data or not data["choices"]:
        raise ProviderError(f"Malformed response (no choices): {str(data)[:300]}")
    choice = data["choices"][0]
    message = choice.get("message", {}) or {}
    text = message.get("content") or ""
    # Some reasoning models put the chain in a separate field; keep it so the
    # language analysis can see it.
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning and reasoning not in text:
        text = f"{reasoning}\n\n{text}"
    finish = choice.get("finish_reason", "") or ""
    usage = data.get("usage", {}) or {}
    return LLMResponse(
        model_id=resolved["model_id"], content=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=elapsed, finish_reason=finish,
        truncated=(finish == "length"), raw=data,
    )


# ── Mock provider (offline pipeline test) ───────────────────────────────

async def _call_mock(resolved: dict, system: str, user: str,
                     max_tokens: int, temperature: float) -> LLMResponse:
    """
    Returns a fake answer so the pipeline can be exercised without a network.
    It echoes whichever label appears first in the user prompt's label list,
    which produces *known-meaningless* accuracy.  Results from this provider
    must never be presented as findings.
    """
    await asyncio.sleep(0.001)
    label = "Yes"
    marker = "Answer with exactly one of:"
    if marker in user:
        tail = user.split(marker, 1)[1].strip()
        first = tail.split("\n", 1)[0].split(",")[0].strip()
        if first:
            label = first
    text = f"[mock reasoning – no real model was called]\nANSWER: {label}"
    return LLMResponse(
        model_id="mock", content=text, input_tokens=len(user) // 4,
        output_tokens=12, latency_ms=1.0, finish_reason="stop",
    )


# ── Dispatch ────────────────────────────────────────────────────────────

PROVIDER_MAP = {
    "anthropic":     _call_anthropic,
    "openai_compat": _call_openai_compat,
    "mock":          _call_mock,
}


async def call_model(resolved: dict, system: str, user: str,
                     max_tokens: int = 2048, temperature: float = 0.0) -> LLMResponse:
    """Unified entry point – dispatches to the right provider with retries."""
    fn = PROVIDER_MAP.get(resolved["provider"])
    if fn is None:
        raise ValueError(f"Unknown provider: {resolved['provider']}")
    return await _retry(fn, resolved, system, user, max_tokens, temperature)


async def smoke_test(resolved: dict) -> LLMResponse:
    """Cheapest possible call to confirm the key, endpoint and model id work."""
    return await call_model(
        resolved,
        system="Reply with the single word OK.",
        user="Say OK.",
        max_tokens=8,
        temperature=0.0,
    )
