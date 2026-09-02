"""
Provider Adapters
=================
Unified async interface for calling every model in config.MODELS.

Three provider kinds:
  anthropic      – native Anthropic Messages API
  openai_compat  – any OpenAI-style POST {base_url}/chat/completions endpoint
                   (OpenAI, Gemini's OpenAI-compatible layer, DeepSeek, Z.ai,
                   DashScope/Qwen, MiniMax, Mistral, OpenRouter, Ollama,
                   LM Studio, vLLM, Azure AI Foundry, ...)
  mock           – offline test double, returns a canned answer

Credentials and endpoints are read from environment variables named in the
model config (`api_key_env`, `base_url_env`).  `.env` is loaded by
run_experiment.py via python-dotenv.

Hidden reasoning: thinking-capable models are asked to switch thinking off
through `request_overrides` in the model config.  Whatever hidden reasoning
still comes back (a `reasoning_content` / `reasoning` field, or an inline
<think>...</think> block) is separated from the visible answer and returned
in `LLMResponse.reasoning`, never merged into `content`.
"""

import asyncio
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx


@dataclass
class LLMResponse:
    """Standardised response from any provider."""
    model_id: str
    content: str                # visible text (CoT + answer)
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str = ""     # "stop", "length"/"max_tokens", ...
    truncated: bool = False     # True when the output hit the token cap
    reasoning: str = ""         # hidden reasoning returned by the provider, if any
    raw: Optional[dict] = None  # provider-specific payload for debugging


class ProviderError(Exception):
    """Raised when an API call fails after retries (or is not retryable)."""


class ConfigError(Exception):
    """Raised when a model is missing its key / endpoint configuration."""


# ── Retry policy ─────────────────────────────────────────────────────────

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 5, 15, 30]          # seconds, plus jitter
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
TIMEOUT_SECONDS = 300                   # local models can be slow

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
                raise ProviderError(f"Non-retryable error: {_describe(e)}") from e
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

_PLACEHOLDER_KEYS = ("sk-ant-...", "sk-...", "AIza...", "...")


def resolve_model(model_cfg: dict) -> dict:
    """
    Turn a config.MODELS entry into concrete call parameters, reading env
    vars.  Raises ConfigError with an actionable message if something is
    missing, so a long run never starts with a broken model.
    """
    provider = model_cfg["provider"]
    resolved = {
        "provider": provider,
        "model_id": model_cfg["model_id"],
        "request_overrides": dict(model_cfg.get("request_overrides") or {}),
        "max_tokens_param": model_cfg.get("max_tokens_param", "max_tokens"),
        # "temperature" key present with None means "omit the field"
        "temperature": model_cfg.get("temperature", "default"),
        "hidden_reasoning": model_cfg.get("hidden_reasoning", "unknown"),
    }

    if provider == "mock":
        return resolved

    key_env = model_cfg.get("api_key_env")
    api_key = os.environ.get(key_env, "") if key_env else ""
    if not api_key or api_key.startswith(_PLACEHOLDER_KEYS):
        api_key = model_cfg.get("api_key_default", "")
    if not api_key:
        raise ConfigError(f"{key_env} is not set (needed for {model_cfg['display']})")
    resolved["api_key"] = api_key

    if provider == "openai_compat":
        base_env = model_cfg.get("base_url_env")
        base_url = (os.environ.get(base_env) if base_env else None) or model_cfg.get("base_url")
        if not base_url:
            raise ConfigError(f"No base URL for {model_cfg['display']} (set {base_env})")
        resolved["base_url"] = base_url.rstrip("/")

    model_id_env = model_cfg.get("model_id_env")
    if model_id_env and os.environ.get(model_id_env):
        resolved["model_id"] = os.environ[model_id_env]

    return resolved


# ── Reasoning separation ────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_reasoning(content: str, reasoning: str = "") -> Tuple[str, str]:
    """
    Return (visible_content, hidden_reasoning).  Inline <think> blocks are
    moved out of the content.  If the content is empty but a reasoning field
    is present (Ollama + Gemma 4 on the /v1 endpoint do this), the reasoning
    is promoted to content because it is the only text the model produced.
    """
    content = content or ""
    reasoning = reasoning or ""
    blocks = _THINK_RE.findall(content)
    if blocks:
        reasoning = "\n".join([reasoning] + [b.strip() for b in blocks]).strip()
        content = _THINK_RE.sub("", content).strip()
    if not content.strip() and reasoning.strip():
        return reasoning.strip(), ""
    return content, reasoning


# ── OpenAI-compatible chat completions ──────────────────────────────────

def build_openai_body(resolved: dict, system: str, user: str,
                      max_tokens: int, temperature: float) -> dict:
    """Pure function so the request shape can be unit-tested."""
    body = {
        "model": resolved["model_id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        resolved.get("max_tokens_param", "max_tokens"): max_tokens,
    }
    model_temp = resolved.get("temperature", "default")
    if model_temp == "default":
        body["temperature"] = temperature
    elif model_temp is not None:
        body["temperature"] = model_temp
    body.update(resolved.get("request_overrides") or {})
    return body


async def _call_openai_compat(resolved: dict, system: str, user: str,
                              max_tokens: int, temperature: float) -> LLMResponse:
    url = f"{resolved['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {resolved['api_key']}",
        "content-type": "application/json",
    }
    body = build_openai_body(resolved, system, user, max_tokens, temperature)
    t0 = time.monotonic()
    r = await _get_client().post(url, headers=headers, json=body)
    r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    if "choices" not in data or not data["choices"]:
        raise ProviderError(f"Malformed response (no choices): {str(data)[:300]}")
    choice = data["choices"][0]
    message = choice.get("message", {}) or {}
    raw_content = message.get("content") or ""
    raw_reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    content, reasoning = split_reasoning(raw_content, raw_reasoning)
    finish = choice.get("finish_reason", "") or ""
    usage = data.get("usage", {}) or {}
    return LLMResponse(
        model_id=resolved["model_id"], content=content,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=elapsed, finish_reason=finish,
        truncated=(finish == "length"), reasoning=reasoning, raw=data,
    )


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
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    model_temp = resolved.get("temperature", "default")
    if model_temp == "default":
        body["temperature"] = temperature
    elif model_temp is not None:
        body["temperature"] = model_temp
    body.update(resolved.get("request_overrides") or {})
    t0 = time.monotonic()
    r = await _get_client().post(url, headers=headers, json=body)
    r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    blocks = data.get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    thinking = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
    content, reasoning = split_reasoning(text, thinking)
    stop = data.get("stop_reason", "") or ""
    usage = data.get("usage", {})
    return LLMResponse(
        model_id=resolved["model_id"], content=content,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        latency_ms=elapsed, finish_reason=stop,
        truncated=(stop == "max_tokens"), reasoning=reasoning, raw=data,
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
    """
    Cheapest possible call that still exercises the real request shape
    (thinking toggle, token parameter, temperature handling).
    """
    return await call_model(
        resolved,
        system="Reply with the single word OK.",
        user="Say OK.",
        max_tokens=64,
        temperature=0.0,
    )
