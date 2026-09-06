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
import collections
import json
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
    reasoning_promoted: bool = False  # content was empty; reasoning text used as the answer
    raw: Optional[dict] = None  # provider-specific payload for debugging


# Anthropic's Messages API requires max_tokens, so an uncapped run needs a
# concrete number there. Well above any observed completion length.
REQUIRED_MAX_TOKENS_FALLBACK = 32000


class ProviderError(Exception):
    """Raised when an API call fails after retries (or is not retryable)."""


class ConfigError(Exception):
    """Raised when a model is missing its key / endpoint configuration."""


# ── Retry policy ─────────────────────────────────────────────────────────

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 5, 15, 30]          # seconds, plus jitter
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
# Streaming makes the read timeout a *gap between chunks*, so a generation may
# run arbitrarily long (no output cap) while a stalled connection is caught in
# ~2 min instead of blocking a concurrency slot for the whole call.
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 120.0          # max silence between streamed chunks
TOTAL_TIMEOUT = 3600.0        # httpx pool/default; NOT a whole-request deadline
# There is deliberately NO wall-clock cut on a streamed generation. How long a
# model runs before it stops is benchmark data, not a fault: 9% of this model's
# degenerate loops run past 40 minutes, reaching 139k tokens and 55 minutes, and
# cutting them would silently truncate that distribution. READ_TIMEOUT already
# kills a genuinely stalled connection (silence, not output), and runaways do
# terminate on their own at the model's ceiling, so nothing here is unbounded.
TIMEOUT_SECONDS = TOTAL_TIMEOUT

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """One shared connection pool for the whole run."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(
            TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=CONNECT_TIMEOUT))
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


CONNECT_RETRIES = 2                     # unreachable host: one quick retry, then give up


async def _retry(coro_fn, *args, _limiter: "Optional[RateLimiter]" = None, **kwargs):
    """
    Retry on transient failures only. 4xx auth/validation errors fail fast.

    The rate limiter is acquired before EVERY attempt, not once per call:
    a retry is another request against the provider's quota, so limiting only
    the first attempt lets a burst of retries blow straight through the cap.
    """
    last_err: Optional[Exception] = None
    max_attempts = MAX_RETRIES
    for attempt in range(MAX_RETRIES):
        try:
            if _limiter is not None:
                await _limiter.acquire()
            return await coro_fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 – we classify below
            last_err = e
            if not _is_retryable(e):
                raise ProviderError(f"Non-retryable error: {_describe(e)}") from e
            if isinstance(e, httpx.ConnectError):
                max_attempts = CONNECT_RETRIES
            if attempt >= max_attempts - 1:
                break
            if attempt < MAX_RETRIES - 1:
                wait = _retry_after(e) or (RETRY_BACKOFF[attempt] + random.uniform(0, 1))
                print(f"  ⚠ Retry {attempt + 1}/{MAX_RETRIES} after {wait:.0f}s: {_describe(e)}")
                await asyncio.sleep(wait)
    raise ProviderError(f"Failed after {max_attempts} attempts: {_describe(last_err)}") from last_err


def _describe(exc: Optional[Exception]) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:300].replace("\n", " ")
        return f"HTTP {exc.response.status_code} from {exc.request.url.host}: {body}"
    return f"{type(exc).__name__}: {exc}"


# ── Rate limiting ───────────────────────────────────────────────────────
# Free tiers cap requests per minute (TokenRouter's free GLM-5.3 allows 8/min).
# Without a limiter the runner burns its retry budget on 429s, so a model may
# declare `requests_per_minute` in config and calls are spaced accordingly.

class RateLimiter:
    """Sliding-window limiter: at most `rpm` acquisitions in any 60 s."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._times: collections.deque = collections.deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0]) + 0.05
            await asyncio.sleep(wait)


_limiters: dict = {}


def get_limiter(key: str, rpm: Optional[int]) -> Optional[RateLimiter]:
    """One limiter per model key, so all its concurrent calls share the window."""
    if not rpm:
        return None
    if key not in _limiters or _limiters[key].rpm != rpm:
        _limiters[key] = RateLimiter(rpm)
    return _limiters[key]


def reset_limiters():
    _limiters.clear()


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
        "requests_per_minute": model_cfg.get("requests_per_minute"),
        "stream": model_cfg.get("stream", True),
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
_THINK_OPEN_RE = re.compile(r"<think>(.*)$", re.DOTALL | re.IGNORECASE)


def split_reasoning(content: str, reasoning: str = "") -> Tuple[str, str, bool]:
    """
    Return (visible_content, hidden_reasoning, promoted).  Inline <think>
    blocks (closed or truncated-open) are moved out of the content.  If the
    content is empty but reasoning is present (Ollama + Gemma 4 on the /v1
    endpoint do this; a thinking model that spent its whole budget thinking
    does too), the reasoning text is used as the visible content so an answer
    can still be extracted, BUT it stays recorded as hidden reasoning and
    `promoted` is True so the row is counted as a hidden-reasoning sample.
    """
    content = content or ""
    reasoning = reasoning or ""
    blocks = _THINK_RE.findall(content)
    if blocks:
        reasoning = "\n".join([reasoning] + [b.strip() for b in blocks]).strip()
        content = _THINK_RE.sub("", content).strip()
    open_block = _THINK_OPEN_RE.search(content)
    if open_block:  # unclosed <think>: the output was cut off mid-thought
        reasoning = "\n".join([reasoning, open_block.group(1).strip()]).strip()
        content = content[:open_block.start()].strip()
    if not content.strip() and reasoning.strip():
        return reasoning.strip(), reasoning.strip(), True
    return content, reasoning, False


# ── OpenAI-compatible chat completions ──────────────────────────────────

def build_openai_body(resolved: dict, system: str, user: str,
                      max_tokens: Optional[int], temperature: float) -> dict:
    """
    Pure function so the request shape can be unit-tested.
    `max_tokens=None` omits the field entirely: the model stops when it is
    done, so nothing the study measures can be clipped by an arbitrary cap.
    """
    body = {
        "model": resolved["model_id"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if max_tokens is not None:
        body[resolved.get("max_tokens_param", "max_tokens")] = max_tokens
    model_temp = resolved.get("temperature", "default")
    if model_temp == "default":
        body["temperature"] = temperature
    elif model_temp is not None:
        body["temperature"] = model_temp
    body.update(resolved.get("request_overrides") or {})
    return body


async def _call_openai_compat_stream(resolved: dict, system: str, user: str,
                                     max_tokens, temperature: float) -> LLMResponse:
    """
    Streamed chat completion. Preferred because the read timeout then applies
    between chunks: an uncapped generation can take as long as it needs, while
    a stalled gateway is caught in seconds instead of holding a concurrency
    slot for the whole call (which collapsed throughput on a live run).
    """
    url = f"{resolved['base_url']}/chat/completions"
    headers = {"Authorization": f"Bearer {resolved['api_key']}", "content-type": "application/json"}
    body = build_openai_body(resolved, system, user, max_tokens, temperature)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})

    content_parts, reasoning_parts = [], []
    finish, usage = "", {}
    t0 = time.monotonic()
    async with _get_client().stream("POST", url, headers=headers, json=body) as r:
        if r.status_code >= 400:
            await r.aread()          # load the body so the error message is usable
            r.raise_for_status()
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []) or []:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc:
                    reasoning_parts.append(rc)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    elapsed = (time.monotonic() - t0) * 1000
    content, reasoning, promoted = split_reasoning("".join(content_parts), "".join(reasoning_parts))
    return LLMResponse(
        model_id=resolved["model_id"], content=content,
        input_tokens=usage.get("prompt_tokens") or 0,
        output_tokens=usage.get("completion_tokens") or 0,
        latency_ms=elapsed, finish_reason=finish,
        truncated=(finish == "length"), reasoning=reasoning,
        reasoning_promoted=promoted, raw=None,
    )


async def _call_openai_compat(resolved: dict, system: str, user: str,
                              max_tokens: int, temperature: float) -> LLMResponse:
    if resolved.get("stream", True):
        return await _call_openai_compat_stream(resolved, system, user, max_tokens, temperature)
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
    content, reasoning, promoted = split_reasoning(raw_content, raw_reasoning)
    finish = choice.get("finish_reason", "") or ""
    usage = data.get("usage", {}) or {}
    return LLMResponse(
        model_id=resolved["model_id"], content=content,
        input_tokens=usage.get("prompt_tokens") or 0,
        output_tokens=usage.get("completion_tokens") or 0,
        latency_ms=elapsed, finish_reason=finish,
        truncated=(finish == "length"), reasoning=reasoning,
        reasoning_promoted=promoted, raw=data,
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
        # Anthropic requires this field, so an uncapped run uses the fallback.
        "max_tokens": REQUIRED_MAX_TOKENS_FALLBACK if max_tokens is None else max_tokens,
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
    content, reasoning, promoted = split_reasoning(text, thinking)
    stop = data.get("stop_reason", "") or ""
    usage = data.get("usage", {}) or {}
    return LLMResponse(
        model_id=resolved["model_id"], content=content,
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        latency_ms=elapsed, finish_reason=stop,
        truncated=(stop == "max_tokens"), reasoning=reasoning,
        reasoning_promoted=promoted, raw=data,
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
                     max_tokens: Optional[int] = None, temperature: float = 0.0) -> LLMResponse:
    """Unified entry point – dispatches to the right provider with retries."""
    fn = PROVIDER_MAP.get(resolved["provider"])
    if fn is None:
        raise ValueError(f"Unknown provider: {resolved['provider']}")
    limiter = get_limiter(resolved["model_id"], resolved.get("requests_per_minute"))
    return await _retry(fn, resolved, system, user, max_tokens, temperature, _limiter=limiter)


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
