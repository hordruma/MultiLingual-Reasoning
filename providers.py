"""
Provider Adapters
=================
Unified interface for calling all 6 model providers.
Each adapter takes a system prompt + user prompt and returns the full response.

Environment variables required:
  ANTHROPIC_API_KEY          – for Claude
  AZURE_OPENAI_ENDPOINT      – for GPT-4o
  AZURE_OPENAI_API_KEY       – for GPT-4o
  AZURE_MISTRAL_ENDPOINT     – for Mistral Large
  AZURE_MISTRAL_API_KEY      – for Mistral Large
  AZURE_DEEPSEEK_ENDPOINT    – for DeepSeek V3
  AZURE_DEEPSEEK_API_KEY     – for DeepSeek V3
  QWEN_API_KEY               – for Qwen (via DashScope or compatible)
  QWEN_BASE_URL              – for Qwen endpoint
  GEMINI_API_KEY             – for Gemini
"""

import os
import time
import json
import httpx
import asyncio
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Standardised response from any provider."""
    model_id: str
    content: str                # full text response (CoT + answer)
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: Optional[dict] = None  # provider-specific payload for debugging


class ProviderError(Exception):
    """Raised when an API call fails after retries."""
    pass


# ── Retry wrapper ────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 15]  # seconds


async def _retry(coro_fn, *args, **kwargs):
    """Retry async function with exponential backoff."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                print(f"  ⚠ Retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}")
                await asyncio.sleep(wait)
    raise ProviderError(f"Failed after {MAX_RETRIES} retries: {last_err}")


# ── Anthropic (Claude) ──────────────────────────────────────────────────

async def _call_anthropic(system: str, user: str, model_id: str,
                          max_tokens: int, temperature: float) -> LLMResponse:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
    return LLMResponse(
        model_id=model_id, content=text,
        input_tokens=data["usage"]["input_tokens"],
        output_tokens=data["usage"]["output_tokens"],
        latency_ms=elapsed, raw=data,
    )


# ── Azure OpenAI (GPT-4o) ──────────────────────────────────────────────

async def _call_azure_openai(system: str, user: str, model_id: str,
                             max_tokens: int, temperature: float) -> LLMResponse:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/openai/deployments/{model_id}/chat/completions?api-version=2024-10-21"
    headers = {
        "api-key": os.environ["AZURE_OPENAI_API_KEY"],
        "content-type": "application/json",
    }
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return LLMResponse(
        model_id=model_id, content=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=elapsed, raw=data,
    )


# ── Azure Mistral ───────────────────────────────────────────────────────

async def _call_azure_mistral(system: str, user: str, model_id: str,
                              max_tokens: int, temperature: float) -> LLMResponse:
    endpoint = os.environ["AZURE_MISTRAL_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['AZURE_MISTRAL_API_KEY']}",
        "content-type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return LLMResponse(
        model_id=model_id, content=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=elapsed, raw=data,
    )


# ── Azure DeepSeek ──────────────────────────────────────────────────────

async def _call_azure_deepseek(system: str, user: str, model_id: str,
                               max_tokens: int, temperature: float) -> LLMResponse:
    endpoint = os.environ["AZURE_DEEPSEEK_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['AZURE_DEEPSEEK_API_KEY']}",
        "content-type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return LLMResponse(
        model_id=model_id, content=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=elapsed, raw=data,
    )


# ── Qwen (user-configured endpoint) ────────────────────────────────────

async def _call_qwen(system: str, user: str, model_id: str,
                     max_tokens: int, temperature: float) -> LLMResponse:
    base_url = os.environ["QWEN_BASE_URL"].rstrip("/")
    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['QWEN_API_KEY']}",
        "content-type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return LLMResponse(
        model_id=model_id, content=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=elapsed, raw=data,
    )


# ── Gemini (user-configured, OpenAI-compatible) ────────────────────────

async def _call_gemini(system: str, user: str, model_id: str,
                       max_tokens: int, temperature: float) -> LLMResponse:
    api_key = os.environ["GEMINI_API_KEY"]
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
    elapsed = (time.monotonic() - t0) * 1000
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return LLMResponse(
        model_id=model_id, content=text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        latency_ms=elapsed, raw=data,
    )


# ── Dispatch ────────────────────────────────────────────────────────────

PROVIDER_MAP = {
    "anthropic":      _call_anthropic,
    "azure_openai":   _call_azure_openai,
    "azure_mistral":  _call_azure_mistral,
    "azure_deepseek": _call_azure_deepseek,
    "qwen":           _call_qwen,
    "gemini":         _call_gemini,
}


async def call_model(provider: str, system: str, user: str, model_id: str,
                     max_tokens: int = 2048, temperature: float = 0.0) -> LLMResponse:
    """Unified entry point – dispatches to the right provider with retries."""
    fn = PROVIDER_MAP.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider}")
    return await _retry(fn, system, user, model_id, max_tokens, temperature)
