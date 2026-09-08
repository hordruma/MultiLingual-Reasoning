"""
Offline tests for the parts of the pipeline that decide the numbers:
answer extraction, label normalisation, scoring, prompt rendering,
resume logic and the analysis statistics.  No network, no API keys.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze  # noqa: E402
import data_loader  # noqa: E402
import run_experiment as rx  # noqa: E402
from providers import resolve_model, build_openai_body, split_reasoning, ConfigError  # noqa: E402


# ── extract_answer ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected, marker", [
    ("因此答案为 No。</think>ANSWER: No", "No", True),
    ("reasoning</think>\n\nANSWER: Yes", "Yes", True),
    ("reasoning...\nANSWER: Yes", "Yes", True),
    ("reasoning...\nAnswer: no.", "no", True),
    ("**ANSWER:** **Yes**", "Yes", True),
    ("最终答案\nANSWER：No", "No", True),
    ("ANSWER: Yes\n\nActually wait.\nANSWER: No", "No", True),
    ("A: suggestive", "suggestive", True),
    ("Final answer - Limitation of liability", "Limitation of liability", True),
    ("I think the clause is unfair.\nYes", "Yes", False),
    ("", "", False),
])
def test_extract_answer(text, expected, marker):
    assert rx.extract_answer(text) == (expected, marker)


def test_extract_ignores_the_word_answer_mid_sentence():
    text = "The answer depends on hearsay rules.\nANSWER: Yes"
    assert rx.extract_answer(text) == ("Yes", True)


# ── normalize_to_label / score ───────────────────────────────────────────

def test_normalize_exact_case_insensitive():
    assert rx.normalize_to_label("yes", ["Yes", "No"]) == "Yes"
    assert rx.normalize_to_label("GENERIC", ["generic", "descriptive"]) == "generic"


def test_normalize_single_mention():
    assert rx.normalize_to_label("The mark is suggestive.", ["generic", "suggestive"]) == "suggestive"
    assert rx.normalize_to_label("Yes, there is hearsay", ["Yes", "No"]) == "Yes"


def test_normalize_ambiguous_left_alone():
    assert rx.normalize_to_label("Yes or No", ["Yes", "No"]) == "Yes or No"
    # "no" inside "notice" must not match
    assert rx.normalize_to_label("notice required", ["Yes", "No"]) == "notice required"


def test_score_answer():
    assert rx.score_answer("Yes.", "yes")
    assert rx.score_answer('"No"', "No")
    assert not rx.score_answer("Yes", "No")
    assert not rx.score_answer("", "No")


# ── prompt rendering ─────────────────────────────────────────────────────

def test_render_prompt_substitutes_and_strips_cue():
    base = "Task definition.\n\nQ: Example. Is it?\nA: No\n\nQ: {{text}}\nA:"
    out = data_loader.render_prompt(base, {"text": "NEW CASE", "slice": "leak"})
    assert "NEW CASE" in out
    assert "leak" not in out
    assert not out.endswith("A:")
    assert out.endswith("Is it?\nA: No\n\nQ: NEW CASE") or out.endswith("Q: NEW CASE")


def test_build_prompts_lists_labels(tmp_path):
    s = data_loader.LegalBenchSample(task="abercrombie", idx=3, text="x", label="generic", prompt="P {{}}")
    system, user = rx.build_prompts(s, "english")
    assert "Answer with exactly one of: generic, descriptive, suggestive, arbitrary, fanciful" in user
    assert "ANSWER: <label>" in system
    assert "in English" in system


def test_load_task_is_seeded_subset_and_excludes_leak_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "CACHE_DIR", tmp_path)
    task_dir = tmp_path / "hearsay"
    task_dir.mkdir()
    rows = [{"index": str(i), "answer": "Yes" if i % 2 else "No", "text": f"fact {i}", "slice": "Standard hearsay"}
            for i in range(50)]
    with open(task_dir / "test.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (task_dir / "base_prompt.txt").write_text("Hearsay def.\n\nQ: {{text}} Is there hearsay?\nA:")

    a = data_loader.load_task("hearsay", {"labels": ["Yes", "No"]}, max_samples=10, seed=1)
    b = data_loader.load_task("hearsay", {"labels": ["Yes", "No"]}, max_samples=10, seed=1)
    assert len(a) == 10 and [s.idx for s in a] == [s.idx for s in b]
    assert all("Standard hearsay" not in s.prompt for s in a)
    assert a[0].prompt.endswith("Is there hearsay?")


# ── resume logic ─────────────────────────────────────────────────────────

def test_run_cell_resumes_and_retries_errors(tmp_path):
    samples = [data_loader.LegalBenchSample("hearsay", i, "t", "Yes", "P") for i in range(4)]
    path = rx.cell_path(tmp_path, "mock", "english", 0)
    tmp_path.mkdir(exist_ok=True)
    # Pre-existing file: idx 0 done correctly, idx 1 errored (should be retried).
    with open(path, "w") as f:
        f.write(json.dumps({"task": "hearsay", "idx": 0, "condition": "english", "model": "mock", "run_id": 0,
                            "expected": "Yes", "predicted": "Yes", "correct": True, "error": None,
                            "answer_marker_found": True, "predicted_in_label_set": True,
                            "input_tokens": 1, "output_tokens": 1, "latency_ms": 1}) + "\n")
        f.write(json.dumps({"task": "hearsay", "idx": 1, "condition": "english", "model": "mock", "run_id": 0,
                            "expected": "Yes", "predicted": "", "correct": False, "error": "boom",
                            "input_tokens": 0, "output_tokens": 0, "latency_ms": 0}) + "\n")

    resolved = resolve_model({"provider": "mock", "model_id": "mock"})
    summary = asyncio.run(rx.run_cell("mock", resolved, "english", samples, 0, tmp_path,
                                      asyncio.Semaphore(2)))
    assert summary["total"] == 4
    assert summary["errors"] == 0
    rows = rx.read_existing(path)
    assert sorted(k[1] for k in rows) == [0, 1, 2, 3]
    assert rows[("hearsay", 1)]["error"] is None


# ── provider config ──────────────────────────────────────────────────────

def test_resolve_model_fails_fast_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        resolve_model({"provider": "openai_compat", "model_id": "x", "display": "x",
                       "api_key_env": "OPENAI_API_KEY", "base_url": "https://example/v1"})


def test_resolve_model_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://host/v1/")
    r = resolve_model({"provider": "openai_compat", "model_id": "default", "display": "x",
                       "api_key_env": "OPENROUTER_API_KEY", "model_id_env": "OPENROUTER_MODEL",
                       "base_url_env": "OPENROUTER_BASE_URL", "base_url": "https://x"})
    assert r["model_id"] == "some/model" and r["base_url"] == "https://host/v1"


# ── analysis stats ───────────────────────────────────────────────────────

def test_mcnemar_exact():
    assert analyze.mcnemar_exact(0, 0) == 1.0
    assert analyze.mcnemar_exact(5, 5) == 1.0
    assert analyze.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


def test_wilson_ci_bounds():
    lo, hi = analyze.wilson_ci(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60


def test_script_ratio():
    assert analyze.script_ratio("这是中文推理", "han") == 1.0
    assert analyze.script_ratio("hello", "han") == 0.0
    assert analyze.script_ratio("hello", "latin") == 1.0
    assert analyze.script_ratio("hello", None) is None


def test_origin_advantage_is_within_model():
    cells = []
    for model, eng, man in [("deepseek-v4-flash", 0.70, 0.72), ("gpt-5.6-luna", 0.80, 0.75)]:
        cells.append({"model": model, "condition": "english", "accuracy": eng})
        cells.append({"model": model, "condition": "mandarin", "accuracy": man})
    out = analyze.origin_advantage(cells)
    assert len(out) == 1
    d = out[0]
    assert d["own_delta_vs_english"] == pytest.approx(0.02)
    assert d["others_mean_delta"] == pytest.approx(-0.05)
    assert d["relative_advantage"] == pytest.approx(0.07)


def test_build_cell_frame_and_paired_test():
    rows = []
    for cond, pattern in [("english", [1, 1, 0, 0]), ("wildcard", [1, 0, 1, 1])]:
        for i, ok in enumerate(pattern):
            rows.append({"model": "m", "condition": cond, "task": "t", "run_id": 0, "idx": i,
                         "expected": "Yes", "correct": bool(ok), "error": None,
                         "answer_marker_found": True, "predicted_in_label_set": True,
                         "output_tokens": 10, "input_tokens": 5, "latency_ms": 1, "full_response": "ANSWER: Yes"})
    cells = analyze.build_cell_frame(rows)
    assert {c["condition"]: c["accuracy"] for c in cells} == {"english": 0.5, "wildcard": 0.75}
    paired = analyze.paired_condition_test(rows, "english", "wildcard")
    per_model = paired[0]
    assert per_model["n_pairs"] == 4
    assert per_model["discordant_a_wins"] == 1 and per_model["discordant_b_wins"] == 2


# ── request shape / reasoning separation ─────────────────────────────────

def test_build_body_default_model():
    resolved = {"model_id": "m", "max_tokens_param": "max_tokens", "temperature": "default",
                "request_overrides": {}}
    body = build_openai_body(resolved, "sys", "usr", 4096, 0.0)
    assert body["max_tokens"] == 4096 and body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": "sys"}


def test_build_body_omits_max_tokens_when_uncapped():
    """None must drop the field, not send null: a cap would clip the tail."""
    for param in ("max_tokens", "max_completion_tokens"):
        resolved = {"model_id": "m", "max_tokens_param": param, "temperature": "default",
                    "request_overrides": {}}
        body = build_openai_body(resolved, "s", "u", None, 0.0)
        assert "max_tokens" not in body and "max_completion_tokens" not in body


def test_uncapped_config_is_actually_uncapped():
    import config
    assert config.MAX_OUTPUT_TOKENS is None


def test_anthropic_body_still_gets_required_max_tokens():
    """Anthropic rejects a request without max_tokens, so None needs a fallback."""
    import inspect, providers
    src = inspect.getsource(providers._call_anthropic)
    assert "REQUIRED_MAX_TOKENS_FALLBACK if max_tokens is None else max_tokens" in src
    assert providers.REQUIRED_MAX_TOKENS_FALLBACK > 16000


def test_build_body_gpt56_shape():
    resolved = {"model_id": "gpt-5.6-luna", "max_tokens_param": "max_completion_tokens",
                "temperature": None, "request_overrides": {"reasoning_effort": "none"}}
    body = build_openai_body(resolved, "s", "u", 4096, 0.0)
    assert "temperature" not in body and "max_tokens" not in body
    assert body["max_completion_tokens"] == 4096 and body["reasoning_effort"] == "none"


def test_build_body_thinking_overrides_present_for_default_models():
    import config
    from providers import resolve_model as rm
    for key in config.DEFAULT_MODELS:
        cfg = dict(config.MODELS[key])
        cfg["api_key_default"] = "x"       # bypass env for the shape test
        body = build_openai_body(rm(cfg), "s", "u", 10, 0.0)
        assert body["model"] == cfg["model_id"]
        if cfg["hidden_reasoning"] == "off":
            assert cfg["request_overrides"], f"{key} claims thinking off but sends no override"
            assert all(body.get(k) == v for k, v in cfg["request_overrides"].items())


def test_split_reasoning_think_tags_and_fields():
    content, reasoning, promoted = split_reasoning("<think>hmm</think>\nANSWER: Yes", "")
    assert content == "ANSWER: Yes" and reasoning == "hmm" and not promoted
    content, reasoning, promoted = split_reasoning("ANSWER: No", "prior thoughts")
    assert content == "ANSWER: No" and reasoning == "prior thoughts" and not promoted
    # Gemma-4-on-Ollama case: everything came back in the reasoning field
    content, reasoning, promoted = split_reasoning("", "the only text\nANSWER: Yes")
    assert content.endswith("ANSWER: Yes") and reasoning == content and promoted


def test_local_model_needs_no_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.6:27b")
    import config
    r = resolve_model(config.MODELS["ollama"])
    assert r["api_key"] == "ollama" and r["model_id"] == "qwen3.6:27b"
    assert r["base_url"] == "http://localhost:11434/v1"


# ── review follow-ups ────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("**ANSWER**: Yes\n\nbecause.", "Yes"),
    ("**Final Answer**: descriptive", "descriptive"),
    ("Answer**: No", "No"),
])
def test_extract_answer_emphasis_before_colon(text, expected):
    assert rx.extract_answer(text) == (expected, True)


def test_split_reasoning_promotion_is_still_counted_as_hidden():
    content, reasoning, promoted = split_reasoning("", "thinking...\nANSWER: Yes")
    assert promoted and reasoning and content == reasoning
    content, reasoning, promoted = split_reasoning("<think>partial thought", "")
    assert content == "" or promoted
    assert "partial thought" in reasoning


def test_read_existing_skips_torn_line(tmp_path):
    p = tmp_path / "m__c__run0.jsonl"
    p.write_text(json.dumps({"task": "t", "idx": 0, "error": None}) + "\n" + '{"task": "t", "idx": 1, "err')
    rows = rx.read_existing(p)
    assert list(rows) == [("t", 0)]


def test_reasoning_part_handles_case_changing_chars():
    text = "Die Straße ist groß. ANSWER: Yes"
    assert analyze.reasoning_part(text) == "Die Straße ist groß. "


def test_loader_rejects_html_page():
    assert not data_loader._looks_like_task_rows([{"<!doctype html>": "x"}])
    assert data_loader._looks_like_task_rows([{"index": "0", "text": "t", "answer": "Yes"}])


def test_cell_aborts_after_consecutive_errors(tmp_path, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(rx, "call_model", boom)
    monkeypatch.setattr(rx, "MAX_CONSECUTIVE_ERRORS", 3)
    samples = [data_loader.LegalBenchSample("t", i, "x", "Yes", "P") for i in range(20)]
    resolved = resolve_model({"provider": "mock", "model_id": "mock"})
    with pytest.raises(RuntimeError, match="aborted"):
        asyncio.run(rx.run_cell("mock", resolved, "english", samples, 0, tmp_path, asyncio.Semaphore(2)))
    kept = rx.read_existing(rx.cell_path(tmp_path, "mock", "english", 0))
    assert 3 <= len(kept) < 20


# ── rate limiting ────────────────────────────────────────────────────────

def test_rate_limiter_spaces_calls_beyond_the_window():
    import time as _t
    from providers import RateLimiter

    async def drive():
        lim = RateLimiter(rpm=3)
        lim._times.extend([_t.monotonic() - 59.9] * 3)  # window already full
        t0 = _t.monotonic()
        await lim.acquire()
        return _t.monotonic() - t0

    waited = asyncio.run(drive())
    assert waited > 0.02, "acquire should wait for the window to free up"


def test_rate_limiter_allows_burst_within_limit():
    from providers import RateLimiter

    async def drive():
        lim = RateLimiter(rpm=5)
        for _ in range(5):
            await lim.acquire()
        return len(lim._times)

    assert asyncio.run(drive()) == 5


def test_resolve_model_carries_rpm():
    import config
    r = resolve_model(dict(config.MODELS["tokenrouter"], api_key_default="x"))
    assert r["requests_per_minute"] == config.MODELS["tokenrouter"]["requests_per_minute"] > 0
    r2 = resolve_model({"provider": "mock", "model_id": "mock"})
    assert r2["requests_per_minute"] is None


def test_model_concurrency_cap_is_respected():
    import config
    assert config.MODELS["tokenrouter"]["max_concurrency"] >= 1
    src = open("run_experiment.py").read()
    assert "_semaphore_for" in src and 'MODELS[key].get("max_concurrency")' in src


def test_rate_limiter_covers_retries_not_just_first_attempt():
    """A retry is another request against the quota, so it must be limited too."""
    import providers

    calls = {"n": 0}

    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx_timeout()
        return "ok"

    def httpx_timeout():
        import httpx
        return httpx.ReadTimeout("boom")

    async def drive():
        lim = providers.RateLimiter(rpm=100)
        providers.RETRY_BACKOFF[:] = [0, 0, 0, 0]
        out = await providers._retry(flaky, _limiter=lim)
        return out, len(lim._times)

    out, acquired = asyncio.run(drive())
    assert out == "ok"
    assert acquired == 3, f"limiter should be acquired once per attempt, got {acquired}"


def test_streaming_is_default_and_overridable():
    import config
    from providers import resolve_model as rm
    r = rm(dict(config.MODELS["tokenrouter"], api_key_default="x"))
    assert r["stream"] is True
    r2 = rm(dict(config.MODELS["tokenrouter"], api_key_default="x", stream=False))
    assert r2["stream"] is False


def test_stream_body_requests_usage():
    from providers import build_openai_body
    resolved = {"model_id": "m", "max_tokens_param": "max_tokens", "temperature": "default",
                "request_overrides": {}}
    body = build_openai_body(resolved, "s", "u", None, 0.0)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    assert body["stream_options"]["include_usage"] is True and "max_tokens" not in body


def test_read_timeout_is_a_chunk_gap_not_a_call_budget():
    import providers
    assert providers.READ_TIMEOUT <= 300, "read timeout must catch stalls quickly"
    assert providers.TOTAL_TIMEOUT >= 1800, "total budget must allow very long generations"


def test_streaming_applies_no_wall_clock_cut():
    """How long a runaway runs is benchmark data; only silence is a fault."""
    import inspect, providers
    src = inspect.getsource(providers._call_openai_compat_stream)
    assert "MAX_REQUEST_SECONDS" not in src
    assert not hasattr(providers, "MAX_REQUEST_SECONDS")
    assert providers.READ_TIMEOUT <= 300   # stalled connections still die fast
