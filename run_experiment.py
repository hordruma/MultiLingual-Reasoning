"""
Experiment Runner
=================
Orchestrates the models × conditions × tasks × runs matrix.

Usage:
    python run_experiment.py --list                 # show models / conditions / tasks
    python run_experiment.py --smoke-test           # one tiny call per configured model
    python run_experiment.py --dry-run              # show matrix, no API calls
    python run_experiment.py --estimate --runs 1    # rough cost estimate (loads data)
    python run_experiment.py --pilot --runs 1       # 8 conditions, default cheap models
    python run_experiment.py --models gpt-5.6-luna,deepseek-v4-flash --conditions english,mandarin
    python run_experiment.py --models ollama --concurrency 1 --runs 1   # local model via Ollama
    python run_experiment.py --models mock --runs 1 # offline pipeline check (fake answers)

Results are appended per sample to results/<model>__<condition>__run<N>.jsonl,
so an interrupted run resumes where it stopped.  Use --fresh to discard them.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from config import (
    MODELS, DEFAULT_MODELS, CONDITIONS, PILOT_CONDITIONS, LEGALBENCH_TASKS,
    MAX_TASKS_PER_BENCHMARK, SAMPLE_SEED, NUM_RUNS, MAX_OUTPUT_TOKENS,
    TEMPERATURE, RESULTS_DIR,
)
from providers import call_model, resolve_model, smoke_test, close_client, ConfigError
from data_loader import load_all_tasks, LegalBenchSample

load_dotenv()


# ── Prompt construction ──────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are an expert legal analyst. You will be given a legal reasoning task with its definition and a few worked examples, followed by a new instance to decide.

{condition_instruction}

The worked examples show only the final label; for the new instance follow the reasoning instruction above.
When you are done, put your final answer on its own line in exactly this form:
ANSWER: <label>
where <label> is one of the allowed labels listed at the end of the task, spelled exactly as listed."""


def build_prompts(sample: LegalBenchSample, condition_key: str) -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) for a given sample and condition."""
    condition = CONDITIONS[condition_key]
    system = SYSTEM_TEMPLATE.format(condition_instruction=condition["instruction"])
    labels = LEGALBENCH_TASKS.get(sample.task, {}).get("labels", [])
    label_line = f"\n\nAnswer with exactly one of: {', '.join(labels)}" if labels else ""
    user = f"{sample.prompt}{label_line}"
    return system, user


# ── Answer extraction ────────────────────────────────────────────────────

_ANSWER_RE = re.compile(
    r"^[\s\*\_#>`\-]*(?:final\s+)?(?:answer|a)[\*\_`]*\s*[:：\-–—]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_STRIP_RE = re.compile(r"^[\s\*\_`\"'“”‘’\[\(]+|[\s\*\_`\"'“”‘’\]\)\.。!,;:]+$")


def clean_answer(s: str) -> str:
    """Strip markdown emphasis, quotes and trailing punctuation."""
    return _STRIP_RE.sub("", s.strip())


def extract_answer(response_text: str) -> Tuple[str, bool]:
    """
    Return (answer, marker_found).  Looks for the last "ANSWER: x" line
    (also accepts "Answer:", "**ANSWER:**", full-width colon, "A:").
    Falls back to the last non-empty line when no marker is present.
    """
    text = response_text or ""
    matches = _ANSWER_RE.findall(text)
    if matches:
        return clean_answer(matches[-1]), True
    for line in reversed(text.strip().split("\n")):
        line = clean_answer(line)
        if line:
            return line, False
    return "", False


def normalize_to_label(pred: str, labels: List[str]) -> str:
    """
    Map a free-form prediction onto the task's label set.
    Exact (case-insensitive) match wins; otherwise a single unambiguous
    label mention inside the prediction is accepted.  Returns the original
    string when no label can be attributed.
    """
    if not labels:
        return pred
    p = pred.strip().lower()
    for label in labels:
        if p == label.lower():
            return label
    hits = [label for label in labels
            if re.search(r"(?<![a-z])" + re.escape(label.lower()) + r"(?![a-z])", p)]
    if len(hits) == 1:
        return hits[0]
    return pred


def score_answer(predicted: str, expected: str) -> bool:
    return clean_answer(predicted).lower() == clean_answer(expected).lower()


# ── Persistence helpers ──────────────────────────────────────────────────

MAX_CONSECUTIVE_ERRORS = 10   # abort a cell when the endpoint is clearly down

def cell_path(results_dir: Path, model_key: str, condition_key: str, run_id: int) -> Path:
    return results_dir / f"{model_key}__{condition_key}__run{run_id}.jsonl"


def read_existing(path: Path) -> Dict[Tuple[str, int], dict]:
    """Rows already written for a cell. A torn final line (crash mid-write) is skipped, not fatal."""
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  ⚠ {path.name}: skipping undecodable line {lineno} (partial write); it will be redone")
                    continue
                done[(r["task"], r["idx"])] = r
    return done


def summarize_cell(model_key: str, condition_key: str, run_id: int, rows: List[dict]) -> dict:
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    errors = sum(1 for r in rows if r.get("error"))
    answered = n - errors
    truncated = sum(1 for r in rows if r.get("truncated"))
    marker_missing = sum(1 for r in rows if not r.get("error") and not r.get("answer_marker_found"))
    unmapped = sum(1 for r in rows if not r.get("error") and not r.get("predicted_in_label_set"))
    hidden = sum(1 for r in rows if r.get("hidden_reasoning_chars", 0) > 0)
    out_tokens = sum(r.get("output_tokens", 0) for r in rows)
    in_tokens = sum(r.get("input_tokens", 0) for r in rows)
    latency = sum(r.get("latency_ms", 0) for r in rows)

    per_task = defaultdict(lambda: {"n": 0, "correct": 0, "errors": 0})
    for r in rows:
        t = per_task[r["task"]]
        t["n"] += 1
        t["correct"] += int(bool(r["correct"]))
        t["errors"] += int(bool(r.get("error")))
    for t in per_task.values():
        t["accuracy"] = t["correct"] / t["n"] if t["n"] else 0.0

    return {
        "model": model_key,
        "model_display": MODELS[model_key]["display"],
        "condition": condition_key,
        "condition_family": CONDITIONS[condition_key]["family"],
        "model_origin": MODELS[model_key]["origin_country"],
        "run_id": run_id,
        "total": n,
        "answered": answered,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,                 # errors count as wrong
        "accuracy_answered": correct / answered if answered else 0.0,
        "errors": errors,
        "truncated": truncated,
        "answer_marker_missing": marker_missing,
        "predicted_outside_label_set": unmapped,
        "hidden_reasoning_samples": hidden,
        "hidden_reasoning_policy": MODELS[model_key].get("hidden_reasoning", "unknown"),
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
        "avg_output_tokens": out_tokens / answered if answered else 0.0,
        "total_latency_ms": latency,
        "avg_latency_ms": latency / answered if answered else 0.0,
        "per_task": dict(per_task),
    }


# ── Single cell runner ──────────────────────────────────────────────────

async def _run_sample(resolved: dict, model_key: str, condition_key: str, run_id: int,
                      sample: LegalBenchSample, semaphore: asyncio.Semaphore) -> dict:
    system, user = build_prompts(sample, condition_key)
    labels = LEGALBENCH_TASKS.get(sample.task, {}).get("labels", [])
    base = {
        "task": sample.task, "idx": sample.idx, "condition": condition_key,
        "model": model_key, "run_id": run_id, "expected": sample.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    async with semaphore:
        try:
            resp = await call_model(resolved, system, user,
                                    max_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE)
        except Exception as e:  # noqa: BLE001
            return {**base, "predicted_raw": "", "predicted": "", "correct": False,
                    "answer_marker_found": False, "predicted_in_label_set": False,
                    "truncated": False, "finish_reason": "", "error": str(e)[:500],
                    "input_tokens": 0, "output_tokens": 0, "latency_ms": 0, "full_response": "",
                    "hidden_reasoning": "", "hidden_reasoning_chars": 0, "reasoning_promoted": False}

    raw_pred, marker = extract_answer(resp.content)
    pred = normalize_to_label(raw_pred, labels)
    in_set = (not labels) or any(pred.lower() == l.lower() for l in labels)
    return {
        **base,
        "predicted_raw": raw_pred,
        "predicted": pred,
        "correct": score_answer(pred, sample.label),
        "answer_marker_found": marker,
        "predicted_in_label_set": in_set,
        "truncated": resp.truncated,
        "finish_reason": resp.finish_reason,
        "error": None,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "latency_ms": resp.latency_ms,
        "full_response": resp.content,
        "hidden_reasoning": resp.reasoning,
        "hidden_reasoning_chars": len(resp.reasoning or ""),
        "reasoning_promoted": resp.reasoning_promoted,
    }


async def run_cell(model_key: str, resolved: dict, condition_key: str,
                   samples: List[LegalBenchSample], run_id: int, results_dir: Path,
                   semaphore: asyncio.Semaphore, fresh: bool = False) -> dict:
    """One model × one condition × one run over all samples; resumable."""
    path = cell_path(results_dir, model_key, condition_key, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and path.exists():
        path.unlink()
    existing = read_existing(path)
    # Retry samples that previously errored; keep everything else.
    todo = [s for s in samples if (s.task, s.idx) not in existing or existing[(s.task, s.idx)].get("error")]
    rows = {k: v for k, v in existing.items() if not v.get("error")}

    aborted = False
    if todo:
        t0 = time.monotonic()
        tasks = [asyncio.ensure_future(_run_sample(resolved, model_key, condition_key, run_id, s, semaphore))
                 for s in todo]
        consecutive_errors = 0
        with open(path, "a", encoding="utf-8") as f:
            for coro in asyncio.as_completed(tasks):
                r = await coro
                rows[(r["task"], r["idx"])] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                consecutive_errors = consecutive_errors + 1 if r.get("error") else 0
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"  ✗ {MAX_CONSECUTIVE_ERRORS} consecutive errors – aborting this cell "
                          f"(last: {r['error'][:120]})")
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    aborted = True
                    break
        took = time.monotonic() - t0
    else:
        took = 0.0

    # Rewrite the file without stale error rows, atomically, so an interrupt
    # here cannot lose the rows that were already paid for.
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for key in sorted(rows):
            f.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    if aborted:
        raise RuntimeError(f"cell aborted after {MAX_CONSECUTIVE_ERRORS} consecutive errors; "
                           f"{len(rows)} rows kept, rerun to resume")

    summary = summarize_cell(model_key, condition_key, run_id, list(rows.values()))
    print(
        f"  ✓ {model_key:18s} | {condition_key:14s} | run {run_id} | "
        f"acc={summary['accuracy']:.1%} (n={summary['total']}, err={summary['errors']}, "
        f"no-marker={summary['answer_marker_missing']}, hidden-cot={summary['hidden_reasoning_samples']}) | "
        f"out-tok={summary['total_output_tokens']:,} | "
        f"{'resumed' if not todo else f'{took:.0f}s'}"
    )
    return summary


# ── Estimate ─────────────────────────────────────────────────────────────

def estimate_cost(model_keys: List[str], condition_keys: List[str], samples: List[LegalBenchSample],
                  num_runs: int):
    print("\n── COST ESTIMATE (rough; prices in config.py are placeholders – verify them) ──\n")
    print(f"{'Model':<20} {'Calls':>8} {'In tok (M)':>11} {'Out tok (M)':>12} {'USD':>9}")
    grand = 0.0
    for m in model_keys:
        cfg = MODELS[m]
        in_tok = out_tok = 0
        for c in condition_keys:
            for s in samples:
                system, user = build_prompts(s, c)
                in_tok += (len(system) + len(user)) / 4
                out_tok += 15 if c == "no_cot" else 550
        in_tok *= num_runs
        out_tok *= num_runs
        calls = len(condition_keys) * len(samples) * num_runs
        cost = in_tok / 1e6 * cfg.get("price_in", 0) + out_tok / 1e6 * cfg.get("price_out", 0)
        grand += cost
        print(f"{m:<20} {calls:>8,} {in_tok/1e6:>11.2f} {out_tok/1e6:>12.2f} {cost:>9.2f}")
    print(f"{'TOTAL':<20} {'':>8} {'':>11} {'':>12} {grand:>9.2f}")
    print("\nAssumes ~550 output tokens per CoT answer and ~15 for no_cot.")


# ── Main orchestrator ────────────────────────────────────────────────────

async def run_experiment(model_keys: List[str], condition_keys: List[str], task_keys: List[str],
                         num_runs: int, max_concurrent: int, max_samples: int,
                         results_dir: Path, dry_run: bool, estimate: bool, fresh: bool):
    results_dir.mkdir(parents=True, exist_ok=True)
    task_cfgs = {t: LEGALBENCH_TASKS[t] for t in task_keys}

    total_cells = len(model_keys) * len(condition_keys) * num_runs
    print(f"\n{'=' * 70}\nLEGALBENCH COT LANGUAGE EXPERIMENT\n{'=' * 70}")
    print(f"Models:      {len(model_keys)} ({', '.join(model_keys)})")
    print(f"Conditions:  {len(condition_keys)} ({', '.join(condition_keys)})")
    print(f"Tasks:       {len(task_keys)} ({', '.join(task_keys)})")
    print(f"Runs:        {num_runs}   Max samples/task: {max_samples}   Concurrency: {max_concurrent}")
    print(f"Total cells: {total_cells}   Results: {results_dir}/")
    if "mock" in model_keys:
        print("\n!!! MOCK MODEL SELECTED – answers are fake, results are meaningless !!!")
    print(f"{'=' * 70}\n")

    if dry_run:
        print("DRY RUN – no API calls.\n")
        for m in model_keys:
            for r in range(num_runs):
                for c in condition_keys:
                    print(f"  Would run: {m} × {c} × run {r}")
        return

    # Fail fast on missing credentials before downloading anything.
    resolved_models = {}
    if not estimate:
        for m in model_keys:
            try:
                resolved_models[m] = resolve_model(MODELS[m])
            except ConfigError as e:
                print(f"✗ {m}: {e}")
                sys.exit(1)

    print("Loading LegalBench data...")
    all_data = load_all_tasks(task_cfgs, max_per_task=max_samples, seed=SAMPLE_SEED)
    all_samples = [s for samples in all_data.values() for s in samples]
    print(f"Total samples per cell: {len(all_samples)} across {len(all_data)} tasks\n")
    if not all_samples:
        print("ERROR: no samples loaded.")
        sys.exit(1)
    missing = [t for t in task_keys if t not in all_data]
    if missing:
        print(f"⚠ Tasks skipped (no data): {', '.join(missing)}\n")

    if estimate:
        estimate_cost(model_keys, condition_keys, all_samples, num_runs)
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    summary_file = results_dir / "experiment_summary.json"
    all_summaries: List[dict] = []
    start = time.monotonic()

    def write_summary(final: bool):
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "complete": final,
                "elapsed_seconds": time.monotonic() - start,
                "models": model_keys,
                "conditions": condition_keys,
                "tasks": list(all_data.keys()),
                "num_runs": num_runs,
                "max_samples_per_task": max_samples,
                "sample_seed": SAMPLE_SEED,
                "temperature": TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "total_samples_per_cell": len(all_samples),
                "contains_mock": "mock" in model_keys,
                "results": all_summaries,
            }, f, indent=2, ensure_ascii=False)

    try:
        for model_key in model_keys:
            print(f"\n── Model: {MODELS[model_key]['display']} ──")
            for run_id in range(num_runs):
                for condition_key in condition_keys:
                    try:
                        s = await run_cell(model_key, resolved_models[model_key], condition_key,
                                           all_samples, run_id, results_dir, semaphore, fresh=fresh)
                        all_summaries.append(s)
                    except Exception as e:  # noqa: BLE001
                        print(f"  ✗ Cell {model_key}/{condition_key}/run{run_id} failed: {e}")
                    write_summary(final=False)
    finally:
        await close_client()

    write_summary(final=True)
    elapsed = time.monotonic() - start
    print(f"\n{'=' * 70}\nEXPERIMENT COMPLETE\n{'=' * 70}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes   Summary: {summary_file}")

    print("\n── ACCURACY LEADERBOARD (mean over runs; errors count as wrong) ──\n")
    agg = defaultdict(list)
    for s in all_summaries:
        agg[(s["model"], s["condition"])].append(s["accuracy"])
    rows = sorted(((sum(v) / len(v), m, c) for (m, c), v in agg.items()), reverse=True)
    print(f"{'Rank':<5} {'Accuracy':<10} {'Model':<20} {'Condition':<18} {'Family'}")
    for i, (acc, m, c) in enumerate(rows[:40], 1):
        print(f"{i:<5} {acc:<10.1%} {m:<20} {c:<18} {CONDITIONS[c]['family']}")
    print("\nRun `python analyze.py` for the full report.")


# ── Smoke test ───────────────────────────────────────────────────────────

async def run_smoke_test(model_keys: List[str]) -> bool:
    ok = True
    print("\n── SMOKE TEST (one tiny request per model) ──\n")
    for m in model_keys:
        cfg = MODELS[m]
        try:
            resolved = resolve_model(cfg)
        except ConfigError as e:
            print(f"  ✗ {m:<20} not configured: {e}")
            ok = False
            continue
        try:
            resp = await smoke_test(resolved)
            hidden = f", hidden reasoning {len(resp.reasoning)} chars" if resp.reasoning else ""
            print(f"  ✓ {m:<20} {resolved['model_id']:<32} → {resp.content.strip()[:30]!r} "
                  f"({resp.latency_ms:.0f} ms, {resp.output_tokens} out tokens{hidden})")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {m:<20} {resolved.get('model_id', '')}: {e}")
            ok = False
    await close_client()
    return ok


async def list_remote_models(model_key: str):
    """GET {base_url}/models with the model's key and print the ids the key can see."""
    import httpx
    cfg = MODELS[model_key]
    if cfg["provider"] != "openai_compat":
        sys.exit(f"{model_key} is not an OpenAI-compatible provider")
    try:
        resolved = resolve_model(cfg)
    except ConfigError as e:
        sys.exit(f"✗ {e}")
    url = f"{resolved['base_url']}/models"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url, headers={"Authorization": f"Bearer {resolved['api_key']}"})
    if r.status_code != 200:
        sys.exit(f"✗ {url} → HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    items = data.get("data", data if isinstance(data, list) else [])
    print(f"\n{len(items)} models at {url}  (current default for {model_key}: {resolved['model_id']})\n")
    for m in sorted(items, key=lambda x: str(x.get("id", ""))):
        mid = m.get("id", "?")
        pricing = m.get("pricing") or {}
        price = ""
        if pricing:
            price = f"  in={pricing.get('prompt', pricing.get('input', '?'))} out={pricing.get('completion', pricing.get('output', '?'))}"
        print(f"  {mid}{price}")
    print(f"\nPick one with {cfg.get('model_id_env', 'the model_id in config.py')}=<id>")


def list_everything():
    print("\nMODELS (default set marked *; thinking = hidden-reasoning policy; env var = API key, * = optional):")
    for k, v in MODELS.items():
        mark = "*" if k in DEFAULT_MODELS else " "
        print(f"  {mark} {k:<22} {v['model_id']:<32} thinking={v.get('hidden_reasoning', '?'):<15} "
              f"{(v.get('api_key_env') or '-') + ('*' if v.get('api_key_default') else ''):<18} {v['display']}")
    print("\nCONDITIONS:")
    for k, v in CONDITIONS.items():
        print(f"    {k:<16} {v['family']}")
    print("\nTASKS:")
    for k, v in LEGALBENCH_TASKS.items():
        print(f"    {k:<58} labels={v['labels']} ~n={v['approx_test_size']}")


# ── CLI ──────────────────────────────────────────────────────────────────

def _split(arg: Optional[str]) -> Optional[List[str]]:
    return [x.strip() for x in arg.split(",") if x.strip()] if arg else None


def main():
    try:  # let `| head` close the pipe quietly
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="LegalBench CoT Language Experiment")
    parser.add_argument("--list", action="store_true", help="List models, conditions and tasks")
    parser.add_argument("--remote-models", type=str, default=None, metavar="MODEL_KEY",
                        help="Query an OpenAI-compatible provider's /models endpoint (e.g. tokenrouter, ollama)")
    parser.add_argument("--smoke-test", action="store_true", help="One tiny call per model, then exit")
    parser.add_argument("--dry-run", action="store_true", help="Show matrix without API calls")
    parser.add_argument("--estimate", action="store_true", help="Load data and print a rough cost estimate")
    parser.add_argument("--pilot", action="store_true", help=f"Pilot conditions: {', '.join(PILOT_CONDITIONS)}")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys, or 'all'")
    parser.add_argument("--conditions", type=str, default=None, help="Comma-separated condition keys")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task names")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Runs per cell")
    parser.add_argument("--max-samples", type=int, default=MAX_TASKS_PER_BENCHMARK, help="Max samples per task")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--fresh", action="store_true", help="Discard existing per-cell results instead of resuming")
    args = parser.parse_args()

    if args.list:
        list_everything()
        return
    if args.remote_models:
        if args.remote_models not in MODELS:
            sys.exit(f"Unknown model: {args.remote_models}  (see --list)")
        asyncio.run(list_remote_models(args.remote_models))
        return

    if args.models == "all":
        model_keys = [m for m in MODELS if m != "mock"]
    else:
        model_keys = _split(args.models) or list(DEFAULT_MODELS)
    condition_keys = _split(args.conditions) or (PILOT_CONDITIONS if args.pilot else list(CONDITIONS))
    task_keys = _split(args.tasks) or list(LEGALBENCH_TASKS)

    for m in model_keys:
        if m not in MODELS:
            sys.exit(f"Unknown model: {m}  (see --list)")
    for c in condition_keys:
        if c not in CONDITIONS:
            sys.exit(f"Unknown condition: {c}  (see --list)")
    for t in task_keys:
        if t not in LEGALBENCH_TASKS:
            sys.exit(f"Unknown task: {t}  (see --list)")

    if args.smoke_test:
        ok = asyncio.run(run_smoke_test(model_keys))
        sys.exit(0 if ok else 1)

    asyncio.run(run_experiment(
        model_keys=model_keys, condition_keys=condition_keys, task_keys=task_keys,
        num_runs=args.runs, max_concurrent=args.concurrency, max_samples=args.max_samples,
        results_dir=Path(args.results_dir), dry_run=args.dry_run, estimate=args.estimate,
        fresh=args.fresh,
    ))


if __name__ == "__main__":
    main()
