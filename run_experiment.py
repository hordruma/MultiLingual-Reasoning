"""
Experiment Runner
=================
Orchestrates the full 21 conditions × 6 models × N tasks × 3 runs matrix.

Usage:
    python run_experiment.py                    # full run
    python run_experiment.py --pilot            # pilot: 7 conditions only
    python run_experiment.py --dry-run          # show matrix without API calls
    python run_experiment.py --models claude-sonnet,gpt-4o  # subset of models
    python run_experiment.py --conditions english,mandarin  # subset of conditions
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

from config import MODELS, CONDITIONS, LEGALBENCH_TASKS, MAX_TASKS_PER_BENCHMARK, NUM_RUNS, MAX_OUTPUT_TOKENS, TEMPERATURE, RESULTS_DIR
from providers import call_model, LLMResponse
from data_loader import load_all_tasks, LegalBenchSample


# ── Prompt construction ──────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are an expert legal analyst. You will be given a legal reasoning task.

{condition_instruction}

After your reasoning, clearly mark your final answer on a new line starting with "ANSWER: " followed by your answer.
The answer should match the expected format exactly (e.g., "Yes", "No", or the specific label requested)."""


def build_prompts(sample: LegalBenchSample, condition_key: str) -> tuple:
    """Build (system_prompt, user_prompt) for a given sample and condition."""
    condition = CONDITIONS[condition_key]
    system = SYSTEM_TEMPLATE.format(condition_instruction=condition["instruction"])
    user = f"Task: {sample.task}\n\n{sample.input_text}"
    return system, user


# ── Answer extraction ────────────────────────────────────────────────────

def extract_answer(response_text: str) -> str:
    """Extract the final answer from model response."""
    # Look for "ANSWER: ..." pattern
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()

    # Fallback: take the last non-empty line
    for line in reversed(lines):
        line = line.strip()
        if line:
            return line

    return ""


def score_answer(predicted: str, expected: str) -> bool:
    """Compare predicted answer to ground truth (case-insensitive, trimmed)."""
    p = predicted.lower().strip().rstrip(".")
    e = expected.lower().strip().rstrip(".")
    return p == e


# ── Single cell runner ──────────────────────────────────────────────────

async def run_cell(
    model_key: str,
    condition_key: str,
    samples: List[LegalBenchSample],
    run_id: int,
    results_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Dict:
    """Run one cell: one model × one condition × all samples × one run."""
    model_cfg = MODELS[model_key]
    cell_results = []
    correct = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0.0

    for sample in samples:
        system, user = build_prompts(sample, condition_key)

        async with semaphore:
            try:
                resp = await call_model(
                    provider=model_cfg["provider"],
                    system=system,
                    user=user,
                    model_id=model_cfg["model_id"],
                    max_tokens=MAX_OUTPUT_TOKENS,
                    temperature=TEMPERATURE,
                )

                predicted = extract_answer(resp.content)
                is_correct = score_answer(predicted, sample.label)
                if is_correct:
                    correct += 1

                total_input_tokens += resp.input_tokens
                total_output_tokens += resp.output_tokens
                total_latency += resp.latency_ms

                cell_results.append({
                    "task": sample.task,
                    "idx": sample.idx,
                    "condition": condition_key,
                    "model": model_key,
                    "run_id": run_id,
                    "predicted": predicted,
                    "expected": sample.label,
                    "correct": is_correct,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "latency_ms": resp.latency_ms,
                    "full_response": resp.content,
                })

            except Exception as e:
                cell_results.append({
                    "task": sample.task,
                    "idx": sample.idx,
                    "condition": condition_key,
                    "model": model_key,
                    "run_id": run_id,
                    "predicted": "",
                    "expected": sample.label,
                    "correct": False,
                    "error": str(e),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "full_response": "",
                })

    n = len(samples)
    accuracy = correct / n if n > 0 else 0

    # Save detailed results
    out_file = results_dir / f"{model_key}__{condition_key}__run{run_id}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for r in cell_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model": model_key,
        "condition": condition_key,
        "condition_family": CONDITIONS[condition_key]["family"],
        "model_origin": model_cfg["origin_country"],
        "run_id": run_id,
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_output_tokens": total_output_tokens / n if n > 0 else 0,
        "total_latency_ms": total_latency,
        "avg_latency_ms": total_latency / n if n > 0 else 0,
    }

    print(
        f"  ✓ {model_key} | {condition_key:15s} | run {run_id} | "
        f"acc={accuracy:.1%} | tokens={total_output_tokens:,} | "
        f"latency={total_latency/1000:.1f}s"
    )

    return summary


# ── Main orchestrator ────────────────────────────────────────────────────

async def run_experiment(
    model_keys: List[str],
    condition_keys: List[str],
    num_runs: int = NUM_RUNS,
    max_concurrent: int = 5,
    dry_run: bool = False,
):
    """Run the full experiment matrix."""
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Calculate matrix size
    total_cells = len(model_keys) * len(condition_keys) * num_runs
    print(f"\n{'='*70}")
    print(f"LEGALBENCH COT LANGUAGE EXPERIMENT")
    print(f"{'='*70}")
    print(f"Models:     {len(model_keys)} ({', '.join(model_keys)})")
    print(f"Conditions: {len(condition_keys)}")
    print(f"Runs:       {num_runs}")
    print(f"Total cells: {total_cells}")
    print(f"Tasks:      {', '.join(LEGALBENCH_TASKS)}")
    print(f"{'='*70}\n")

    if dry_run:
        print("DRY RUN – no API calls will be made.\n")
        for m in model_keys:
            for c in condition_keys:
                for r in range(num_runs):
                    print(f"  Would run: {m} × {c} × run {r}")
        print(f"\nTotal: {total_cells} cells")
        return

    # Load data
    print("Loading LegalBench data...")
    all_data = load_all_tasks(LEGALBENCH_TASKS, max_per_task=MAX_TASKS_PER_BENCHMARK)
    all_samples = []
    for task_name, samples in all_data.items():
        all_samples.extend(samples)
    print(f"Total samples: {len(all_samples)}\n")

    if not all_samples:
        print("ERROR: No samples loaded. Check data availability.")
        sys.exit(1)

    # Run matrix
    semaphore = asyncio.Semaphore(max_concurrent)
    all_summaries = []
    start_time = time.monotonic()

    for model_key in model_keys:
        print(f"\n── Model: {MODELS[model_key]['display']} ──")
        for run_id in range(num_runs):
            tasks = []
            for condition_key in condition_keys:
                tasks.append(
                    run_cell(model_key, condition_key, all_samples, run_id, results_dir, semaphore)
                )
            summaries = await asyncio.gather(*tasks, return_exceptions=True)
            for s in summaries:
                if isinstance(s, Exception):
                    print(f"  ✗ Error: {s}")
                else:
                    all_summaries.append(s)

    elapsed = time.monotonic() - start_time

    # Save summary
    summary_file = results_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "models": model_keys,
            "conditions": condition_keys,
            "num_runs": num_runs,
            "total_samples_per_cell": len(all_samples),
            "results": all_summaries,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Elapsed: {elapsed/60:.1f} minutes")
    print(f"Results: {summary_file}")
    print(f"Detailed: {results_dir}/")

    # Print leaderboard
    print(f"\n── ACCURACY LEADERBOARD (averaged across runs) ──\n")
    from collections import defaultdict
    agg = defaultdict(list)
    for s in all_summaries:
        key = (s["model"], s["condition"])
        agg[key].append(s["accuracy"])

    rows = []
    for (model, condition), accs in agg.items():
        avg = sum(accs) / len(accs)
        rows.append((avg, model, condition, CONDITIONS[condition]["family"]))

    rows.sort(reverse=True)
    print(f"{'Rank':<5} {'Accuracy':<10} {'Model':<20} {'Condition':<18} {'Family'}")
    print(f"{'-'*5} {'-'*10} {'-'*20} {'-'*18} {'-'*15}")
    for i, (acc, model, cond, family) in enumerate(rows[:30], 1):
        print(f"{i:<5} {acc:<10.1%} {model:<20} {cond:<18} {family}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LegalBench CoT Language Experiment")
    parser.add_argument("--pilot", action="store_true", help="Run pilot (7 conditions)")
    parser.add_argument("--dry-run", action="store_true", help="Show matrix without API calls")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys")
    parser.add_argument("--conditions", type=str, default=None, help="Comma-separated condition keys")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs per cell")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API calls")
    args = parser.parse_args()

    # Select models
    if args.models:
        model_keys = [m.strip() for m in args.models.split(",")]
    else:
        model_keys = list(MODELS.keys())

    # Select conditions
    if args.conditions:
        condition_keys = [c.strip() for c in args.conditions.split(",")]
    elif args.pilot:
        condition_keys = [
            "english", "mandarin", "french", "arabic", "finnish",
            "formal_logic", "wildcard",
        ]
        # Note: french isn't in our conditions – swap to german for pilot
        condition_keys = [
            "english", "mandarin", "german", "arabic", "finnish",
            "formal_logic", "wildcard",
        ]
    else:
        condition_keys = list(CONDITIONS.keys())

    # Validate
    for m in model_keys:
        if m not in MODELS:
            print(f"Unknown model: {m}")
            sys.exit(1)
    for c in condition_keys:
        if c not in CONDITIONS:
            print(f"Unknown condition: {c}")
            sys.exit(1)

    asyncio.run(run_experiment(
        model_keys=model_keys,
        condition_keys=condition_keys,
        num_runs=args.runs,
        max_concurrent=args.concurrency,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
