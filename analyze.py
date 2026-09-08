"""
Results Analyzer
================
Reads the per-sample JSONL files in results/ (the ground truth of a run) and
produces a text report plus CSV exports.  No pandas needed; the notebook
imports `load_sample_rows` / `build_cell_frame` from here.

Usage:
    python analyze.py                        # analyze results/
    python analyze.py --results-dir other/   # specify directory
"""

import argparse
import csv
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from config import MODELS, CONDITIONS, LEGALBENCH_TASKS


# ── Loading ──────────────────────────────────────────────────────────────


def prompt_version_of(row: dict) -> int:
    """Rows written before the field existed were all made with prompt version 1."""
    return int(row.get("prompt_version") or 1)


def has_hidden_reasoning(row: dict) -> bool:
    """Hidden reasoning came back either as text (reasoning_content) or only as a
    token count in usage (OpenAI-style providers never return the text)."""
    return (row.get("hidden_reasoning_chars") or 0) > 0 or (row.get("reasoning_tokens") or 0) > 0

def load_sample_rows(results_dir: Path) -> List[dict]:
    """Every per-sample record from every <model>__<condition>__runN.jsonl file."""
    rows = []
    for f in sorted(Path(results_dir).glob("*__*__run*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def build_cell_frame(rows: Iterable[dict]) -> List[dict]:
    """
    Aggregate per-sample rows to (model, condition, task, run) cells.
    Returns plain dicts; the notebook wraps them in a DataFrame.
    """
    cells = defaultdict(list)
    for r in rows:
        cells[(r["model"], r["condition"], r["task"], r["run_id"])].append(r)
    out = []
    for (model, cond, task, run), rs in sorted(cells.items()):
        n = len(rs)
        answered = [r for r in rs if not r.get("error")]
        correct = sum(1 for r in rs if r["correct"])
        out_tok = sum(r.get("output_tokens", 0) for r in answered)
        out.append({
            "model": model,
            "model_display": MODELS.get(model, {}).get("display", model),
            "model_origin": MODELS.get(model, {}).get("origin_country", "?"),
            "condition": cond,
            "condition_family": CONDITIONS.get(cond, {}).get("family", "?"),
            "task": task,
            "run": run,
            "total": n,
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
            "errors": n - len(answered),
            "answer_marker_missing": sum(1 for r in answered if not r.get("answer_marker_found")),
            "predicted_outside_label_set": sum(1 for r in answered if not r.get("predicted_in_label_set", True)),
            "truncated": sum(1 for r in answered if r.get("truncated")),
            "complete": sum(1 for r in answered if not r.get("truncated")),
            "correct_complete": sum(1 for r in answered if r["correct"] and not r.get("truncated")),
            # accuracy over samples that terminated; runaways are failures to answer, not wrong answers
            "accuracy_complete": (sum(1 for r in answered if r["correct"] and not r.get("truncated"))
                                  / max(1, sum(1 for r in answered if not r.get("truncated")))),
            "hidden_reasoning": sum(1 for r in answered if has_hidden_reasoning(r)),
            "avg_output_tokens": out_tok / len(answered) if answered else 0.0,
            "total_output_tokens": out_tok,
            "total_input_tokens": sum(r.get("input_tokens", 0) for r in answered),
            "avg_latency_ms": (sum(r.get("latency_ms", 0) for r in answered) / len(answered)) if answered else 0.0,
        })
    return out


# ── Small stats helpers ──────────────────────────────────────────────────

def _mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """
    Exact McNemar test on discordant pairs: b = A right & B wrong,
    c = A wrong & B right.  Two-sided p-value under Binomial(b+c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# ── Language-compliance heuristic ────────────────────────────────────────

_SCRIPT_RANGES = {
    "cyrillic":   [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "devanagari": [(0x0900, 0x097F)],
    "han":        [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3000, 0x303F)],
    "arabic":     [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "hebrew":     [(0x0590, 0x05FF)],
    "japanese":   [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF), (0x3000, 0x303F)],
    "hangul":     [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "latin":      [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
}


def script_ratio(text: str, script: Optional[str]) -> Optional[float]:
    """
    Fraction of alphabetic characters in `text` that belong to `script`.
    Only a sanity check: it says whether the reasoning used the expected
    writing system, not whether it was the expected language.
    """
    if not script or script not in _SCRIPT_RANGES:
        return None
    ranges = _SCRIPT_RANGES[script]
    letters = hits = 0
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        letters += 1
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in ranges):
            hits += 1
    return hits / letters if letters else None


_ANSWER_WORD = re.compile(r"answer", re.IGNORECASE)


def reasoning_part(full_response: str) -> str:
    """Everything before the final ANSWER line."""
    text = full_response or ""
    last = None
    for m in _ANSWER_WORD.finditer(text):
        last = m
    return text[:last.start()] if last and last.start() > 0 else text


# ── Analyses ─────────────────────────────────────────────────────────────

def condition_table(cells: List[dict]) -> List[dict]:
    """
    Accuracy per condition.  Both columns are SAMPLE-POOLED (sum correct / sum
    samples), the same definition the paired tests use, so a task with 50
    samples does not count as much as one with 200.  `std` is the spread of
    per-cell (model x task x run) accuracies, kept only as a dispersion hint.
    """
    by = defaultdict(list)
    raw = defaultdict(lambda: [0, 0])    # [correct, total]
    comp = defaultdict(lambda: [0, 0])   # [correct_complete, complete]
    fam = {}
    for c in cells:
        by[c["condition"]].append(c["accuracy"])
        raw[c["condition"]][0] += c.get("correct", 0)
        raw[c["condition"]][1] += c.get("total", 0)
        comp[c["condition"]][0] += c.get("correct_complete", 0)
        comp[c["condition"]][1] += c.get("complete", 0)
        fam[c["condition"]] = c["condition_family"]
    return sorted(
        [{"condition": k, "family": fam[k],
          "mean": (raw[k][0] / raw[k][1]) if raw[k][1] else 0.0, "n": raw[k][1],
          "std": _std(v), "n_cells": len(v),
          "mean_complete": (comp[k][0] / comp[k][1]) if comp[k][1] else 0.0,
          "n_complete": comp[k][1]}
         for k, v in by.items()],
        key=lambda d: -d["mean"],
    )


def model_table(cells: List[dict]) -> List[dict]:
    by = defaultdict(list)
    for c in cells:
        by[c["model"]].append(c["accuracy"])
    return sorted([{"model": k, "mean": _mean(v), "std": _std(v), "n_cells": len(v)} for k, v in by.items()],
                  key=lambda d: -d["mean"])


def family_table(cells: List[dict]) -> List[dict]:
    by = defaultdict(list)
    for c in cells:
        by[c["condition_family"]].append(c["accuracy"])
    return sorted([{"family": k, "mean": _mean(v), "std": _std(v)} for k, v in by.items()],
                  key=lambda d: -d["mean"])


def majority_baselines(rows: List[dict]) -> Dict[str, dict]:
    """Majority-class accuracy per task, from the expected labels actually used."""
    seen = {}
    for r in rows:
        seen[(r["task"], r["idx"])] = r["expected"]
    per_task = defaultdict(list)
    for (task, _), label in seen.items():
        per_task[task].append(label.lower())
    out = {}
    for task, labels in per_task.items():
        counts = defaultdict(int)
        for l in labels:
            counts[l] += 1
        top = max(counts.items(), key=lambda kv: kv[1])
        out[task] = {"n": len(labels), "majority_label": top[0], "majority_acc": top[1] / len(labels),
                     "n_labels": len(counts)}
    return out


def compliance_table(rows: List[dict]) -> List[dict]:
    """Mean script ratio, marker-missing rate and error rate per (model, condition)."""
    by = defaultdict(lambda: {"ratios": [], "n": 0, "no_marker": 0, "errors": 0, "outside": 0, "trunc": 0,
                              "hidden": 0})
    for r in rows:
        d = by[(r["model"], r["condition"])]
        d["n"] += 1
        if r.get("error"):
            d["errors"] += 1
            continue
        if not r.get("answer_marker_found"):
            d["no_marker"] += 1
        if not r.get("predicted_in_label_set", True):
            d["outside"] += 1
        if r.get("truncated"):
            d["trunc"] += 1
        if has_hidden_reasoning(r):
            d["hidden"] += 1
        if r.get("reasoning_promoted"):
            continue  # visible text is really hidden reasoning; not a compliance signal
        script = CONDITIONS.get(r["condition"], {}).get("script")
        ratio = script_ratio(reasoning_part(r.get("full_response", "")), script)
        if ratio is not None:
            d["ratios"].append(ratio)
    out = []
    for (model, cond), d in sorted(by.items()):
        out.append({
            "model": model, "condition": cond, "n": d["n"],
            "script_ratio": _mean(d["ratios"]) if d["ratios"] else None,
            "no_marker_rate": d["no_marker"] / d["n"],
            "outside_label_rate": d["outside"] / d["n"],
            "error_rate": d["errors"] / d["n"],
            "truncated_rate": d["trunc"] / d["n"],
            "hidden_reasoning_rate": d["hidden"] / d["n"],
        })
    return out


def paired_vs_baseline(rows: List[dict], baseline: str = "english",
                       drop_truncated: bool = False) -> List[dict]:
    """
    Pooled exact-McNemar comparison of every other condition against the
    baseline on identical (task, idx, run) samples.  With drop_truncated the
    runaway (non-terminating) samples are removed from both sides, so the test
    asks "given the model answered, was it as accurate?" rather than scoring a
    failure to answer as a wrong answer.  Adds a Bonferroni-corrected
    significance flag because many conditions are compared to one baseline.
    """
    by_cond = defaultdict(dict)
    for r in rows:
        if r.get("error") or (drop_truncated and r.get("truncated")):
            continue
        by_cond[r["condition"]][(r["model"], r["task"], r["idx"], r["run_id"])] = bool(r["correct"])
    base = by_cond.get(baseline, {})
    out = []
    others = [c for c in by_cond if c != baseline]
    alpha = 0.05 / max(1, len(others))
    for c in sorted(others):
        shared = set(base) & set(by_cond[c])
        if not shared:
            continue
        b = sum(1 for k in shared if base[k] and not by_cond[c][k])
        cc = sum(1 for k in shared if not base[k] and by_cond[c][k])
        p = mcnemar_exact(b, cc)
        out.append({
            "condition": c, "n_pairs": len(shared),
            "acc_baseline": sum(base[k] for k in shared) / len(shared),
            "acc_condition": sum(by_cond[c][k] for k in shared) / len(shared),
            "delta": (sum(by_cond[c][k] for k in shared) - sum(base[k] for k in shared)) / len(shared),
            "p_mcnemar": p, "significant_bonferroni": p < alpha, "alpha_bonferroni": alpha,
        })
    return out


def runaway_table(rows: List[dict]) -> List[dict]:
    """Runaway (non-terminating) rate and length per condition."""
    tot = defaultdict(int); ra = defaultdict(list)
    for r in rows:
        tot[r["condition"]] += 1
        if r.get("truncated"):
            ra[r["condition"]].append((r.get("output_tokens", 0), r.get("latency_ms", 0) / 60000))
    out = []
    for c in sorted(tot):
        toks = sorted(x[0] for x in ra[c]); mins = sorted(x[1] for x in ra[c])
        out.append({"condition": c, "n": tot[c], "runaways": len(toks), "rate": len(toks) / tot[c],
                    "median_tokens": toks[len(toks) // 2] if toks else 0, "max_tokens": toks[-1] if toks else 0,
                    "median_minutes": mins[len(mins) // 2] if mins else 0.0, "max_minutes": mins[-1] if mins else 0.0})
    return sorted(out, key=lambda d: -d["rate"])


def paired_condition_test(rows: List[dict], cond_a: str, cond_b: str) -> List[dict]:
    """
    Per model (and pooled): sample-level paired comparison of cond_b vs cond_a
    on identical (task, idx, run) keys, with exact McNemar p-value.
    """
    by_model = defaultdict(lambda: {"a": {}, "b": {}})
    for r in rows:
        if r.get("error"):
            continue
        key = (r["model"], r["task"], r["idx"], r["run_id"])
        if r["condition"] == cond_a:
            by_model[r["model"]]["a"][key] = bool(r["correct"])
        elif r["condition"] == cond_b:
            by_model[r["model"]]["b"][key] = bool(r["correct"])

    results = []
    pooled_b = pooled_c = pooled_n = pooled_a_ok = pooled_b_ok = 0
    for model, d in sorted(by_model.items()):
        shared = set(d["a"]) & set(d["b"])
        if not shared:
            continue
        b = sum(1 for k in shared if d["a"][k] and not d["b"][k])
        c = sum(1 for k in shared if not d["a"][k] and d["b"][k])
        a_ok = sum(1 for k in shared if d["a"][k])
        b_ok = sum(1 for k in shared if d["b"][k])
        results.append({
            "model": model, "n_pairs": len(shared),
            f"acc_{cond_a}": a_ok / len(shared), f"acc_{cond_b}": b_ok / len(shared),
            "delta": (b_ok - a_ok) / len(shared),
            "discordant_a_wins": b, "discordant_b_wins": c, "p_mcnemar": mcnemar_exact(b, c),
        })
        pooled_b += b; pooled_c += c; pooled_n += len(shared); pooled_a_ok += a_ok; pooled_b_ok += b_ok
    if pooled_n:
        results.append({
            "model": "ALL (pooled)", "n_pairs": pooled_n,
            f"acc_{cond_a}": pooled_a_ok / pooled_n, f"acc_{cond_b}": pooled_b_ok / pooled_n,
            "delta": (pooled_b_ok - pooled_a_ok) / pooled_n,
            "discordant_a_wins": pooled_b, "discordant_b_wins": pooled_c,
            "p_mcnemar": mcnemar_exact(pooled_b, pooled_c),
        })
    return results



def thinking_pairs(models_present) -> List[Tuple[str, str]]:
    """
    (reasoning-off key, reasoning-on key) pairs: two MODELS entries that hit the
    same model_id with hidden_reasoning "off" and "on".  Derived from config so
    the on/off comparison is never hand-typed here.
    """
    from config import MODELS
    present = [m for m in models_present if m in MODELS]
    pairs = []
    for off in present:
        if MODELS[off].get("hidden_reasoning") != "off":
            continue
        for on in present:
            if on != off and MODELS[on].get("hidden_reasoning") == "on" \
                    and MODELS[on]["model_id"] == MODELS[off]["model_id"]:
                pairs.append((off, on))
    return pairs


def paired_model_test(rows: List[dict], model_a: str, model_b: str,
                      drop_truncated: bool = False) -> List[dict]:
    """
    Per condition: exact-McNemar comparison of model_b against model_a on
    identical (task, idx, run) samples.  Used for the thinking-on (b) vs
    thinking-off (a) comparison of the same underlying model.  Bonferroni over
    the number of conditions compared.
    """
    by_cond = defaultdict(lambda: {"a": {}, "b": {}})
    for r in rows:
        if r.get("error") or (drop_truncated and r.get("truncated")):
            continue
        if r["model"] not in (model_a, model_b):
            continue
        side = "a" if r["model"] == model_a else "b"
        by_cond[r["condition"]][side][(r["task"], r["idx"], r["run_id"])] = bool(r["correct"])
    conds = [c for c, d in by_cond.items() if set(d["a"]) & set(d["b"])]
    alpha = 0.05 / max(1, len(conds))
    out = []
    for c in sorted(conds):
        d = by_cond[c]
        shared = set(d["a"]) & set(d["b"])
        b = sum(1 for k in shared if d["a"][k] and not d["b"][k])
        cc = sum(1 for k in shared if not d["a"][k] and d["b"][k])
        a_ok = sum(d["a"][k] for k in shared)
        b_ok = sum(d["b"][k] for k in shared)
        p = mcnemar_exact(b, cc)
        out.append({
            "condition": c, "model_off": model_a, "model_on": model_b, "n_pairs": len(shared),
            "acc_off": a_ok / len(shared), "acc_on": b_ok / len(shared),
            "delta_on_minus_off": (b_ok - a_ok) / len(shared),
            "discordant_off_wins": b, "discordant_on_wins": cc,
            "p_mcnemar": p, "alpha_bonferroni": alpha, "significant_bonferroni": p < alpha,
        })
    return out

# Which reasoning-language condition a model's training-data origin would
# favour.  Derived from MODELS[*]["origin_country"] at analysis time so adding
# a model needs no edit here.  "France" -> german is the original study's
# stated proxy (there is no French condition); it is a weak hypothesis and is
# labelled as such in the report.
ORIGIN_LANGUAGE = {
    "China": "mandarin", "France": "german",
    # language-native local models (continued pre-training on the home language)
    "Japan": "japanese", "South Korea": "korean", "India": "hindi",
    "Saudi Arabia": "arabic", "United Arab Emirates": "arabic",
}


def origin_hypotheses(models_present) -> List[Tuple[str, str]]:
    out = []
    for m in sorted(models_present):
        lang = ORIGIN_LANGUAGE.get(MODELS.get(m, {}).get("origin_country"))
        if lang:
            out.append((m, lang))
    return out


def origin_advantage(cells: List[dict], baseline: str = "english") -> List[dict]:
    """
    Within-model test: (acc[lang] - acc[english]) for the origin model versus
    the same delta averaged over the other models.  Comparing raw accuracies
    across models would only measure which model is stronger overall.
    """
    acc = defaultdict(list)
    for c in cells:
        acc[(c["model"], c["condition"])].append(c["accuracy"])
    models = sorted({m for m, _ in acc})
    out = []
    for model, lang in origin_hypotheses(models):
        if (model, lang) not in acc or (model, baseline) not in acc:
            continue
        own = _mean(acc[(model, lang)]) - _mean(acc[(model, baseline)])
        others = [_mean(acc[(m, lang)]) - _mean(acc[(m, baseline)])
                  for m in models if m != model and (m, lang) in acc and (m, baseline) in acc]
        if not others:
            continue
        out.append({"model": model, "language": lang, "own_delta_vs_english": own,
                    "others_mean_delta": _mean(others), "relative_advantage": own - _mean(others),
                    "n_other_models": len(others)})
    return out


# ── Report ───────────────────────────────────────────────────────────────

def print_report(rows: List[dict], cells: List[dict]):
    models = sorted({r["model"] for r in rows})
    if "mock" in models:
        print("\n!!! RESULTS CONTAIN THE MOCK MODEL – its numbers are fake by construction !!!")

    print("\n" + "=" * 72 + "\nEXPERIMENT ANALYSIS REPORT\n" + "=" * 72)
    print(f"Samples: {len(rows):,}   Cells (model×condition×task×run): {len(cells)}   Models: {', '.join(models)}")

    print("\n── MAJORITY-CLASS BASELINE PER TASK (what a constant answer scores) ──\n")
    print(f"{'Task':<58} {'n':>5} {'labels':>6} {'majority':>9}")
    for task, b in sorted(majority_baselines(rows).items()):
        print(f"{task:<58} {b['n']:>5} {b['n_labels']:>6} {b['majority_acc']:>9.1%}  ({b['majority_label']})")

    print("\n   Terms: a RUNAWAY is a response that never terminated (finish_reason=length) and so")
    print("   contains no answer; it is recorded as truncated=True. 'raw' scores it as wrong;")
    print("   'excl.runaway' asks how accurate the model was when it did answer.")

    print("\n── CONDITION RANKING (mean accuracy over model×task×run cells) ──\n")
    print("   (accuracies are sample-pooled: sum correct / sum samples, same as the paired tests)\n")
    print(f"{'Rank':<5} {'Condition':<16} {'Family':<15} {'raw':>7} {'n':>6} {'excl.runaway':>13} {'n':>6} {'cell std':>9}")
    for i, d in enumerate(condition_table(cells), 1):
        print(f"{i:<5} {d['condition']:<16} {d['family']:<15} {d['mean']:>7.1%} {d['n']:>6} "
              f"{d['mean_complete']:>13.1%} {d['n_complete']:>6} {d['std']:>9.3f}")

    print("\n── MODEL RANKING ──\n")
    for i, d in enumerate(model_table(cells), 1):
        print(f"{i:<5} {d['model']:<20} {d['mean']:>7.1%} ±{d['std']:.3f}  (cells={d['n_cells']})")

    print("\n── LANGUAGE FAMILY RANKING ──\n")
    for i, d in enumerate(family_table(cells), 1):
        print(f"{i:<5} {d['family']:<16} {d['mean']:>7.1%} ±{d['std']:.3f}")

    print("\n── COMPLIANCE / FAILURE MODES per model×condition ──")
    print("   script_ratio = share of letters in the expected writing system (None for Latin/abstract)")
    print("   hidden = share of samples where the provider returned hidden reasoning (a confound: the")
    print("   visible chain of thought is then a write-up, not the reasoning itself)\n")
    print(f"{'Model':<22} {'Condition':<14} {'n':>5} {'script':>7} {'no-mark':>8} {'off-lbl':>8} {'error':>7} {'runaway':>8} {'hidden':>7}")
    for d in compliance_table(rows):
        sr = f"{d['script_ratio']:.0%}" if d["script_ratio"] is not None else "  n/a"
        print(f"{d['model']:<22} {d['condition']:<14} {d['n']:>5} {sr:>7} {d['no_marker_rate']:>8.1%} "
              f"{d['outside_label_rate']:>8.1%} {d['error_rate']:>7.1%} {d['truncated_rate']:>8.1%} "
              f"{d['hidden_reasoning_rate']:>7.1%}")

    conds = {r["condition"] for r in rows}
    if "english" in conds and len(conds) > 1:
        raw = {d["condition"]: d for d in paired_vs_baseline(rows, "english", drop_truncated=False)}
        exc = {d["condition"]: d for d in paired_vs_baseline(rows, "english", drop_truncated=True)}
        alpha = next(iter(exc.values()))["alpha_bonferroni"] if exc else 0.05
        print("\n── EVERY CONDITION vs ENGLISH (same samples, exact McNemar, pooled over models) ──")
        print(f"   * = significant after Bonferroni correction for {len(exc)} comparisons (alpha={alpha:.4f})\n")
        print(f"{'Condition':<16} {'pairs':>6} {'raw Δ':>8} {'p':>8}   {'excl.runaway Δ':>15} {'p':>8}")
        for c in sorted(exc, key=lambda k: exc[k]["delta"], reverse=True):
            r, e = raw.get(c), exc[c]
            rs = f"{r['delta']:>+8.1%} {r['p_mcnemar']:>7.4f}{'*' if r['significant_bonferroni'] else ' '}" if r else f"{'—':>8} {'—':>8}"
            print(f"{c:<16} {e['n_pairs']:>6} {rs}   {e['delta']:>+15.1%} {e['p_mcnemar']:>7.4f}{'*' if e['significant_bonferroni'] else ' '}")

    for off, on in thinking_pairs({r["model"] for r in rows}):
        raw = {d["condition"]: d for d in paired_model_test(rows, off, on, drop_truncated=False)}
        exc = {d["condition"]: d for d in paired_model_test(rows, off, on, drop_truncated=True)}
        if not exc:
            continue
        alpha = next(iter(exc.values()))["alpha_bonferroni"]
        print(f"\n── HIDDEN REASONING ON vs OFF: {on} vs {off} (same samples, exact McNemar) ──")
        print(f"   Δ = accuracy(on) − accuracy(off).  * = significant after Bonferroni for "
              f"{len(exc)} conditions (alpha={alpha:.4f})\n")
        print(f"{'Condition':<16} {'pairs':>6} {'acc off':>8} {'acc on':>8} {'raw Δ':>8} {'p':>8}   {'excl.runaway Δ':>15} {'p':>8}")
        for c in sorted(exc, key=lambda k: exc[k]["delta_on_minus_off"], reverse=True):
            r, e = raw.get(c), exc[c]
            rs = f"{r['delta_on_minus_off']:>+8.1%} {r['p_mcnemar']:>7.4f}{'*' if r['significant_bonferroni'] else ' '}" if r else f"{'—':>8} {'—':>8}"
            print(f"{c:<16} {e['n_pairs']:>6} {e['acc_off']:>8.1%} {e['acc_on']:>8.1%} {rs}   "
                  f"{e['delta_on_minus_off']:>+15.1%} {e['p_mcnemar']:>7.4f}{'*' if e['significant_bonferroni'] else ' '}")

    print("\n── TRAINING-DATA ORIGIN ADVANTAGE (within-model delta vs english) ──\n")
    oa = origin_advantage(cells)
    if not oa:
        print("  Not testable: needs the origin model AND at least one other model, both run on english")
        print("  and the hypothesis language (hypotheses come from MODELS[*]['origin_country']).")
    for d in oa:
        verdict = "suggestive" if d["relative_advantage"] > 0.02 else "not found"
        print(f"  {d['model']:<16} {d['language']:<9} own Δ={d['own_delta_vs_english']:+.1%}  "
              f"others Δ={d['others_mean_delta']:+.1%}  relative={d['relative_advantage']:+.1%}  → {verdict}")

    print("\n── RUNAWAY (non-terminating) GENERATIONS per condition ──")
    print("   Length is real benchmark data: no cap or deadline is applied to it.\n")
    print(f"{'condition':<16} {'rate':>7} {'med tok':>9} {'max tok':>9} {'med min':>8} {'max min':>8}")
    for d in runaway_table(rows):
        print(f"{d['condition']:<16} {d['rate']:>7.1%} {d['median_tokens']:>9,} {d['max_tokens']:>9,} "
              f"{d['median_minutes']:>8.1f} {d['max_minutes']:>8.1f}")

    print("\n── TOKEN USE per condition (output tokens pooled over answered samples) ──\n")
    tok = defaultdict(lambda: [0, 0])   # [tokens, answered]
    for c in cells:
        tok[c["condition"]][0] += c.get("total_output_tokens", 0)
        tok[c["condition"]][1] += c.get("total", 0) - c.get("errors", 0)
    for cond, (t, n) in sorted(tok.items(), key=lambda kv: (kv[1][0] / kv[1][1]) if kv[1][1] else 0):
        print(f"  {cond:<16} {t / n if n else 0:>8.0f}")

    print("\nCaveats: raw accuracy counts errors and runaways as wrong; 'no-mark' answers were scored from the "
          "last line; script_ratio is a writing-system check, not language identification.\n")


# ── Exports ──────────────────────────────────────────────────────────────

def export_csv(records: List[dict], path: Path):
    if not records:
        return
    fields = []
    for r in records:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f"  CSV exported: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--models", type=str, default=None,
                        help="comma-separated model keys to report on (default: all in results-dir)")
    parser.add_argument("--prompt-version", type=int, default=None,
                        help="only rows made with this prompt wording (rows without the field are version 1)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="where to write the CSVs (default: results-dir); use with --models to keep "
                             "reports for different model sets apart")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_sample_rows(results_dir)
    if args.models:
        keep = {m.strip() for m in args.models.split(",") if m.strip()}
        rows = [r for r in rows if r["model"] in keep]
    if args.prompt_version is not None:
        rows = [r for r in rows if prompt_version_of(r) == args.prompt_version]
    versions = sorted({prompt_version_of(r) for r in rows})
    if len(versions) > 1:
        print(f"WARNING: rows from {len(versions)} prompt versions {versions} are pooled in this report; "
              f"pass --prompt-version N to keep them apart.\n")
    if not rows:
        raise SystemExit(f"No per-sample results in {results_dir}/ (expected <model>__<condition>__runN.jsonl files)")
    cells = build_cell_frame(rows)
    print_report(rows, cells)
    export_csv(cells, out_dir / "results_matrix.csv")
    export_csv(condition_table(cells), out_dir / "condition_summary.csv")
    export_csv(compliance_table(rows), out_dir / "compliance.csv")
    export_csv(paired_vs_baseline(rows, "english", drop_truncated=True), out_dir / "paired_vs_english_excl_runaway.csv")
    export_csv(runaway_table(rows), out_dir / "runaways.csv")
    for off, on in thinking_pairs({r["model"] for r in rows}):
        export_csv(paired_model_test(rows, off, on, drop_truncated=True),
                   out_dir / f"thinking_on_vs_off__{on}.csv")


if __name__ == "__main__":
    main()
