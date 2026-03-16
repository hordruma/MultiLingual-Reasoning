"""
Results Analyzer
================
Processes experiment results into summary tables, statistical analysis,
and CSV exports for further visualization.

Usage:
    python analyze.py                  # analyze latest results
    python analyze.py --results-dir results/  # specify directory
"""

import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import statistics


def load_summaries(results_dir: Path) -> List[Dict]:
    """Load the experiment summary file."""
    summary_file = results_dir / "experiment_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"No summary file at {summary_file}")
    with open(summary_file) as f:
        data = json.load(f)
    return data["results"]


def load_detailed_results(results_dir: Path) -> List[Dict]:
    """Load all detailed JSONL result files."""
    results = []
    for f in results_dir.glob("*.jsonl"):
        with open(f) as fh:
            for line in fh:
                results.append(json.loads(line))
    return results


def compute_aggregates(summaries: List[Dict]) -> Dict:
    """Compute key aggregations across the matrix."""
    # Group by (model, condition)
    by_cell = defaultdict(list)
    # Group by condition (across models)
    by_condition = defaultdict(list)
    # Group by model (across conditions)
    by_model = defaultdict(list)
    # Group by family
    by_family = defaultdict(list)
    # Group by (model_origin, condition)
    by_origin_condition = defaultdict(list)

    for s in summaries:
        acc = s["accuracy"]
        by_cell[(s["model"], s["condition"])].append(acc)
        by_condition[s["condition"]].append(acc)
        by_model[s["model"]].append(acc)
        by_family[s["condition_family"]].append(acc)
        by_origin_condition[(s["model_origin"], s["condition"])].append(acc)

    def _stats(values):
        if len(values) < 2:
            return {"mean": values[0] if values else 0, "std": 0, "n": len(values)}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "n": len(values),
        }

    return {
        "by_cell": {str(k): _stats(v) for k, v in sorted(by_cell.items())},
        "by_condition": {k: _stats(v) for k, v in sorted(by_condition.items(), key=lambda x: -statistics.mean(x[1]))},
        "by_model": {k: _stats(v) for k, v in sorted(by_model.items(), key=lambda x: -statistics.mean(x[1]))},
        "by_family": {k: _stats(v) for k, v in sorted(by_family.items(), key=lambda x: -statistics.mean(x[1]))},
        "by_origin_condition": {str(k): _stats(v) for k, v in by_origin_condition.items()},
    }


def compute_token_efficiency(summaries: List[Dict]) -> Dict:
    """Compute token efficiency metrics per condition."""
    by_condition = defaultdict(lambda: {"accuracies": [], "output_tokens": []})
    for s in summaries:
        by_condition[s["condition"]]["accuracies"].append(s["accuracy"])
        by_condition[s["condition"]]["output_tokens"].append(s["avg_output_tokens"])

    result = {}
    for cond, data in by_condition.items():
        avg_acc = statistics.mean(data["accuracies"])
        avg_tokens = statistics.mean(data["output_tokens"])
        # Efficiency = accuracy per 1000 tokens of reasoning
        efficiency = (avg_acc / avg_tokens * 1000) if avg_tokens > 0 else 0
        result[cond] = {
            "avg_accuracy": avg_acc,
            "avg_output_tokens": avg_tokens,
            "efficiency": efficiency,
        }

    return dict(sorted(result.items(), key=lambda x: -x[1]["efficiency"]))


def check_origin_advantage(summaries: List[Dict]) -> List[Dict]:
    """
    Key analysis: does a model perform better when reasoning in a language
    associated with its training data origin?
    E.g., does Mistral (France) reason better in French/German than Claude (USA)?
    """
    # Define expected advantages
    expected_advantages = {
        "deepseek-v3": ["mandarin"],
        "qwen-max": ["mandarin"],
        "mistral-large": ["german"],  # closest we have to French
    }

    findings = []
    by_cell = defaultdict(list)
    for s in summaries:
        by_cell[(s["model"], s["condition"])].append(s["accuracy"])

    for model, advantage_langs in expected_advantages.items():
        for lang in advantage_langs:
            # This model's accuracy in this language
            model_acc = by_cell.get((model, lang), [])
            if not model_acc:
                continue
            model_avg = statistics.mean(model_acc)

            # Other models' accuracy in this language
            other_accs = []
            for (m, c), accs in by_cell.items():
                if c == lang and m != model:
                    other_accs.extend(accs)

            if not other_accs:
                continue
            others_avg = statistics.mean(other_accs)

            findings.append({
                "model": model,
                "language": lang,
                "model_accuracy": model_avg,
                "others_accuracy": others_avg,
                "advantage": model_avg - others_avg,
                "hypothesis": f"{model} has training data advantage in {lang}",
            })

    return findings


def export_csv(summaries: List[Dict], output_path: Path):
    """Export full results matrix as CSV for external visualization."""
    if not summaries:
        return

    fieldnames = list(summaries[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"  CSV exported: {output_path}")


def print_report(summaries: List[Dict]):
    """Print a human-readable analysis report."""
    agg = compute_aggregates(summaries)
    efficiency = compute_token_efficiency(summaries)
    origin = check_origin_advantage(summaries)

    print("\n" + "=" * 70)
    print("EXPERIMENT ANALYSIS REPORT")
    print("=" * 70)

    # 1. Overall condition ranking
    print("\n── CONDITION RANKING (accuracy, all models averaged) ──\n")
    print(f"{'Rank':<5} {'Condition':<20} {'Mean Acc':<10} {'Std':<8} {'N'}")
    print("-" * 55)
    for i, (cond, stats) in enumerate(agg["by_condition"].items(), 1):
        print(f"{i:<5} {cond:<20} {stats['mean']:<10.1%} {stats['std']:<8.3f} {stats['n']}")

    # 2. Model ranking
    print("\n── MODEL RANKING (accuracy, all conditions averaged) ──\n")
    print(f"{'Rank':<5} {'Model':<20} {'Mean Acc':<10} {'Std':<8} {'N'}")
    print("-" * 55)
    for i, (model, stats) in enumerate(agg["by_model"].items(), 1):
        print(f"{i:<5} {model:<20} {stats['mean']:<10.1%} {stats['std']:<8.3f} {stats['n']}")

    # 3. Language family ranking
    print("\n── LANGUAGE FAMILY RANKING ──\n")
    print(f"{'Rank':<5} {'Family':<20} {'Mean Acc':<10} {'Std':<8}")
    print("-" * 45)
    for i, (family, stats) in enumerate(agg["by_family"].items(), 1):
        print(f"{i:<5} {family:<20} {stats['mean']:<10.1%} {stats['std']:<8.3f}")

    # 4. Token efficiency
    print("\n── TOKEN EFFICIENCY (accuracy per 1K reasoning tokens) ──\n")
    print(f"{'Rank':<5} {'Condition':<20} {'Accuracy':<10} {'Avg Tokens':<12} {'Efficiency'}")
    print("-" * 62)
    for i, (cond, data) in enumerate(efficiency.items(), 1):
        print(f"{i:<5} {cond:<20} {data['avg_accuracy']:<10.1%} {data['avg_output_tokens']:<12.0f} {data['efficiency']:<.4f}")

    # 5. Training data origin advantage
    print("\n── TRAINING DATA ORIGIN ADVANTAGE ──\n")
    if origin:
        for f in origin:
            direction = "✓ CONFIRMED" if f["advantage"] > 0.02 else "✗ NOT FOUND"
            print(f"  {direction}: {f['hypothesis']}")
            print(f"    {f['model']}: {f['model_accuracy']:.1%}  |  others: {f['others_accuracy']:.1%}  |  Δ = {f['advantage']:+.1%}")
    else:
        print("  No origin advantage data available.")

    # 6. Key findings
    print("\n── KEY FINDINGS ──\n")
    conditions_ranked = list(agg["by_condition"].items())
    if conditions_ranked:
        best = conditions_ranked[0]
        worst = conditions_ranked[-1]
        english_stats = agg["by_condition"].get("english", {"mean": 0})
        wildcard_stats = agg["by_condition"].get("wildcard", {"mean": 0})

        print(f"  Best condition:  {best[0]} ({best[1]['mean']:.1%})")
        print(f"  Worst condition: {worst[0]} ({worst[1]['mean']:.1%})")
        print(f"  English baseline: {english_stats['mean']:.1%}")
        print(f"  Wildcard (unconstrained): {wildcard_stats['mean']:.1%}")

        if wildcard_stats["mean"] > english_stats["mean"]:
            print(f"\n  ★ WILDCARD OUTPERFORMS ENGLISH by {wildcard_stats['mean'] - english_stats['mean']:+.1%}")
            print(f"    → Supports hypothesis that unconstrained polyglot reasoning is superior")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summaries = load_summaries(results_dir)

    print_report(summaries)
    export_csv(summaries, results_dir / "results_matrix.csv")


if __name__ == "__main__":
    main()
