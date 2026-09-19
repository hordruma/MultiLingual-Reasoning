"""Ithkuil fidelity of visible reasoning: share of words that are well-formed Ithkuil IV.

Uses the parser from https://github.com/christian-oudard/ithkuil (Go CLI, built
outside this repo; see ITHKUIL_BIN / ITHKUIL_DATA).  For every response in the
chosen conditions each word is classified as

  parsed      – the CLI accepts it (phonotactics + slot grammar), exit status 0
  known_root  – parsed, and a root in its gloss is in the toolkit's lexicon

The parser is permissive: many English words are accidentally well-formed
("the" parses as a referential), so always read the numbers against the
English baseline produced by the same script.  It implements Ithkuil IV only;
Ithkuil III-style words (e.g. with "q") count as not parsed.

    python ithkuil_fidelity.py --results-dir results_prompt_v2 \
        --conditions ithkuil,english --out results_prompt_v2/ithkuil_fidelity.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOLS = Path(os.environ.get("ITHKUIL_TOOLS", Path.home() / ".cache/ithkuil-tools"))
ITHKUIL_BIN = os.environ.get("ITHKUIL_BIN", str(TOOLS / "bin/ithkuil"))
ITHKUIL_DATA = os.environ.get("ITHKUIL_DATA", str(TOOLS / "data.db"))
ROOTS_TSV = os.environ.get("ITHKUIL_ROOTS", str(TOOLS / "ithkuil/data/roots.tsv"))

WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
ANSWER_LINE_RE = re.compile(r"^\W*(?:ANSWER|Answer|A)\s*[:：].*$", re.MULTILINE)
ROOT_SEGMENT_RE = re.compile(r"^[^\W\dA-Z_]+$", re.UNICODE)


def load_roots(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)
        return {r[0].strip() for r in rows if r and r[0].strip()}


def words_of(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(ANSWER_LINE_RE.sub("", text or ""))]


def classify(word: str, roots: set[str]) -> tuple[bool, bool]:
    """(parsed, known_root) for one word."""
    proc = subprocess.run([ITHKUIL_BIN, "--data", ITHKUIL_DATA, "parse", "--short", "--color", "never", word],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return False, False
    segments = re.split(r"[-./]", proc.stdout.strip().splitlines()[0])
    return True, any(ROOT_SEGMENT_RE.match(s) and s in roots for s in segments)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="results_prompt_v2")
    ap.add_argument("--conditions", default="ithkuil,english")
    ap.add_argument("--models", default=None, help="comma-separated; default all found")
    ap.add_argument("--out", default=None, help="per-cell CSV (default <results-dir>/ithkuil_fidelity.csv)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    conditions = args.conditions.split(",")
    models = args.models.split(",") if args.models else None
    roots = load_roots(ROOTS_TSV)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*__run*.jsonl"))):
        model, condition = os.path.basename(path).split("__")[:2]
        if condition not in conditions or (models and model not in models) or model == "mock":
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("error"):
                    rows.append((model, condition, bool(r.get("truncated")), words_of(r.get("full_response"))))

    vocab = sorted({w for *_, ws in rows for w in ws})
    print(f"{len(rows)} responses, {len(vocab)} distinct words to parse")
    with ThreadPoolExecutor(args.workers) as pool:
        verdict = dict(zip(vocab, pool.map(lambda w: classify(w, roots), vocab)))

    cells = defaultdict(lambda: defaultdict(int))
    types = defaultdict(set)
    for model, condition, truncated, ws in rows:
        for key in ((model, condition, "all"), (model, condition, "runaway" if truncated else "terminated")):
            c = cells[key]
            c["responses"] += 1
            c["words"] += len(ws)
            c["parsed"] += sum(verdict[w][0] for w in ws)
            c["known_root"] += sum(verdict[w][1] for w in ws)
            types[key].update(ws)

    out = args.out or os.path.join(args.results_dir, "ithkuil_fidelity.csv")
    header = ["model", "condition", "subset", "responses", "words", "distinct_words",
              "pct_words_parsed", "pct_words_known_root", "pct_distinct_parsed", "pct_distinct_known_root"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        print("  ".join(header))
        for key in sorted(cells):
            c, t = cells[key], types[key]
            n, nt = max(c["words"], 1), max(len(t), 1)
            line = [*key, c["responses"], c["words"], len(t),
                    round(100 * c["parsed"] / n, 1), round(100 * c["known_root"] / n, 1),
                    round(100 * sum(verdict[w][0] for w in t) / nt, 1),
                    round(100 * sum(verdict[w][1] for w in t) / nt, 1)]
            wr.writerow(line)
            print("  ".join(str(x) for x in line))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
