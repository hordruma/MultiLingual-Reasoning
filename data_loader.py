
"""
LegalBench Data Loader
======================
Loads the *test* split of each LegalBench task plus the task's official
base prompt (task definition + few-shot examples).

Sources, in order:
  1. local cache            data/legalbench_cache/<task>/test.jsonl
  2. HuggingFace hub file   https://huggingface.co/datasets/nguha/legalbench/resolve/main/data/<task>/test.tsv
  3. `datasets` library     load_dataset("nguha/legalbench", <task>, split="test")  (optional dependency)

The LegalBench GitHub repo only carries each task's few-shot *train* split
(4–160 rows) and base_prompt.txt.  Train rows are the examples embedded in
the prompt; they are never used as the evaluation set.  If the test split
cannot be fetched, the task is skipped loudly rather than silently replaced.
"""

import csv
import io
import sys
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import httpx

CACHE_DIR = Path("data/legalbench_cache")
HF_RESOLVE_BASE = "https://huggingface.co/datasets/nguha/legalbench/resolve/main/data"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks"

# Columns that are never model input.  `slice` (hearsay, personal_jurisdiction)
# names the legal sub-category of the fact pattern and leaks the answer.
NON_INPUT_FIELDS = {"answer", "label", "index", "idx", "id", "slice", "document_name", "doctrine"}

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # long disclosures exceed the 128 KiB default

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_TRAILING_ANSWER_CUE = re.compile(r"(?:\n|^)\s*(?:A|Answer|Label|Output)\s*:\s*$", re.IGNORECASE)


@dataclass
class LegalBenchSample:
    task: str
    idx: int                # row index in the original test split
    text: str               # raw input text (the {{text}} field)
    label: str              # ground-truth label
    prompt: str             # official base prompt with fields substituted


# ── Fetch helpers ─────────────────────────────────────────────────────────

def _get(url: str, timeout: float = 60) -> Optional[str]:
    try:
        r = httpx.get(url, follow_redirects=True, timeout=timeout)
        if r.status_code == 200 and r.text.strip():
            return r.text
        print(f"  ✗ {url} → HTTP {r.status_code}")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ {url} → {type(e).__name__}: {e}")
    return None


def _parse_tsv(text: str) -> List[dict]:
    reader = csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    rows = []
    for row in reader:
        if row and any((v or "").strip() for v in row.values()):
            rows.append({k: (v if v is not None else "") for k, v in row.items() if k is not None})
    return rows


def _looks_like_task_rows(rows: List[dict]) -> bool:
    """Guard against caching an HTML error/consent page that happened to parse."""
    if not rows:
        return False
    keys = set(rows[0].keys())
    return "text" in keys and ("answer" in keys or "label" in keys)


def _write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Test split ────────────────────────────────────────────────────────────

def fetch_test_rows(task_name: str) -> Optional[List[dict]]:
    """Return the raw rows of the task's test split, caching on success."""
    cache_path = CACHE_DIR / task_name / "test.jsonl"
    if cache_path.exists():
        return _read_jsonl(cache_path)

    # 2. HuggingFace hub raw file
    print(f"  Downloading {task_name}/test.tsv from HuggingFace hub...")
    text = _get(f"{HF_RESOLVE_BASE}/{task_name}/test.tsv")
    if text:
        rows = _parse_tsv(text)
        if _looks_like_task_rows(rows):
            _write_jsonl(cache_path, rows)
            return rows
        print(f"  ✗ {task_name}/test.tsv did not parse as a LegalBench table (columns: "
              f"{list(rows[0].keys())[:4] if rows else 'none'}); not caching")

    # 3. datasets library (optional)
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        load_dataset = None
    if load_dataset is not None:
        print(f"  Trying `datasets` library for {task_name}...")
        try:
            try:
                ds = load_dataset("nguha/legalbench", task_name, split="test", trust_remote_code=True)
            except TypeError:
                ds = load_dataset("nguha/legalbench", task_name, split="test")
            rows = [dict(item) for item in ds]
            if _looks_like_task_rows(rows):
                _write_jsonl(cache_path, rows)
                return rows
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ datasets: {type(e).__name__}: {str(e)[:200]}")

    print(
        f"  ✗ Could not fetch the test split for {task_name}.\n"
        f"    Manual fix: download test.tsv from\n"
        f"    https://huggingface.co/datasets/nguha/legalbench/tree/main/data/{task_name}\n"
        f"    and place it at data/legalbench_cache/{task_name}/test.tsv"
    )
    manual = CACHE_DIR / task_name / "test.tsv"
    if manual.exists():
        rows = _parse_tsv(manual.read_text(encoding="utf-8"))
        if _looks_like_task_rows(rows):
            _write_jsonl(cache_path, rows)
            return rows
    return None


# ── Base prompt ───────────────────────────────────────────────────────────

def fetch_base_prompt(task_name: str) -> Optional[str]:
    """The official LegalBench prompt: task definition + few-shot examples."""
    cache_path = CACHE_DIR / task_name / "base_prompt.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    print(f"  Downloading {task_name}/base_prompt.txt from GitHub...")
    text = _get(f"{GITHUB_RAW_BASE}/{task_name}/base_prompt.txt")
    if text is None:
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def render_prompt(base_prompt: str, row: dict) -> str:
    """
    Substitute {{field}} placeholders and drop the trailing "A:" cue so the
    model reasons first instead of completing the label immediately.
    """
    def _sub(m):
        field = m.group(1)
        return str(row.get(field, ""))
    rendered = _PLACEHOLDER.sub(_sub, base_prompt)
    rendered = _TRAILING_ANSWER_CUE.sub("", rendered.rstrip())
    return rendered.strip()


# ── Public API ────────────────────────────────────────────────────────────

def load_task(task_name: str, task_cfg: dict, max_samples: int = 200,
              seed: int = 0) -> List[LegalBenchSample]:
    """
    Load up to `max_samples` test rows for a task as a seeded random subset.
    The same seed gives every model and condition the identical subset.
    """
    rows = fetch_test_rows(task_name)
    if not rows:
        return []

    base_prompt = fetch_base_prompt(task_name)
    if base_prompt is None:
        print(f"  ✗ No base prompt for {task_name}; skipping task (the model would not know the task)")
        return []

    label_field = "answer" if "answer" in rows[0] else "label"
    if label_field not in rows[0]:
        print(f"  ✗ {task_name}: no answer/label column in {list(rows[0].keys())}")
        return []

    n = len(rows)
    if n > max_samples:
        chosen = sorted(random.Random(seed).sample(range(n), max_samples))
    else:
        chosen = list(range(n))

    labels_cfg = {l.lower() for l in task_cfg.get("labels", [])}
    samples, bad_labels = [], 0
    for i in chosen:
        row = rows[i]
        label = str(row.get(label_field, "")).strip()
        if labels_cfg and label.lower() not in labels_cfg:
            bad_labels += 1
        samples.append(LegalBenchSample(
            task=task_name,
            idx=i,
            text=str(row.get("text", "")),
            label=label,
            prompt=render_prompt(base_prompt, {k: v for k, v in row.items() if k not in NON_INPUT_FIELDS or k == "text"}),
        ))
    if bad_labels:
        print(f"  ⚠ {task_name}: {bad_labels}/{len(samples)} labels are outside the configured label set "
              f"{sorted(labels_cfg)} – check config.LEGALBENCH_TASKS")
    return samples


def load_all_tasks(task_cfgs: Dict[str, dict], max_per_task: int = 200,
                   seed: int = 0) -> Dict[str, List[LegalBenchSample]]:
    """Load multiple tasks and return {task_name: samples}. Missing tasks are skipped loudly."""
    all_data = {}
    for task, cfg in task_cfgs.items():
        print(f"Loading task: {task}")
        samples = load_task(task, cfg, max_samples=max_per_task, seed=seed)
        if samples:
            all_data[task] = samples
            print(f"  ✓ {len(samples)} samples")
        else:
            print(f"  ✗ SKIPPED {task} – no samples loaded")
    return all_data
