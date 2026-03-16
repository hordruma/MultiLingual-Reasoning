"""
LegalBench Data Loader
======================
Downloads LegalBench tasks from HuggingFace (preferred) or GitHub (fallback)
and formats them for the experiment.  Falls back to a local cache after first
download.
"""

import json
import os
import csv
import io
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


CACHE_DIR = Path("data/legalbench_cache")
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks"
)


@dataclass
class LegalBenchSample:
    task: str
    idx: int
    input_text: str         # the full question/prompt text
    label: str              # ground truth answer
    input_field: str        # which field(s) were used


def _identify_fields(sample: dict, task: str) -> tuple:
    """
    LegalBench tasks use varying field names. This identifies the input
    and output fields for a given sample.
    """
    # Common output field names
    output_field = None
    for f in ["answer", "label", "output", "ground_truth"]:
        if f in sample:
            output_field = f
            break

    # Common input field names
    input_fields = []
    skip = {output_field, "idx", "index", "id"}
    for f in sample:
        if f not in skip:
            input_fields.append(f)

    return input_fields, output_field


def _download_from_github(task_name: str) -> Optional[List[dict]]:
    """Download a task's data directly from the LegalBench GitHub repo."""
    if not HTTPX_AVAILABLE:
        return None

    for split in ["test", "train"]:
        url = f"{GITHUB_RAW_BASE}/{task_name}/{split}.tsv"
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
            if r.status_code == 200 and r.text.strip():
                reader = csv.DictReader(io.StringIO(r.text), delimiter="\t")
                rows = [dict(row) for row in reader]
                if rows:
                    print(f"  Downloaded {len(rows)} samples from GitHub ({split}.tsv)")
                    return rows
        except Exception:
            continue
    return None


def load_task(task_name: str, max_samples: int = 200, split: str = "test") -> List[LegalBenchSample]:
    """
    Load a LegalBench task. Checks local cache first, then tries HuggingFace,
    then falls back to GitHub raw download.
    Returns up to max_samples samples.
    """
    # Try cache first (check both requested split and train as fallback)
    for try_split in [split, "train"]:
        cache_path = CACHE_DIR / task_name / f"{try_split}.jsonl"
        if cache_path.exists():
            return _load_from_cache(cache_path, task_name, max_samples)

    # Try HuggingFace
    if HF_AVAILABLE:
        print(f"  Downloading {task_name} from HuggingFace...")
        for try_split in [split, "train"]:
            try:
                ds = load_dataset("nguha/legalbench", task_name, split=try_split)
                cache_path = CACHE_DIR / task_name / f"{try_split}.jsonl"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    for item in ds:
                        f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
                return _load_from_cache(cache_path, task_name, max_samples)
            except Exception as e:
                print(f"  ⚠ HuggingFace {task_name}/{try_split}: {e}")

    # Fallback: download from GitHub
    print(f"  Trying GitHub fallback for {task_name}...")
    rows = _download_from_github(task_name)
    if rows:
        cache_path = CACHE_DIR / task_name / "train.jsonl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return _load_from_cache(cache_path, task_name, max_samples)

    print(f"  ✗ Failed to load {task_name} from any source")
    return []


def _load_from_cache(cache_path: Path, task_name: str, max_samples: int) -> List[LegalBenchSample]:
    """Load samples from local JSONL cache."""
    samples = []
    with open(cache_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            raw = json.loads(line)
            input_fields, output_field = _identify_fields(raw, task_name)

            if not output_field or not input_fields:
                continue

            # Build input text from all input fields
            parts = []
            for field in input_fields:
                val = raw.get(field, "")
                if val:
                    parts.append(f"{field}: {val}")
            input_text = "\n".join(parts)

            samples.append(LegalBenchSample(
                task=task_name,
                idx=i,
                input_text=input_text,
                label=str(raw[output_field]).strip(),
                input_field=",".join(input_fields),
            ))

    return samples


def load_all_tasks(task_names: List[str], max_per_task: int = 200) -> Dict[str, List[LegalBenchSample]]:
    """Load multiple tasks and return a dict of task_name -> samples."""
    all_data = {}
    for task in task_names:
        print(f"Loading task: {task}")
        samples = load_task(task, max_samples=max_per_task)
        if samples:
            all_data[task] = samples
            print(f"  ✓ Loaded {len(samples)} samples")
        else:
            print(f"  ✗ No samples loaded")
    return all_data
