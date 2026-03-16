"""
LegalBench Data Loader
======================
Downloads LegalBench tasks from HuggingFace and formats them for the experiment.
Falls back to a local cache after first download.
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


CACHE_DIR = Path("data/legalbench_cache")


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


def load_task(task_name: str, max_samples: int = 200, split: str = "test") -> List[LegalBenchSample]:
    """
    Load a LegalBench task. Tries HuggingFace first, then local cache.
    Returns up to max_samples samples.
    """
    cache_path = CACHE_DIR / task_name / f"{split}.jsonl"

    # Try cache first
    if cache_path.exists():
        return _load_from_cache(cache_path, task_name, max_samples)

    # Download from HuggingFace
    if not HF_AVAILABLE:
        raise RuntimeError(
            f"Task {task_name} not cached and `datasets` library not installed. "
            f"Run: pip install datasets --break-system-packages"
        )

    print(f"  Downloading {task_name} from HuggingFace...")
    try:
        ds = load_dataset("nguha/legalbench", task_name, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠ Could not load {task_name}/{split}: {e}")
        # Try loading train split as fallback
        try:
            ds = load_dataset("nguha/legalbench", task_name, split="train", trust_remote_code=True)
        except Exception as e2:
            print(f"  ✗ Failed to load {task_name}: {e2}")
            return []

    # Cache locally
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for item in ds:
            f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")

    return _load_from_cache(cache_path, task_name, max_samples)


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
