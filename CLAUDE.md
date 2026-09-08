# CLAUDE.md — LegalBench CoT Language Experiment

## Project Overview
Research project testing whether the language used for chain-of-thought
reasoning affects LLM accuracy on legal classification tasks. 19 reasoning
conditions (14 natural languages, 3 abstract notations, wildcard, no-CoT
control) × configurable models (default: 6 cheap cloud models, Sept 2026 lineup;
local Ollama/LM Studio supported) × 9 closed-label LegalBench tasks. Audit history and known caveats are in
ASSESSMENT.md; read it before changing scoring or prompts.

## Repository Layout
```
config.py          — MODELS (env vars, thinking toggles, prices), CONDITIONS, LEGALBENCH_TASKS (label sets), parameters
data_loader.py     — LegalBench test split (HF hub files) + official base_prompt.txt (GitHub), local cache
providers.py       — Async adapters: anthropic (native), openai_compat (any /chat/completions host), mock
run_experiment.py  — Resumable runner (CLI): --list, --smoke-test, --estimate, --dry-run, --pilot
analyze.py         — Report + CSVs from per-sample JSONL; stats helpers used by the notebook
legalbench_analysis.ipynb — Charts; reads results/*.jsonl via analyze.py; no synthetic data
tests/test_pipeline.py    — Offline tests (pytest), no network or keys needed
ASSESSMENT.md      — Audit findings, what was fixed, what remains unverified
.env.template      — API key template (never commit .env)
data/              — Cache of test.jsonl + base_prompt.txt per task (gitignored)
results/           — Experiment outputs (gitignored)
figures/           — Notebook output (gitignored)
```

## Quick Start
```bash
pip install -r requirements.txt
cp .env.template .env               # fill in keys for the models you will use
python -m pytest -q                 # 38 offline tests
python run_experiment.py --list
python run_experiment.py --smoke-test
python run_experiment.py --estimate --runs 1
python run_experiment.py --pilot --runs 1 --max-samples 50
python analyze.py
```

## Key Conventions
- Temperature 0.0 where accepted (GPT-5.x omits it), 4096 max output tokens; `truncated` is recorded per sample.
- Hidden thinking is disabled per model via `request_overrides`; any hidden reasoning that
  still comes back is stored in `hidden_reasoning`, never merged into `full_response`.
- Samples per task: seeded random subset (`SAMPLE_SEED`), same for every cell.
- Prompt = official LegalBench base_prompt (definition + few-shot) with
  `{{text}}` substituted, trailing `A:` cue removed, plus "Answer with exactly
  one of: <labels>". System prompt carries the condition instruction and the
  `ANSWER: <label>` format. Only the `text` field reaches the model.
- Scoring: last `ANSWER:`-style line (also `Answer:`, `**ANSWER:**`, `ANSWER：`,
  `A:`), cleaned, mapped onto the task label set, exact match. Records
  `answer_marker_found`, `predicted_in_label_set`, `truncated`, `error`.
- Accuracy in summaries counts errors as wrong; `accuracy_answered` excludes them.
- Results are appended per sample; rerunning resumes and retries errored rows.
- The `mock` model is a pipeline test double. Never report its numbers.

## Commands
- `python run_experiment.py --list` — models (with env vars), conditions, tasks
- `python run_experiment.py --smoke-test [--models a,b]` — one tiny call per model
- `python run_experiment.py --estimate --runs N` — rough cost from real prompts
- `python run_experiment.py --dry-run` — matrix only
- `python run_experiment.py --models m1,m2 --conditions c1,c2 --tasks t1 --runs 1 --max-samples 50`
- `python run_experiment.py --models mock --runs 1` — offline check
- `python analyze.py --results-dir results/` — report + CSVs
- `jupyter lab legalbench_analysis.ipynb`

## Testing
`python -m pytest -q tests`. Tests cover extraction, normalisation, scoring,
prompt rendering, leak-field exclusion, seeded sampling, resume logic,
provider config resolution and the analysis statistics. Network paths (HF
hub download, real providers) are exercised by `--smoke-test` and a first
`--max-samples 5` run, not by the test suite.

## Style
- Python 3.10+, type hints where practical
- Async/await for all LLM calls; one shared httpx client
- dataclasses for structured data (LegalBenchSample, LLMResponse)
- Flat module structure, no deep package hierarchy
- Never add synthetic or placeholder results to results/ or the notebook
