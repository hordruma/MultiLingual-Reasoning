# CLAUDE.md — LegalBench CoT Language Experiment

## Project Overview
Research project investigating whether the language used for chain-of-thought reasoning affects LLM accuracy on legal tasks. Tests 21 reasoning conditions (16 natural languages, 3 abstract formats, wildcard, no-CoT control) across 6 models on 10 LegalBench tasks.

## Repository Layout
```
config.py          — Models, conditions, tasks, experiment parameters
data_loader.py     — HuggingFace LegalBench downloader + local JSONL cache
providers.py       — Async LLM API adapters (Anthropic, Azure, Gemini, Qwen)
run_experiment.py  — Full factorial experiment runner (CLI)
analyze.py         — Post-run aggregation, stats, CSV export (CLI)
legalbench_analysis.ipynb — Interactive analysis notebook with visualizations
requirements.txt   — Python dependencies
.env.template      — API key template (never commit .env)
data/              — Auto-created LegalBench cache (gitignored)
results/           — Auto-created experiment outputs (gitignored)
```

## Quick Start
```bash
pip install -r requirements.txt --break-system-packages
cp .env.template .env  # fill in API keys
python run_experiment.py --dry-run  # verify matrix
python run_experiment.py --pilot    # quick 7-condition run
python analyze.py                   # CLI report + CSV
jupyter lab legalbench_analysis.ipynb  # interactive analysis
```

## Key Conventions
- **Temperature 0.0** everywhere for deterministic outputs
- **200 samples per task** max, 3 runs per cell for variance
- Answer extraction: looks for `ANSWER: ` prefix, falls back to last line
- All conditions require final answer in English
- Data caching: first HuggingFace download writes to `data/legalbench_cache/`; subsequent loads use cache

## Commands
- `python run_experiment.py --dry-run` — show experiment matrix, no API calls
- `python run_experiment.py --models claude-sonnet --conditions english,mandarin --runs 1` — subset run
- `python run_experiment.py --pilot` — 7-condition pilot
- `python analyze.py --results-dir results/` — generate text report + CSV
- `jupyter lab legalbench_analysis.ipynb` — run the analysis notebook

## Testing
No test suite yet. Validate with `--dry-run` and spot-check JSONL outputs.

## Style
- Python 3.10+, type hints where practical
- Async/await for all LLM calls
- dataclasses for structured data (LegalBenchSample, LLMResponse)
- Flat module structure, no deep package hierarchy
