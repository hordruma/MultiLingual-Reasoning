# LegalBench Chain-of-Thought Language Experiment

Does the language a model reasons in change its accuracy on legal
classification tasks? This repo runs a factorial of reasoning-language
conditions × models × LegalBench tasks and analyses the results.

State of the project: **prepared, not yet run**. The code has been audited
and reworked (see [ASSESSMENT.md](ASSESSMENT.md)), tested offline, and is
set up to run against six cheap cloud models. No real results exist yet.

## Design

| Factor | Levels |
|---|---|
| Reasoning condition | **19**: 14 natural languages (9 families) + formal logic, pseudocode, emergent notation + wildcard + no-CoT control |
| Models | default 6 cheap models (below); any entry in `config.MODELS` |
| Tasks | 9 closed-label LegalBench tasks (7 Yes/No, one 5-class, one 9-class) |
| Samples | up to 200 per task, seeded random subset, identical across cells |
| Runs | `--runs N` (config default 3; use 1 for a first pass) |
| Decoding | temperature 0, 2048 output tokens |

### Conditions

| Condition | Family | Condition | Family |
|---|---|---|---|
| english | Indo-European | turkish | Turkic |
| german | Indo-European | finnish | Uralic |
| russian | Indo-European | hungarian | Uralic |
| hindi | Indo-European | indonesian | Austronesian |
| mandarin | Sino-Tibetan | vietnamese | Austroasiatic |
| arabic | Afroasiatic | formal_logic | Abstract |
| hebrew | Afroasiatic | pseudocode | Abstract |
| japanese | Japonic | emergent | Abstract |
| korean | Koreanic | wildcard | Wildcard |
| | | no_cot | Control |

Every condition asks for the final answer in English on a line of the form
`ANSWER: <label>`.

### Default models (cheap tier)

| Key | Model id | Route | Env var |
|---|---|---|---|
| gpt-4o-mini | gpt-4o-mini | OpenAI | `OPENAI_API_KEY` |
| claude-haiku | claude-haiku-4-5-20251001 | Anthropic | `ANTHROPIC_API_KEY` |
| gemini-flash-lite | gemini-2.5-flash-lite | Google OpenAI-compatible endpoint | `GEMINI_API_KEY` |
| deepseek-chat | deepseek-chat | DeepSeek | `DEEPSEEK_API_KEY` |
| mistral-small | mistral-small-latest | Mistral | `MISTRAL_API_KEY` |
| qwen-plus | qwen-plus | Alibaba DashScope (intl) | `QWEN_API_KEY` |

Frontier equivalents (`claude-sonnet`, `gpt-4o`, `gemini-2.5-flash`,
`mistral-large`, `qwen-max`) stay in `config.py` and can be requested with
`--models`. `openrouter` routes any OpenRouter model id through one key.
Model ids and the prices used by `--estimate` are placeholders; check them
against the providers before a paid run. Any OpenAI-compatible host (Azure
AI Foundry, vLLM, Ollama) works by pointing a model's `*_BASE_URL` at it.

### Tasks

| Task | Labels | ~test size |
|---|---|---|
| hearsay | Yes/No | 94 |
| personal_jurisdiction | Yes/No | 50 |
| contract_nli_explicit_identification | Yes/No | 109 |
| contract_nli_inclusion_of_verbally_conveyed_information | Yes/No | 139 |
| proa | Yes/No | 95 |
| abercrombie | generic/descriptive/suggestive/arbitrary/fanciful | 95 |
| supply_chain_disclosure_best_practice_verification | Yes/No | 379 |
| unfair_tos | 9 clause types | 3584 (capped at 200) |
| learned_hands_benefits | Yes/No | 66 |

Each task is presented with its official LegalBench `base_prompt.txt`
(definition + few-shot examples) and the allowed label list. Only the
`text` field is shown to the model; metadata columns that leak the answer
are excluded.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or --break-system-packages
pip install -r requirements.txt
cp .env.template .env      # fill in the keys for the models you will use
python -m pytest -q        # 25 offline tests, no network needed
```

Data comes from two places on first use and is cached under `data/`:
the test split of each task from the HuggingFace hub files of
`nguha/legalbench`, and `base_prompt.txt` from the LegalBench GitHub repo.
If the hub download fails, the loader prints the exact file to download by
hand and where to put it.

## Running

```bash
python run_experiment.py --list                     # models, conditions, tasks, env vars
python run_experiment.py --smoke-test               # one tiny call per default model
python run_experiment.py --smoke-test --models gpt-4o-mini,deepseek-chat
python run_experiment.py --estimate --runs 1        # downloads data, prints a rough cost table
python run_experiment.py --dry-run --pilot          # show the matrix, no calls

# Cheap first pass: 8 pilot conditions, 1 run, 50 samples/task
python run_experiment.py --pilot --runs 1 --max-samples 50

# Two models, three conditions, full samples
python run_experiment.py --models gpt-4o-mini,deepseek-chat --conditions english,mandarin,wildcard --runs 1

# Everything for the default models
python run_experiment.py --runs 1

# Offline pipeline check (fake answers, never a finding)
python run_experiment.py --models mock --runs 1 --max-samples 5
```

Results are appended per sample to `results/<model>__<condition>__run<N>.jsonl`.
Rerunning the same command resumes: completed samples are skipped, errored
ones are retried. `--fresh` discards a cell's file first. `--concurrency`
(default 5) is the number of in-flight requests; lower it if a provider
rate-limits you.

## Analysis

```bash
python analyze.py                 # report + CSVs from results/*.jsonl
jupyter lab legalbench_analysis.ipynb
```

The report prints, in order: majority-class baseline per task, condition /
model / family rankings, a compliance table per model × condition (share of
letters in the expected script, missing `ANSWER:` marker, prediction outside
the label set, API errors, truncation), exact McNemar paired tests of
wildcard / no_cot / mandarin / pseudocode against English on identical
samples, a within-model origin-advantage check, and token use.

The notebook reads the same JSONL files. It stops if `results/` is empty; it
does not generate placeholder data.

## Output files

```
results/
├── experiment_summary.json     # per-cell summaries incl. per-task accuracy, errors, truncation
├── <model>__<condition>__run<N>.jsonl   # one record per sample with the full response
├── results_matrix.csv          # (model, condition, task, run) cells
├── condition_summary.csv
└── compliance.csv
figures/                        # written by the notebook
```

## Cost

Use `--estimate`; it builds the real prompts and multiplies by the prices in
`config.py`. For orientation, the default six cheap models × 19 conditions ×
~1,100 samples × 1 run is roughly 125k calls and, at 2025 list prices,
somewhere around 100–150 USD, more than half of it Claude Haiku. Prompts are
long (few-shot examples plus some multi-paragraph disclosures), so input
tokens matter as much as output. The frontier set is about ten times that
per run. `--pilot --runs 1 --max-samples 50` is a few dollars.
