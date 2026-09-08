# LegalBench Chain-of-Thought Language Experiment

Does the language a model reasons in change its accuracy on legal
classification tasks? This repo runs a factorial of reasoning-language
conditions × models × LegalBench tasks and analyses the results.

State of the project: **audited, and validated against a live API**. The code
was reworked (see [ASSESSMENT.md](ASSESSMENT.md)) and a first real run
(GLM-5.3 via TokenRouter, 58 samples, zero errors) confirmed the data
download, scoring, resume, analysis and notebook all work end to end. No
full-scale results exist yet.

## Design

| Factor | Levels |
|---|---|
| Reasoning condition | **19**: 14 natural languages (9 families) + formal logic, pseudocode, emergent notation + wildcard + no-CoT control |
| Models | default 6 cheap cloud models (below); any entry in `config.MODELS`, including local ones |
| Tasks | 9 closed-label LegalBench tasks (7 Yes/No, one 5-class, one 9-class) |
| Samples | up to 200 per task, seeded random subset, identical across cells |
| Runs | `--runs N` (config default 3; use 1 for a first pass) |
| Decoding | temperature 0 where the API allows it, **no output cap** (models run to natural stop), hidden "thinking" off where the API allows it |

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

### Default models (cheap tier, September 2026)

| Key | Model id | Route | Env var | Thinking | $/1M in / out |
|---|---|---|---|---|---|
| gpt-5.6-luna | gpt-5.6-luna | OpenAI | `OPENAI_API_KEY` | off (`reasoning_effort: none`) | 0.20 / 1.20 |
| gpt-5.6-luna-think | gpt-5.6-luna | OpenAI | `OPENAI_API_KEY` | on (`reasoning_effort: low`) — paired counterpart for the thinking-on/off test | 0.20 / 1.20 |
| gemini-3.1-flash-lite | gemini-3.1-flash-lite | Google OpenAI-compatible layer | `GEMINI_API_KEY` | lowest setting only | 0.25 / 1.50 |
| deepseek-v4-flash | deepseek-v4-flash | DeepSeek | `DEEPSEEK_API_KEY` | off | 0.14 / 0.28 |
| qwen3.7-flash | qwen3.7-flash | Alibaba DashScope (intl) | `QWEN_API_KEY` | off | 0.03 / 0.13 |
| glm-5.3-flash | glm-5.3-flash | Z.ai | `ZAI_API_KEY` | cannot be disabled (set to low) | 0.15 / 0.50 |
| minimax-m3 | MiniMax-M3 | MiniMax (intl) | `MINIMAX_API_KEY` | off | 0.30 / 1.20 |

Opt-in: `tokenrouter` (GLM-5.3 free via TokenRouter, id `z-ai/glm-5.3-free`;
thinking cannot be disabled, so its hidden reasoning is recorded),
`claude-haiku` (Haiku 4.5, 1.00 / 5.00), `claude-sonnet` (Sonnet 5),
`gemini-2.5-flash-lite` (thinking off by default, 0.10 / 0.40),
`mistral-small`, and `openrouter` (any OpenRouter model id through one key).
For any aggregator, `python run_experiment.py --remote-models <key>` lists the
model ids your key can see.

Free tiers carry throughput limits, not just price limits. A model may declare
`requests_per_minute` and `max_concurrency` in `config.py`; the runner spaces
its calls and clamps `--concurrency` accordingly. TokenRouter's free GLM-5.3
is measured at 8 requests/minute and 2 concurrent, which is about **41 hours**
for one model across the full 19-condition matrix — a pilot budget, not a
full-run budget.

Prices are from public price lists in early September 2026 and only feed
`--estimate`; check them before a paid run. Model ids are verified by
`--smoke-test`, which also reports whether a provider returned hidden
reasoning despite the toggle.

**Why the thinking column matters.** The experiment manipulates the language
of the *visible* chain of thought. A model that first thinks in a hidden
channel and then writes the visible reasoning is not reasoning in the
requested language. Thinking is therefore switched off wherever the API
allows it. Where it cannot be (GLM-5.3 Flash; Gemini 3.1 can only be turned
down), the hidden reasoning is stored per sample and its rate is reported so
those models can be analysed separately or excluded.

This is not hypothetical. In the live GLM-5.3 pilot **100 % of samples
returned hidden reasoning**, and in the mandarin condition the visible
reasoning was 78 % Chinese while the hidden channel was 13 % — the model
thought in English and wrote up in Chinese. For such models the `no_cot`
control is also invalid: it still reasons, just invisibly.

### Local models

Any OpenAI-compatible local server works with no key and no cost:

```bash
ollama pull qwen3.5:9b            # ~8 GB VRAM; qwen3.6:27b needs ~17 GB; gemma4:12b also works
python run_experiment.py --models ollama --concurrency 1 --pilot --runs 1 --max-samples 30
# or: OLLAMA_MODEL=qwen3.6:27b python run_experiment.py --models ollama ...
# LM Studio: start its server, load a model, set LMSTUDIO_MODEL, use --models lmstudio
```

Ollama's `/v1` endpoint honours `reasoning_effort: none` for Qwen 3.x. Gemma 4
on that endpoint returns its text in the reasoning field; the adapter
promotes it to the visible answer. Small local models will need more of the
lenient answer mapping, so watch the `no-mark` and `off-lbl` columns.

### Tasks

| Task | Labels | test size |
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
are excluded. Test sizes are the HuggingFace test-split sizes, verified on download.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or --break-system-packages
pip install -r requirements.txt
# no pip on the machine? uv works: uv venv .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
cp .env.template .env      # fill in the keys for the models you will use
python -m pytest -q        # 56 offline tests, no network needed
```

Data comes from two places on first use and is cached under `data/`:
the test split of each task from the HuggingFace hub files of
`nguha/legalbench`, and `base_prompt.txt` from the LegalBench GitHub repo.
If the hub download fails, the loader prints the exact file to download by
hand and where to put it.

## Running

Local language-native models (Ollama) have their own runbook: see `HANDOFF_LOCAL.md`
and `run_local_native.sh`.

```bash
python run_experiment.py --list                     # models, thinking policy, env vars, tasks
python run_experiment.py --smoke-test               # one tiny call per default model
python run_experiment.py --smoke-test --models gpt-5.6-luna,deepseek-v4-flash
python run_experiment.py --estimate --runs 1        # downloads data, prints a rough cost table
python run_experiment.py --dry-run --pilot          # show the matrix, no calls

# Cheap first pass: 8 pilot conditions, 1 run, 50 samples/task
python run_experiment.py --pilot --runs 1 --max-samples 50

# Two models, three conditions, full samples
python run_experiment.py --models gpt-5.6-luna,deepseek-v4-flash --conditions english,mandarin,wildcard --runs 1

# Everything for the default models
python run_experiment.py --runs 1

# Offline pipeline check (fake answers, never a finding)
python run_experiment.py --models mock --runs 1 --max-samples 5
```

Results are appended per sample to `results/<model>__<condition>__run<N>.jsonl`.
Rerunning the same command resumes: completed samples are skipped, errored
ones are retried. `--fresh` discards a cell's file first. `--concurrency`
(default 5) is the number of in-flight requests; lower it if a provider
rate-limits you, and use 1 or 2 for local models.

Recommended order for a real run:

1. `--smoke-test` until every model you want is green. Fix ids or keys in
   `.env` / `config.py` if a provider has renamed something.
2. `--estimate --runs 1` to see the sample count per task and the cost.
3. `--pilot --runs 1 --max-samples 50`, then `python analyze.py`. Look at
   the compliance table: high `no-mark`, `off-lbl`, `trunc` or `hidden`
   rates for a model mean its numbers need care before the full run.
4. Full run with `--runs 1`; add runs later if the variance matters.

## Analysis

```bash
python analyze.py                 # report + CSVs from results/*.jsonl
python analyze.py --models tokenrouter --out-dir results/report_glm   # one model set, CSVs kept apart
jupyter lab legalbench_analysis.ipynb
```

The report prints, in order: majority-class baseline per task; condition /
model / family rankings with accuracy both raw and excluding runaways; a
compliance table per model × condition (share of letters in the expected
script, missing `ANSWER:` marker, prediction outside the label set, API
errors, runaways, hidden reasoning returned); an exact McNemar comparison of
**every** condition against English on identical samples, raw and
runaway-corrected, with a Bonferroni-corrected significance flag; a
within-model origin-advantage check derived from each model's
`origin_country`; runaway rate and length per condition; and token use.

A **runaway** is a response that never terminated (`finish_reason: length`)
and therefore contains no answer. Raw accuracy scores it as wrong; the
runaway-excluded view asks how accurate the model was when it did answer.
Nothing caps or cuts these generations — their length is part of the result.

The notebook reads the same JSONL files. It stops if `results/` is empty; it
does not generate placeholder data.

## Output files

```
results/
├── experiment_summary.json     # per-cell summaries incl. per-task accuracy, errors, truncation
├── <model>__<condition>__run<N>.jsonl   # one record per sample: full visible response, hidden reasoning if any
├── results_matrix.csv          # (model, condition, task, run) cells
├── condition_summary.csv
├── compliance.csv
├── paired_vs_english_excl_runaway.csv
└── runaways.csv
figures/                        # written by the notebook
```

## Cost

Use `--estimate`; it builds the real prompts and multiplies by the prices in
`config.py`. For orientation, the default six models × 19 conditions ×
~1,100 samples × 1 run is roughly 125k calls. Prompts are long (few-shot
examples plus some multi-paragraph disclosures), so input tokens matter as
much as output. At the September 2026 list prices above that is on the
order of 60–90 USD for the full default run, of which Gemini, MiniMax and
GPT-5.6 Luna are about three quarters and Qwen3.7 Flash a few dollars.
`--pilot --runs 1 --max-samples 50` is roughly 10–15 USD. Local models are
free and slow.
