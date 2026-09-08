# Handoff: redo the local (language-native model) portion on a bigger machine

Written 2026-09-08. Everything below refers to the committed state of this
repo; nothing depends on the laptop this was developed on.

## What this portion is for

The cloud runs (GLM-5.3, GPT-5.6 Luna on/off, gpt-4.1-mini) are complete and
analysed (see `ASSESSMENT.md`, last three addenda). The open question is the
**training-origin hypothesis**: does a model trained heavily on language X do
relatively better when it reasons in X? That needs models actually trained on
the target languages, run on the same samples and the same prompt, and it is
what this local portion provides.

Design (already encoded in `run_local_native.sh` and `config.py`):

- Every local model runs the **same 7 conditions**: `english`, `no_cot`,
  `mandarin`, `japanese`, `korean`, `hindi`, `arabic`. Each model's
  home-language delta (home minus English) is then compared with every other
  model's delta on that language by `analyze.origin_advantage`, which derives
  the hypotheses from `origin_country` in `config.py`.
- **Prompt version 2** (`--prompt-version 2`). Small local models only follow
  the language instruction under this wording. The GPT-5.6 Luna v2 rows in
  `results_prompt_v2/` were made with the same prompt and act as the
  non-native comparison; keep the local rows in that directory.
- **Same samples as the cloud runs.** `--max-samples 200` is the canonical
  per-task subset every cloud model scored; smaller values are nested inside
  it (`data_loader.select_indices`), so any size pairs with the cloud rows.
- Temperature 0, no output cap except the context window
  (`max_output_tokens = num_ctx` on local entries: past the window Ollama
  shifts context and emits garbage).
- Thinking models run twice, `think: false` and `think: true`, through the
  **native Ollama provider** (`provider: "ollama"`, `/api/chat`). Ollama's
  OpenAI-compatible endpoint ignores `think`; do not use it for this.

## What was done on the laptop (8 GB VRAM), and what it showed

Probed on real prompts, prompt v2 (details and a table in `ASSESSMENT.md`,
addendum "language-native local models"):

| model | verdict |
|---|---|
| Qwen3.5 9B | usable; Mandarin CoT 67 %; thinking toggle works |
| Llama-3.1-Swallow 8B | usable; Japanese 39–71 % |
| EXAONE 3.5 7.8B | runs; reasons in English for Korean (0 % Hangul) |
| Sarvam-M 24B, Fanar-1 9B, Nanda 10B, ALLaM 7B | reason in English for Hindi/Arabic, or cannot follow the answer format |

Partial data on disk: `results_prompt_v2/qwen3.5-9b__english__run0.jsonl`
(154 rows, 50 samples/task run, stopped). It is resumable but a DGX run with
bigger models should simply use the bigger model keys; delete that file or
leave it, it does not interfere (rows are keyed by model).

The finding to keep in mind: **every small language-native fine-tune except
Qwen and Swallow reasons in English on this English-language task even when
instructed in its own script.** Bigger models may comply better; check the
`script` column in the compliance table before trusting any home-language
number, and treat `origin_advantage` as meaningless for a model whose script
ratio in its home language is near zero.

## Steps on the DGX

1. **Clone and install** (Python 3.10+; `uv` or plain pip):
   ```bash
   git clone <this repo> && cd MultiLingual-Reasoning
   uv venv .venv && uv pip install -r requirements.txt     # or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
   .venv/bin/python -m pytest -q tests                      # 60 offline tests, no keys needed
   ```
   No API keys are needed for the local portion. Copy `results_prompt_v2/`
   from the laptop (or this repo's owner) so the Luna v2 comparison rows are
   present; they are gitignored.

2. **Ollama** (root install is fine on the DGX; user-space also works):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   # server settings that matter:
   OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=1h ollama serve
   ```
   `OLLAMA_NUM_PARALLEL` is how many generations run at once; pass the same
   number as `CONCURRENCY` to the run script. On a DGX 4–8 is reasonable for
   a 27–35B model; the KV cache for 16k context × N slots must fit.

3. **Pull models.** Config entries already exist for these keys
   (`python run_experiment.py --list`); verify each tag pulls, HF GGUF repos
   change:
   ```bash
   ollama pull qwen3.6:27b                                               # qwen3.6-27b / qwen3.6-27b-think
   ollama pull hf.co/mmnga/Llama-3.1-Swallow-70B-Instruct-v0.3-gguf:Q4_K_M  # swallow-70b
   ollama pull exaone3.5:32b                                             # exaone3.5-32b
   ollama pull hf.co/lmstudio-community/sarvam-m-GGUF:Q4_K_M             # sarvam-m (Hindi; 24B, fine on GPU)
   ollama pull hf.co/mradermacher/Fanar-1-9B-Instruct-GGUF:Q4_K_M        # fanar-1-9b (Arabic)
   ```
   Worth adding if available (needs a config entry: copy `qwen3.6-27b`,
   change `model_id`, `display`, `origin_country`, `request_overrides`):
   - Qwen3.6 35B-A3B (`qwen3.6:35b-a3b`, MoE, fast) as a second Chinese model.
   - EXAONE 4.0 32B (community `ingu627/exaone4.0`, has a reasoning mode).
   - Jais 2 / Jais 30B (Arabic; needs a GGUF conversion, official is gated).
   - Falcon-H1-Arabic 7B (`hf.co/tiiuae/Falcon-H1-Arabic-7B-Instruct-GGUF`,
     gated: log in to Hugging Face first).
   - A Korean model that actually reasons in Korean (EXAONE 3.5 did not);
     HyperCLOVA X SEED or Kanana are candidates.
   Keep `origin_country` values that `analyze.ORIGIN_LANGUAGE` knows: China,
   Japan, South Korea, India, Saudi Arabia, United Arab Emirates, Qatar,
   France (→ german proxy), or extend that dict.

4. **Smoke test and probe before committing GPU hours:**
   ```bash
   .venv/bin/python run_experiment.py --smoke-test --models qwen3.6-27b,swallow-70b,exaone3.5-32b,sarvam-m,fanar-1-9b
   MAX_SAMPLES=3 CONCURRENCY=4 MODELS="qwen3.6-27b swallow-70b exaone3.5-32b sarvam-m fanar-1-9b" ./run_local_native.sh
   .venv/bin/python analyze.py --results-dir results_prompt_v2 | sed -n '/COMPLIANCE/,/ORIGIN/p'
   ```
   Read the `script` and `no-mark` columns. A model with ~0 % script in its
   home language or a high no-marker rate is not worth a full run; drop it
   from `MODELS`. Then delete the 3-sample rows (`rm results_prompt_v2/<model>__*.jsonl`)
   or just continue: the full run resumes over them.

5. **Full run** (overnight; sequential over models, parallel within a model):
   ```bash
   MAX_SAMPLES=200 CONCURRENCY=4 \
   MODELS="qwen3.6-27b swallow-70b exaone3.5-32b sarvam-m fanar-1-9b qwen3.6-27b-think" \
   nohup ./run_local_native.sh > /dev/null 2>&1 &
   tail -f results_prompt_v2/local_native_run.log
   ```
   Thinking-on keys (`hidden_reasoning: "on"`) automatically get only
   `english,no_cot,mandarin`. Rough sizing: 7 conditions × 1,048 samples =
   7,336 calls per model; at ~4 parallel × 40 tok/s and ~500-token answers
   that is 3–5 hours per 27–35B model, longer for 70B and for thinking-on.
   The chain is resumable: rerun the same command after any interruption.

6. **Analyse and hand back:**
   ```bash
   .venv/bin/python analyze.py --results-dir results_prompt_v2 --out-dir results_prompt_v2/report_local > results_prompt_v2/report_local.txt
   ```
   The report's "HIDDEN REASONING ON vs OFF" section appears for each
   off/on pair, "TRAINING-DATA ORIGIN ADVANTAGE" for every model whose
   `origin_country` maps to a condition, and the compliance table tells you
   which of those numbers to believe. Copy `results_prompt_v2/` back (JSONL
   rows are the raw data; everything else regenerates from them).

## Things not to change without thinking

- `--prompt-version 2` for local runs (v1 is the cloud default and stays
  the default; every row records its version and `analyze.py` warns when
  versions are pooled).
- `SAMPLE_SEED`, `CANONICAL_SUBSET` and the task list in `config.py`: they
  define the samples every model shares.
- The `ANSWER:` format and label sets; scoring is exact-match on them.
- Never add a model's numbers by hand anywhere; `analyze.py` and the notebook
  derive everything from the JSONL rows.
