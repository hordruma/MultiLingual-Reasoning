#!/usr/bin/env bash
# Sequential runs of the language-native local models (one GPU, one
# generation at a time).  Every model gets the same 7-condition set: English,
# no_cot, and the five "home" languages of the native models, so each model's
# home-language delta can be compared with every other model's delta on that
# language (analyze.origin_advantage).  Prompt version 2, because the small
# local models only follow the language instruction under that wording, and
# results go next to the GPT-5.6 Luna v2 rows, which then serve as the
# non-native comparison under an identical prompt.
#
#   MAX_SAMPLES=50 ./run_local_native.sh          # default 50 per task (450 samples/condition)
#   MAX_SAMPLES=200 CONCURRENCY=4 MODELS="qwen3.6-27b swallow-70b" ./run_local_native.sh   # DGX
#
# Resumable: rerunning skips finished samples.  Requires `ollama serve` on
# localhost:11434 with the models pulled (see config.py).
set -u
cd "$(dirname "$0")"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
RESULTS_DIR="${RESULTS_DIR:-results_prompt_v2}"
PY="${PY:-.venv/bin/python}"
CONCURRENCY="${CONCURRENCY:-1}"   # match OLLAMA_NUM_PARALLEL on the server
LOG="$RESULTS_DIR/local_native_run.log"
CONDS="english,no_cot,mandarin,japanese,korean,hindi,arabic"
mkdir -p "$RESULTS_DIR"

run() {  # model  conditions
  echo "=== $(date -Is) $1 [$2] max-samples=$MAX_SAMPLES" | tee -a "$LOG"
  "$PY" run_experiment.py --models "$1" --conditions "$2" --runs 1 --concurrency "$CONCURRENCY" \
      --prompt-version 2 --max-samples "$MAX_SAMPLES" --results-dir "$RESULTS_DIR" >> "$LOG" 2>&1
  echo "=== $(date -Is) $1 exit=$?" | tee -a "$LOG"
}

# Override the model list with MODELS="key1 key2 ..." (config.py keys).  Keys whose
# hidden_reasoning is "on" (thinking variants, ~5x slower) get the short set.
MODELS="${MODELS:-qwen3.5-9b swallow-8b exaone3.5-7.8b qwen3.5-9b-think}"
for m in $MODELS; do
  if "$PY" -c "import config,sys; sys.exit(0 if config.MODELS['$m'].get('hidden_reasoning')=='on' else 1)"; then
    run "$m" english,no_cot,mandarin
  else
    run "$m" "$CONDS"
  fi
done
# allam-7b and nanda-10b were probed and cannot follow the answer format
# (empty output, template leaks, prompt echo); see ASSESSMENT.md.
echo "=== $(date -Is) ALL DONE" | tee -a "$LOG"
