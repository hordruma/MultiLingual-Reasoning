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
#
# Resumable: rerunning skips finished samples.  Requires `ollama serve` on
# localhost:11434 with the models pulled (see config.py).
set -u
cd "$(dirname "$0")"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
RESULTS_DIR="${RESULTS_DIR:-results_prompt_v2}"
PY="${PY:-.venv/bin/python}"
LOG="$RESULTS_DIR/local_native_run.log"
CONDS="english,no_cot,mandarin,japanese,korean,hindi,arabic"
mkdir -p "$RESULTS_DIR"

run() {  # model  conditions
  echo "=== $(date -Is) $1 [$2] max-samples=$MAX_SAMPLES" | tee -a "$LOG"
  "$PY" run_experiment.py --models "$1" --conditions "$2" --runs 1 --concurrency 1 \
      --prompt-version 2 --max-samples "$MAX_SAMPLES" --results-dir "$RESULTS_DIR" >> "$LOG" 2>&1
  echo "=== $(date -Is) $1 exit=$?" | tee -a "$LOG"
}

run qwen3.5-9b        "$CONDS"
run swallow-8b        "$CONDS"
run exaone3.5-7.8b    "$CONDS"
# thinking-on Qwen is ~5x slower; home language + controls only
run qwen3.5-9b-think  english,no_cot,mandarin
# allam-7b and nanda-10b were probed and cannot follow the answer format
# (empty output, template leaks, prompt echo); see ASSESSMENT.md.
echo "=== $(date -Is) ALL DONE" | tee -a "$LOG"
