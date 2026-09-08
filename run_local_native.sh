#!/usr/bin/env bash
# Sequential runs of the language-native local models (one GPU, one
# generation at a time).  Each model gets a 4-condition subset: English, its
# home language, no_cot, and Mandarin as a shared non-home reference.
# Prompt version 1 so the rows pair with the cloud runs in results/.
#
#   MAX_SAMPLES=50 ./run_local_native.sh          # default 50 per task (~450 samples/condition)
#   MAX_SAMPLES=200 ./run_local_native.sh         # the full cloud-run subset
#
# Resumable: rerunning skips finished samples.  Requires `ollama serve` on
# localhost:11434 with the models pulled (see config.py).
set -u
cd "$(dirname "$0")"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PY="${PY:-.venv/bin/python}"
LOG="$RESULTS_DIR/local_native_run.log"
mkdir -p "$RESULTS_DIR"

run() {  # model  conditions
  echo "=== $(date -Is) $1 [$2] max-samples=$MAX_SAMPLES" | tee -a "$LOG"
  "$PY" run_experiment.py --models "$1" --conditions "$2" --runs 1 --concurrency 1 \
      --prompt-version 1 --max-samples "$MAX_SAMPLES" --results-dir "$RESULTS_DIR" >> "$LOG" 2>&1
  echo "=== $(date -Is) $1 exit=$?" | tee -a "$LOG"
}

run qwen3.5-9b        english,mandarin,no_cot
run qwen3.5-9b-think  english,mandarin,no_cot
run exaone3.5-7.8b    english,korean,no_cot,mandarin
run swallow-8b        english,japanese,no_cot,mandarin
run allam-7b          english,arabic,no_cot,mandarin
run nanda-10b         english,hindi,no_cot,mandarin
echo "=== $(date -Is) ALL DONE" | tee -a "$LOG"
