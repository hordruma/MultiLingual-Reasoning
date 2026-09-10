#!/usr/bin/env bash
# Queued OpenAI runs waiting on account credit.  Resumable: each command skips
# rows already on disk and retries only errored ones.  No sample cap.
set -u
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
CONDS="ithkuil,toki_pona,lojban,esperanto"
LOG=results/openai_queue.log
echo "=== $(date -Is) start" | tee -a "$LOG"
"$PY" run_experiment.py --models gpt-4.1-mini --conditions "$CONDS" --runs 1 --concurrency 16 \
    --prompt-version 1 --results-dir results >> "$LOG" 2>&1
echo "=== $(date -Is) gpt-4.1-mini exit=$?" | tee -a "$LOG"
"$PY" run_experiment.py --models gpt-5.6-luna,gpt-5.6-luna-think --conditions "$CONDS" --runs 1 --concurrency 16 \
    --prompt-version 2 --results-dir results_prompt_v2 >> "$LOG" 2>&1
echo "=== $(date -Is) luna exit=$?" | tee -a "$LOG"
echo "=== $(date -Is) ALL DONE" | tee -a "$LOG"
