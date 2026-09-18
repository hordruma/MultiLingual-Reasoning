#!/usr/bin/env bash
# Prompt-v2 runs (2026-09-17).  Usage: run_v2_queue.sh <models> <concurrency> [full|capped]
#   capped (default): Ithkuil/Lojban(/Toki Pona for $TOKI_CAPPED=1) at 10 samples per task, since
#   runaways bill the full output limit; full: every condition on all samples.
# Resumable: rows on disk are skipped, errored rows retried.
set -u
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
MODELS="$1"; CONC="${2:-16}"; MODE="${3:-capped}"
NAT="english,german,russian,hindi,mandarin,arabic,hebrew,japanese,korean,turkish,finnish,hungarian,indonesian,vietnamese,formal_logic,pseudocode,emergent,wildcard,no_cot"
LOG="results_prompt_v2/queue_${MODELS//,/_}.log"
run() { echo "=== $(date -Is) $*" >> "$LOG"; "$PY" run_experiment.py --models "$MODELS" --runs 1 --concurrency "$CONC" \
        --prompt-version 2 --results-dir results_prompt_v2 "$@" >> "$LOG" 2>&1; echo "=== exit=$?" >> "$LOG"; }
if [ "$MODE" = full ]; then
    run --conditions "english,esperanto,toki_pona,ithkuil,lojban"
else
    if [ "${TOKI_CAPPED:-0}" = 1 ]; then
        run --conditions "english,esperanto"
        run --conditions "toki_pona,ithkuil,lojban" --max-samples 10
    else
        run --conditions "english,esperanto,toki_pona"
        run --conditions "ithkuil,lojban" --max-samples 10
    fi
fi
run --conditions "$NAT"
echo "=== $(date -Is) ALL DONE" >> "$LOG"
