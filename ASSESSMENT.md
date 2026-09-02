# Repository assessment (2026-09-02)

Scope: everything on `main` at commit `2c04cd7` (the runner, providers, data
loader, analysis script, notebook, docs). Verdict first, then the itemised
findings and what was changed.

## Verdict

The original code had never produced a real result and could not have
produced a trustworthy one. The runner would have executed, but the numbers
it wrote would have reflected prompt defects, label leakage, truncation and
error handling far more than the reasoning language. The notebook, the only
place with charts and "findings", had only ever run on synthetic data with
the hypotheses baked into it, and crashed on the real output schema.

None of this looks like deliberate fraud; it looks like an unexecuted draft
whose docs describe an experiment that was never run. But the artefacts as
they stood would mislead anyone who trusted them, so they are treated here as
fakery and fixed or removed.

## Fakery and misleading claims

1. **Synthetic "findings" with the hypotheses baked in.** The notebook
   generated fake results when `results/` was empty: wildcard got +2 pp over
   English, Chinese models got +5/+6 pp in Mandarin, Mistral +3 pp in German,
   no-CoT −12 pp. The final "Key Findings" cell then printed
   "Wildcard > English: SUPPORTED" and the origin-advantage chart printed
   "CONFIRMED", from invented numbers. A small grey watermark was the only
   flag, and the synthetic frame was exported to `results/results_matrix.csv`,
   the same path as the real export. Removed entirely; the notebook now
   refuses to run without real per-sample results.
2. **The notebook never worked on real data.** The real summary had no `task`
   or `run` columns, which cells 21, 25, 27 and 29 required. Only the
   synthetic branch had ever executed. Rebuilt the loader on the per-sample
   JSONL files and executed the whole notebook end to end on real-schema data.
3. **Inflated design claims.** README, CLAUDE.md and the notebook said
   "21 conditions, 16 natural languages, 8 families". The config defines 19
   conditions, 14 languages, 9 families. The README table itself lists 19 rows
   under a "21 conditions" heading. Corrected everywhere.
4. **A task that does not exist.** `contract_nli_inclusion_of_verbatim_terms`
   is not a LegalBench task (GitHub returns 404). It would have been skipped
   silently, so the "10-task" experiment was a 9-task experiment. Replaced
   with the real `contract_nli_inclusion_of_verbally_conveyed_information`.
5. **An unscoreable task presented as a "clean binary signal".** `rule_qa`
   answers are free-text sentences; exact-match scoring gives ~0 % for every
   condition. Removed from the default list with a note. It can come back with
   an LLM-judge scorer.
6. **Cost estimate off by roughly an order of magnitude.** README said
   "$600–800" for the full frontier run. 19 × 6 × 3 × ~1,100 samples is about
   376 k calls; at Sonnet 4 / GPT-4o prices that is several thousand dollars.
   Replaced by a `--estimate` command that computes from the real prompts and
   per-model prices you can edit.
7. **Provider story did not match the code.** README and `.env.template` said
   Claude ran "via Azure Foundry"; the code called `api.anthropic.com`
   directly. Docs now describe what the code does.
8. **"3 runs for variance" at temperature 0** mostly measures provider
   nondeterminism, not sampling variance. Kept configurable, documented.

## Logic gaps that would have corrupted the numbers

9. **The model was never told the task.** The user prompt was literally
   `Task: abercrombie\n\ntext: ...` with no definition, no label vocabulary
   and no examples. LegalBench ships a `base_prompt.txt` per task (definition
   plus few-shot examples) that the official evaluation uses; it was ignored.
   For `abercrombie` (5 labels) and `unfair_tos` (9 labels) the model could
   not know what strings count as answers, so those tasks would have scored
   near zero regardless of reasoning language. The loader now fetches and
   renders the official prompt and the runner appends the allowed label list.
10. **Label leakage.** Every non-answer column was concatenated into the
    input. In `hearsay` and `personal_jurisdiction` the `slice` column names
    the legal sub-category of the fact pattern (e.g. "Statement made in
    court", which answers the hearsay question). `document_name` and
    `doctrine` leaked similarly. Only the fields the official prompt
    references are used now, and leak columns are explicitly excluded.
11. **Few-shot rows could silently become the test set.** The HuggingFace
    path used a script-based dataset that recent `datasets` versions refuse
    to load; the GitHub fallback then downloaded `train.tsv`, which on GitHub
    holds only the 4–160 few-shot examples, cached it, and the cache lookup
    accepted `train` as a stand-in for `test`. A 5-sample "experiment" would
    have run without warning. The loader now fetches the test split directly
    from the HuggingFace hub files, never substitutes train for test, and
    skips a task loudly if the test split cannot be obtained.
12. **Errors were scored as wrong answers and hidden.** An API failure became
    `correct: False` with no separate count, so an outage or a bad key looked
    like a reasoning deficit. Every exception was retried, including 401/400,
    wasting 22 s per sample. Now: errors are counted and reported separately,
    accuracy is reported both with and without them, only transient failures
    are retried (with `Retry-After`), and credentials are validated before
    any data is downloaded.
13. **Truncation was invisible.** `max_tokens` was 2048 and never checked.
    Hindi, Arabic, Hebrew and similar tokenize 2–4× less efficiently than
    English, so their answer line is the most likely to be cut off, which
    would be scored as wrong. This is the single largest confound for the
    research question and it was not measured. `truncated` is now recorded per
    sample and reported per model × condition.
14. **Answer extraction was brittle in exactly the conditions under test.**
    Only `ANSWER:` at line start matched. `**ANSWER:**` (markdown), `ANSWER：`
    (full-width colon, common in CJK output) and `A: Yes` (the format the
    few-shot examples invite) all fell through to "last non-empty line", which
    then scored whatever the reasoning ended with. No mapping onto the label
    set. Rewritten with tests; whether the marker was found and whether the
    prediction landed in the label set are both recorded.
15. **First-N sampling.** The first 200 rows were taken; `unfair_tos` has
    3,584 rows and no guarantee of shuffling. Now a seeded random subset,
    identical across models and conditions.
16. **Origin-advantage test confounded with model strength.** It compared
    DeepSeek's Mandarin accuracy with other models' Mandarin accuracy, which
    mostly measures which model is better overall. Now within-model: the
    model's own (language − English) delta against the same delta for the
    other models.
17. **No resume, no incremental writes.** A crash lost the whole run and
    re-running overwrote files. Results are now appended per sample and a
    rerun skips completed samples and retries errored ones.
18. **Concurrency only across conditions.** Samples inside a cell were
    sequential, so `--concurrency` barely mattered. Now sample-level.
19. **Paired t-test on cell accuracies** treated 60 aggregated numbers as
    observations. Added an exact McNemar test on per-sample correctness for
    the same (task, idx, run) keys, per model and pooled.
20. **No majority-class baseline.** Several tasks are unbalanced; without the
    baseline a 66 % score on a 66 %-majority task looks like competence. Now
    printed first in the report.
21. **Token-efficiency metric** (accuracy ÷ tokens) trivially crowns no-CoT.
    Kept as a descriptive number, flagged in the report.
22. Dead code: the `--pilot` branch built a list containing a non-existent
    `french` condition and then overwrote it.
23. No tests of any kind. Added 25 offline tests covering extraction,
    normalisation, scoring, prompt rendering, leak exclusion, seeded
    sampling, resume, config resolution and the statistics.

## What could not be verified from this sandbox

- **HuggingFace is blocked here** (proxy 403), so the direct download of
  `data/<task>/test.tsv` from the `nguha/legalbench` hub repo was written from
  the dataset's documented layout, not exercised. The first real run will
  tell; the loader prints a manual-download fallback if the URL is wrong.
- **Test-split sizes** in `config.py` are from the LegalBench paper and are
  approximate.
- **Model ids and prices** for the cheap tier are placeholders that must be
  checked against each provider's current list; `--smoke-test` verifies the
  ids, nothing verifies the prices.
- The cheap-model providers were exercised only through the mock; the
  OpenAI-compatible adapter follows the common `/chat/completions` contract,
  but Gemini's compatibility layer and DashScope have small quirks that a
  smoke test will surface.

## Remaining methodological caveats (not fixed, by design)

- The instruction to reason in a language is a request, not a guarantee.
  `script_ratio` in the report is a writing-system check for non-Latin
  scripts only; it cannot tell German from English. A language-ID model would
  be needed for a proper compliance measure.
- The few-shot examples in the official prompts are in English, so every
  condition sees English examples. That is the LegalBench standard and is
  held constant across conditions, but it is a bias toward English.
- Exact-match scoring after label normalisation is still strict; a model that
  writes "Yes, hearsay" is mapped to "Yes", but "Likely yes" is not.
