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

## Addendum (same day): model refresh and the hidden-reasoning confound

The first pass kept a 2025 model list. It was replaced with the cheap tier
current in September 2026 (GPT-5.6 Luna, Gemini 3.1 Flash-Lite, DeepSeek V4
Flash, Qwen3.7 Flash, GLM-5.3 Flash, MiniMax M3) plus local Ollama / LM
Studio entries. Ids, endpoints and prices come from web searches of public
price lists and provider docs; the provider sites themselves were not
reachable from the sandbox, so `--smoke-test` is the real verification.

Refreshing the list surfaced a confound that the original design did not
consider and that the 2025 list mostly avoided: **every current cheap model
is a thinking model.** If it reasons in a hidden channel and then writes the
requested visible chain of thought, the experiment measures the language of
a write-up, not of the reasoning. Changes:

- Per-model `request_overrides` switch thinking off where the API allows it
  (OpenAI `reasoning_effort: none`, DeepSeek and MiniMax
  `thinking: {type: disabled}`, DashScope `enable_thinking: false`, Ollama
  `reasoning_effort: none`, OpenRouter `reasoning: {enabled: false}`).
- Where it cannot be switched off (GLM-5.3 Flash; Gemini 3.1 only goes down
  to "low"), the model config says so (`hidden_reasoning`), and whatever
  hidden reasoning the provider returns is stored per sample in
  `hidden_reasoning`, never merged into the visible response. The report's
  compliance table has a `hidden` column; a non-zero rate for a model that
  claims "off" means the toggle did not work and the model's numbers need
  the caveat.
- Inline `<think>…</think>` blocks are separated the same way.
- GPT-5.x requires `max_completion_tokens` and rejects `temperature`; the
  adapter handles both per model, so temperature 0 is not uniform across
  models. This is recorded in the summary and is a caveat for cross-model
  comparisons.
- Output cap raised from 2048 to 4096 tokens to reduce the truncation
  confound for verbose scripts.

Still unverified from here: every model id and price above, the exact
`reasoning_effort` values Gemini's OpenAI layer accepts for 3.1 Flash-Lite,
and DeepSeek's peak/off-peak schedule. The smoke test prints output tokens
and hidden-reasoning length per model so a wrong toggle is visible before
any money is spent.

## Addendum: independent review pass

A full-diff review against `main` found ten items; all were fixed and covered
by tests (38 now):

- A thinking-only reply (empty content, all text in the reasoning field)
  was promoted to the visible response and dropped from the hidden-reasoning
  count. It is still promoted so an answer can be extracted, but it stays
  recorded as hidden reasoning, is flagged `reasoning_promoted`, and is
  excluded from the script-ratio compliance signal.
- `**ANSWER**: Yes` (emphasis before the colon) was not recognised as the
  marker. Fixed; unclosed `<think>` blocks from truncated output are also
  separated now.
- A crash mid-write left a torn last line that made the cell unreadable on
  every resume; it is now skipped with a warning and redone. The end-of-cell
  rewrite is atomic (temp file + rename).
- An unreachable endpoint cost ~52 s of retries per sample; connection
  failures now get one quick retry and a cell aborts after 10 consecutive
  errors, keeping the rows already written.
- A 200 response that was not a TSV (consent or proxy page) would have been
  cached as the test split forever; rows are validated before caching, and
  the csv field-size limit is raised for long disclosures.
- Notebook: crashed without an `english` condition; wrote CSVs to the same
  paths as `analyze.py` with a different schema. Guarded and renamed to
  `notebook_*.csv`.
- `reasoning_part` sliced on an upper-cased offset (wrong for ß/ligatures);
  duplicate `.gitignore` entry.

## Addendum: first live run (2026-09-04)

The pipeline was finally executed against a real API (TokenRouter's free
`z-ai/glm-5.3-free`), which closed the gaps the sandbox could not reach.

**Verified working end to end.** 58 real samples across english / mandarin /
no_cot / wildcard on hearsay + abercrombie, **zero API errors**. The
HuggingFace test-split download works and every task matches the sizes in
`config.py` exactly (94, 50, 109, 139, 95, 95, 379→200, 3584→200, 66 =
1048 samples). Labels are all in their configured sets, no answer leakage,
no empty inputs. `analyze.py` and the notebook both run on the real output
(11 figures, no errors). Resume correctly skipped completed rows.

**The hidden-reasoning confound is confirmed, and it is total for this
model.** 100 % of samples returned hidden reasoning. In the mandarin
condition the *visible* reasoning is 78 % Han characters, but the hidden
channel is only 13 % — it opens "Let me think through this problem carefully
in Chinese as instructed" in English. The language manipulation reaches the
write-up, not the thinking. Any model whose `hidden` rate is high must be
reported separately; it cannot anchor the study.

**The `no_cot` control is invalid for thinking models.** 16/16 no-CoT
samples still returned hidden reasoning: the model reasons, just invisibly.
For such models the control measures "CoT hidden vs CoT shown", not
"reasoning vs no reasoning".

**New limits found and handled:**

- *Rate limit.* The free tier allows 8 requests/minute. Retries absorbed the
  429s without losing samples, but a sliding-window `RateLimiter` now spaces
  calls (`requests_per_minute` per model) instead of burning retry budget.
- *Concurrency limit.* Above ~2 in flight the gateway returns 503
  "hard concurrency limit reached". Models may now declare `max_concurrency`
  and the runner clamps `--concurrency` per model.
- *Truncation.* At 4096 tokens 12 % of english and 25 % of mandarin answers
  were cut off, confirming the prediction that non-Latin scripts lose their
  ANSWER line first. Raised to 8192, which cut it to ~10 % on the most
  verbose condition (wildcard); unused tokens are not billed.

**Throughput reality:** ~15 min for 48 samples at 8 rpm. One model across all
19 conditions × 1048 samples is roughly 41 hours on this free tier. Fine for
a pilot, not for the full matrix — use paid endpoints or a local model for
that.

**Task-design issue:** the real `unfair_tos` test split is 90 % "Other"
(180/200 sampled), so its majority baseline is 90 % while it consumes 19 % of
all samples. It contributes little signal and should be dropped or capped
lower.

## Addendum: output cap removed entirely

Capping output tokens is unsound for this benchmark. A truncated answer loses
its ANSWER line and is scored wrong, and truncation falls hardest on verbose
scripts — the exact variable the study manipulates. Measured on live GLM-5.3:
12% (english) / 25% (mandarin) truncated at 4096, still 8% (english) at 8192,
with p50=1150 but p90=6649 output tokens.

`MAX_OUTPUT_TOKENS = None` now omits the `max_tokens` field entirely so the
model stops when it is done. Verified against TokenRouter that omitting it
substitutes no small default: an uncapped request ran to natural stop at 8228
completion tokens. Anthropic requires the field, so that adapter falls back to
`providers.REQUIRED_MAX_TOKENS_FALLBACK`. The HTTP timeout is 1800s, since
uncapped generation runs at roughly 36 tokens/s.

`truncated` is still recorded and reported: any row that appears now is the
model hitting its own hard limit, and the report carries an `excl.trunc`
accuracy column so residual truncation can never masquerade as a language
effect.

## Current state (end of first full run)

The earlier addenda are an audit trail and mention settings that no longer
apply (token caps of 4096/8192, a wall-clock deadline). The configuration that
produced the results is:

- **No output cap** (`MAX_OUTPUT_TOKENS = None`) and **no wall-clock cut**;
  runaways run to the model's own limit and their length is reported.
- Streaming with a 120 s chunk-gap timeout, rate limiter and per-model
  concurrency cap; per-sample resume.
- Hidden reasoning stored separately and its rate reported per condition.

Analysis code contains no hand-typed model lists or pre-written hypothesis
verdicts: origin hypotheses derive from `origin_country`, notebook palettes
from the data, and the findings cell from the exact McNemar tests with
Bonferroni correction. Test-split sizes in `config.py` are the verified
HuggingFace sizes.

## Addendum: "reasoning off" on GLM-5.3 (TokenRouter) is not available

The plan was to rerun a subset with thinking disabled to separate the visible
chain of thought from the hidden one. Probed against the real endpoint:

- `thinking.type=disabled` → HTTP 400 `"GLM-5.3 does not support disabling
  thinking"` on 5 of 7 identical requests; the other 2 returned 200 with no
  hidden channel but the same reasoning emitted as visible content, ending in
  a stray `</think>` right before `ANSWER:` (the gateway routes to more than
  one upstream and they disagree).
- `reasoning_effort=none`, `reasoning.enabled=false`, `enable_thinking=false`
  → accepted, but hidden reasoning still returned (286–431 chars).

So there is no thinking-off condition for this model; the `no_cot` control is
"no *visible* reasoning" only (hidden reasoning present in 99.9% of rows) and is
reported as such. A genuine thinking-off comparison needs a model whose
provider honours the toggle (e.g. DeepSeek V4 Flash, or a local Qwen3 with
`think=false`), which is a between-model comparison, not within-model.

Side finding: 54 of 19,912 rows in the main run carry a stray `</think>` in the
visible content (28 of them runaways). `extract_answer` now treats it as a line
break; rescoring every stored row with the fixed extractor changed the
`answer_marker_found` flag on 10 rows and no `correct` value.

## Addendum: thinking-on vs thinking-off on GPT-5.6 Luna (2026-09-08)

Because GLM-5.3 cannot disable thinking, the on/off comparison was run on
GPT-5.6 Luna: `gpt-5.6-luna` (`reasoning_effort: none`, hidden reasoning 0% of
rows) and `gpt-5.6-luna-think` (`reasoning_effort: low`, hidden reasoning in
100% of CoT rows). Full matrix, 19 conditions × 1,048 samples × both variants
(39,824 rows in `results/` together with the GLM run), zero errors, zero
runaways, about $10 of API spend. Hidden reasoning on OpenAI is reported only
as `usage.completion_tokens_details.reasoning_tokens`; that count is now stored
per row (`reasoning_tokens`) and counts as hidden reasoning in the reports.

Findings (per-model, exact McNemar on identical samples, Bonferroni):

- **Hidden reasoning on beats off in 12 of 19 conditions** (+3.3 to +5.4
  points, all significant) and in none is off better. The gain is nil for
  `no_cot` (+0.5, n.s.): the "do not reason" instruction also suppresses the
  hidden channel (reasoning tokens present in 32% of rows vs 100% for CoT
  conditions), so Luna actually honours it, unlike GLM.
- **With reasoning on, the language of reasoning does not matter**: every
  condition lands at 79.3–81.1%, none differs from English; only `no_cot` is
  significantly worse (−3.0). No runaways at all, unlike GLM.
- **With reasoning off, the "language effect" is a "did it reason at all"
  effect.** Luna mostly ignores "show your full reasoning" and emits a bare
  `ANSWER:` line (median visible output 11 characters in 15 of 19 conditions).
  Conditions that did elicit some visible text scored higher: Hindi (+5.1 vs
  English, visible CoT in 67% of rows) and Korean (+4.7, 31%) are significant;
  English itself had visible CoT in 4% of rows. This is the confound the
  compliance table exists to expose, and it is why the two studies are
  reported separately (`analyze.py --models …`).
- **Script compliance is weak on Luna**: asked to reason in Mandarin it wrote
  Chinese letters in 2% (off) / 36% (on) of visible text; Hebrew 1% / 70%;
  Russian 1% / 24%. GLM complied at 80–100%. Luna's visible CoT is a
  post-hoc summary, usually in English, regardless of instruction.
- **Origin advantage** (now testable with two models): GLM-5.3's Mandarin
  delta vs English is −1.0 points against −0.5 for Luna, relative −0.5 → not
  found.

Reports: `results/report_glm.txt`, `results/report_luna.txt`,
`results/report_all.txt` (CSVs in matching sub-directories). The notebook runs
clean on the combined data.

## Addendum: why GPT-5.6 ignored the language instruction (2026-09-08)

The Luna results above looked too tidy, so the runs were audited:

- **The pipeline sent the right request.** The exact body the runner builds
  (system prompt with the language instruction, user prompt,
  `reasoning_effort: none`) was posted to OpenAI directly and reproduced the
  bare `ANSWER:` answers. Stored rows carry the full visible response.
- **GPT-5.6 obeys the wrong sentence.** Every language instruction in prompt
  version 1 ends with "Your final answer must still be in English." GPT-5.6
  reads that as "respond in English" and drops the (Chinese-language)
  instruction to reason in Chinese. Removing the sentence makes Luna reason in
  Chinese (82 % of letters, reasoning on); adding "do NOT answer with the label
  alone" makes it do so with reasoning off as well (100 % of probes).
- **Not a size effect.** gpt-5.6-sol and gpt-5.6-terra probed with the same
  prompt: 0 % Chinese, mostly bare labels, identical to Luna.
- **Other models are fine with the same wording.** GLM-5.3 wrote the requested
  script in 79–100 % of rows; gpt-4.1-mini wrote a chain of thought on every
  probe, 80–91 % in the requested script, under both prompt versions.
- **Temperature.** The config claimed GPT-5.x rejects temperature. That is
  only true with reasoning on; with `reasoning_effort: none` temperature 0 is
  accepted. The v1 Luna-off run therefore sampled at the default 1.0 while GLM
  ran at 0. Fixed (`temperature: 0.0` on `gpt-5.6-luna`). Even at temperature 0
  Luna flips between "label only" and "reason" on identical input.
- **Consequence for the v1 Luna study:** with reasoning off, the language
  conditions differed mainly in how often they provoked any reasoning at all
  (Hindi 67 % of rows, English 4 %); when Luna did reason in the requested
  script its accuracy was ~89 %. Those numbers measure instruction compliance,
  not reasoning language, and are kept only as a documented negative result.

Changes: prompt wording is now versioned (`--prompt-version`, default 1 = the
original, so all earlier data stays comparable; version 2 removes the English
sentence and forbids label-only answers). Every row records `prompt_version`;
`analyze.py` warns when versions are pooled and takes `--prompt-version`.
Non-reasoning models `gpt-4.1-mini` / `gpt-4.1-nano` were added: no hidden
channel at all, so their visible chain of thought is the reasoning.

Runs launched (results in the next addendum): gpt-4.1-mini, prompt v1, all 19 conditions (`results/`);
GPT-5.6 Luna reasoning off (temperature 0) and on (low), prompt v2, all 19
conditions (`results_prompt_v2/`).

## Addendum: results of the corrected runs (2026-09-08)

Two runs completed after the audit, both 19 conditions × 1,048 samples, zero
errors, temperature 0 where the API allows it:

- **gpt-4.1-mini, prompt v1** (no reasoning mode exists; `results/`,
  $16.36). Compliance near-perfect: marker in 99.9 % of rows, requested script
  66–100 %, runaways 0.0–0.1 %.
- **GPT-5.6 Luna reasoning off (temperature 0) and on (low), prompt v2**
  (`results_prompt_v2/`, $11.85). Compliance now acceptable: with reasoning
  off Luna wrote a chain of thought in 58–98 % of rows per condition (English
  73 %; under v1 it was 4 %), with reasoning on 98–100 %; requested script
  66–100 % (off) and 77–100 % (on).

Findings, per model, exact McNemar vs English on identical samples, Bonferroni
over 18:

- **gpt-4.1-mini: the language of the visible chain of thought matters.**
  9 of 18 conditions are significantly worse than English: Finnish −3.7,
  Hebrew −3.9, Mandarin −4.8, pseudocode −5.0, formal logic −5.0, Korean −5.5,
  Japanese −6.3, Arabic −6.4, Hindi −8.2 points. Nothing beats English. This
  is not a compliance artefact: Hindi rows actually written in Devanagari
  scored 68.7 % against 54.3 % for the few that drifted into English. The
  damage concentrates in the label-imbalanced tasks (contract_nli explicit
  identification 61 % → 29 %).
- **GPT-5.6 Luna, both modes: no language effect.** No condition differs from
  English in either variant (all |Δ| ≤ 2.7, none significant). Only `no_cot`
  is consistently lower (−2.2 off, −2.0 on; pooled −2.1, p = 0.0029, just
  outside the Bonferroni threshold).
- **Reasoning on vs off on Luna, done right: no difference.** With prompt v2
  the on/off deltas are −2.1 to +2.2 points and none is significant. The
  +3 to +5 point "reasoning-on advantage" of the v1 run was entirely the
  compliance artefact documented above (v1 reasoning-off did not reason at
  all). Once Luna writes a visible chain of thought, low-effort hidden
  reasoning adds nothing measurable.
- **GLM-5.3 (reasoning cannot be disabled): no language effect on accuracy
  once runaways are excluded**, but a language effect on *termination*:
  English 1.8 % runaways vs 3–7 % for every other natural language.
- **Training-origin hypothesis**: GLM's Mandarin delta vs English is −1.0
  against −2.3 for the other models (relative +1.3) → not found.

Interpretation, stated carefully: the one model that shows a large language
effect (gpt-4.1-mini) is also the oldest and weakest. Luna with reasoning off
is likewise a non-reasoning mode, and shows none. So the data do not support
"visible reasoning is language-sensitive, hidden reasoning is not"; they are
equally consistent with newer models simply being more robust to the reasoning
language. Separating those needs a stronger non-reasoning model
(gpt-4.1 at ~$45 a run) or an older reasoning model.

Spend on the OpenAI account: about $40 in total (Luna v1 $9.9, probes ~$1,
gpt-4.1-mini $16.4, Luna v2 $11.9). Reports: `results/report_gpt41mini.txt`,
`results_prompt_v2/report_luna_v2.txt`, `results/report_all.txt` (v1 rows,
four models).

## Addendum: language-native local models (2026-09-08)

Ollama 0.33.3 installed in user space (no root) on the RTX 5060 laptop (8 GB
VRAM), 16k context.  Because Ollama's OpenAI-compatible endpoint ignores
`think` (verified: `think: false` still produced 8–14k characters of hidden
reasoning on qwen3.5:9b) and cannot set `num_ctx`, local models use a native
`/api/chat` provider (`provider: "ollama"`), which honours the thinking toggle
and streams hidden reasoning separately.  Local models carry an output ceiling
equal to the context window (16,384 tokens): past it Ollama shifts context and
emits garbage (Nanda produced two 81,920-token runaways of 48 minutes each
before the ceiling existed).  Small `--max-samples` subsets are now nested
inside the canonical 200-per-task subset, so a 50-per-task local run scores
samples the cloud models also scored.

Probed on real prompts (2 samples × English, home language, no_cot):

| model | home | speed | verdict |
|---|---|---|---|
| Qwen3.5 9B (Alibaba) | Mandarin | 54 tok/s; thinking off ≈10 s/call, on ≈50 s | usable; Mandarin CoT 67 % under prompt v2 (0–44 % under v1); thinking toggle works natively |
| Llama-3.1-Swallow 8B v0.5 (Tokyo Tech) | Japanese | 50 tok/s | usable under v2 only (Japanese 39–71 %; English under v1) |
| EXAONE 3.5 7.8B (LG) | Korean | 60 tok/s | runs, but reasons in English under both prompt versions (0 % Hangul); kept as a compliance data point |
| Llama-3-Nanda 10B (MBZUAI) | Hindi | 38 tok/s | unusable: raw GGUF has no chat template; with a ChatML template it still echoes the prompt and gives no marker |
| ALLaM 7B preview (SDAIA) | Arabic | – | unusable: empty outputs, `<\|im` leaks, no marker |
| Sarvam-M 24B (Sarvam AI) | Hindi | 4 tok/s (spills to RAM) | unusable: 2–6 min per call, reasons in English for Hindi, ignores no_cot |
| Fanar-1 9B (QCRI) | Arabic | 8 tok/s | reasons in English for Arabic (0 %), runaways; dropped |
| Falcon-H1-Arabic 7B (TII) | Arabic | – | official GGUF is gated on Hugging Face; not tried |

Pattern worth stating: every small language-native fine-tune except Qwen and
Swallow reasons in English on an English-language task even when the
instruction is written in its home language.  A within-model home-language
test on those models would need the task text translated, which is a different
experiment.

Run in progress (`run_local_native.sh`, `results_prompt_v2/`): Qwen3.5 9B
thinking off, Swallow, EXAONE on the same 7 conditions (English, no_cot,
Mandarin, Japanese, Korean, Hindi, Arabic), then Qwen3.5 thinking on
(English, no_cot, Mandarin); 50 samples per task, prompt v2, temperature 0,
one generation at a time.  GPT-5.6 Luna v2 in the same directory is the
non-native comparison under an identical prompt.

## Addendum: Ithkuil condition (2026-09-10)

A 20th condition, `ithkuil` (family "Constructed"): reason in Ithkuil, the
conlang engineered for maximal precision and minimal ambiguity. It is the
extreme of the "precise notation helps" hypothesis, and since no model has
real Ithkuil fluency it also tests what the *attempt* does. Instruction in
English (an Ithkuil instruction would not be understood); no script check.

**GLM-5.3, prompt v1, all 1,048 samples, zero errors after a resume pass:**

- Runaway (non-terminating) rate **54.1 %** vs 1.8 % in English. Paired
  McNemar on the runaway flag: 555 samples ran away in Ithkuil where English
  terminated, 7 the other way, p ≈ 5×10⁻¹⁵⁴, the strongest effect in the study.
- Raw accuracy −38.2 points vs English (p ≈ 4×10⁻⁹⁷), entirely the runaways.
- On the 474 samples where both terminated: 84.4 % vs 84.2 %, identical. The
  hidden channel carried the answer; the visible Ithkuil-shaped text did not
  add or subtract anything.
- Cost: median 33k output tokens per answer (English 4.3k), runaways median
  51k and up to 83k tokens (20–40 minutes each), plus a median 116k
  characters of hidden reasoning. About 8× the tokens of English for the same
  accuracy when it worked, and no answer at all more than half the time.

Against the other notations on GLM (emergent −0.4, pseudocode −0.6, formal
logic −1.2 excl. runaways, all noise, runaway rates 1–5 %) this is a different
failure mode, not a bigger version of the same one.

**gpt-4.1-mini, prompt v1, partial (75 samples answered before the OpenAI
account ran out of credit):** 69 of 75 ran away, looping pseudo-Ithkuil
(`Vëxšëpšëx, vëxšëpšëx, …`) to the model's 32,768-token limit, ~3.5 minutes
each; accuracy 8 % vs 91 % for English on the same samples. GPT-5.6 Luna
(prompt v2): not started, 10 credit errors. Both resume with the same
commands once credit is added (`run_experiment.py --conditions ithkuil`).

Interpretation: the precision-language hypothesis inverts on both models
reached. Asked to reason in a maximally precise language it cannot produce,
the model generates language-shaped output until it hits a ceiling, and the
metric that moves is termination, not accuracy.

## Addendum: the constructed-language grid (2026-09-10)

Ithkuil confounds two things: maximal precision and the model's lack of
fluency. Three more conditions separate them: **Lojban** (unambiguous logical
grammar, decent web corpus), **Toki Pona** (~130 words, deliberately vague,
well known to models: Ithkuil's opposite) and **Esperanto** (regular,
natural-like, very well known). GLM-5.3, prompt v1, 1,048 samples each,
paired against English on the same samples:

| condition | runaway | p (runaway vs English) | acc when terminated (vs English) | tokens/answer (median) | hidden chars |
|---|---|---|---|---|---|
| english | 1.8 % | – | – | 1.0k | 3.2k |
| esperanto | 3.5 % | 0.01 | 86.3 % vs 85.9 % (+0.4, p = 0.64) | 1.4k | 3.5k |
| toki_pona | 24.2 % | 10⁻⁶⁰ | 87.4 % vs 88.4 % (−1.0, p = 0.06) | 6.4k | 19k |
| ithkuil | 54.1 % | 10⁻¹⁵⁴ | 84.4 % vs 84.2 % (+0.2, p = 1.0) | 33k | 116k |
| lojban | 75.0 % | 10⁻²²⁴ | 85.6 % vs 85.6 % (0.0, p = 1.0) | 66k (at the ceiling) | 205k |

Predictions recorded before the data: Lojban would run away far less than
Ithkuil (fluency, not precision, as the cause), Toki Pona would terminate
normally, Esperanto would behave like a natural language. Only the third held.

- Lojban is the worst condition in the study, worse than Ithkuil, although the
  model plainly knows it (the terminated output is well-formed Lojban). So
  fluency is not what drives the runaways, and neither is precision alone:
  Toki Pona, the least precise language possible, runs away 13× as often as
  English while being written correctly (median 84 % of words from the
  lexicon).
- On every conlang, accuracy on terminated samples is indistinguishable from
  English. Four conditions, four null results on correctness, four enormous
  effects on termination. The token cost ladder is Esperanto 1.4×, Toki Pona
  6×, Ithkuil 33×, Lojban 66× English, for no accuracy gain anywhere.
- What the four share is being a language the model *produces* far less
  fluently than it *recognises*; production quality tracks corpus size
  (Esperanto ≫ Toki Pona > Lojban ≈ Ithkuil) and so does the runaway rate,
  except that Lojban's unambiguous grammar seems to make matters worse rather
  than better. A production-fluency explanation fits; a precision explanation
  does not.

Same commands queued for gpt-4.1-mini and GPT-5.6 Luna (`run_openai_queue.sh`)
once the OpenAI account has credit.
