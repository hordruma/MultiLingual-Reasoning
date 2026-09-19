# Findings: does the language of the chain of thought change legal-classification accuracy?

Condensed from ASSESSMENT.md (2026-09-18). Numbers are paired comparisons
against English on identical samples (exact McNemar, Bonferroni across
conditions), 1,048 LegalBench samples over 9 closed-label tasks per cell,
temperature 0, no output cap. "Runaway" = the model never stopped and hit its
own token limit. "Terminated Δ" drops runaways on either side, i.e. asks
"given it finished, was it as accurate?".

## Setup

- **24 reasoning conditions**: 14 natural languages, 3 notations (formal
  logic, pseudocode, emergent), 4 constructed languages (Esperanto, Toki Pona,
  Ithkuil, Lojban), wildcard (mixing permitted), polyglot (mixing required),
  and a no-reasoning control.
- **Models**: GLM-5.3 (prompt v1; hidden reasoning cannot be disabled),
  DeepSeek V4 Flash, Claude Haiku 4.5, gpt-4.1-mini, GPT-5.6 Luna with
  reasoning off and on (all prompt v2). Local 8–27B models were probed and
  mostly failed to reason in their home language on an English task; they are
  documented, not used.
- Prompt v2 is the wording every model obeys; v1 told GPT-5.6 to answer in
  English and it took that as licence to skip the requested language. GLM was
  run under v1 only, because its free route (TokenRouter) was down when the
  v2 reruns were done.

## 1. For a capable model the language of visible reasoning does not matter

| model | English | natural languages + notations vs English | significant after Bonferroni |
|---|---|---|---|
| GPT-5.6 Luna, reasoning on | 79.8 % | −2.0 … +0.9 | none |
| GPT-5.6 Luna, reasoning off | 80.8 % | −2.7 … +1.0 | none |
| DeepSeek V4 Flash | 77.7 % | −3.1 … +3.2 | none |
| Claude Haiku 4.5 | 79.2 % | −3.7 … +1.5 | Japanese −3.7 |
| GLM-5.3 (v1) | 85.9 % | within noise once runaways excluded | none |
| gpt-4.1-mini | 76.4 % | −8.7 … +1.1 | 12 of 17 worse: Arabic −8.7, formal logic −8.6, Japanese −7.3, Hebrew −7.2, Hungarian −7.1, Mandarin −7.0, pseudocode −7.0, Finnish −6.4, Korean −6.4, German −6.1, Vietnamese −3.8, Indonesian −3.3 |

- Nothing beats English on any model. No notation (formal logic, pseudocode,
  emergent) helps anywhere; on gpt-4.1-mini they hurt as much as the hardest
  scripts.
- The one model with a large language effect is the oldest and weakest. On
  it the damage is real reasoning damage, not non-compliance: Hindi rows
  actually written in Devanagari scored 69 % vs 54 % for rows that drifted to
  English, and it concentrates in label-imbalanced tasks.
- Hidden reasoning on vs off on Luna (same prompt, same samples): no
  difference in any condition (−2.1 … +2.2). The "reasoning-on advantage"
  seen under prompt v1 was entirely because reasoning-off Luna had not
  written any reasoning.
- Training-origin hypothesis (Chinese models better in Mandarin): not found.
  GLM's and DeepSeek's Mandarin deltas are no better than the US models'.
- Even the no-reasoning control is within a few points of English: −2.2 /
  −2.0 on Luna off / on, −1.5 on gpt-4.1-mini, −1.2 on GLM (all n.s. after
  correction) and +3.6 on DeepSeek (p = 0.016, not significant after
  correction). On these closed-label tasks a visible chain of thought is
  worth at most a couple of points, and its language adds nothing on top.

## 2. Constructed languages move termination, not accuracy

Runaway rate and raw Δ vs English (raw counts a runaway as wrong):

| model | Esperanto | Toki Pona | Ithkuil | Lojban |
|---|---|---|---|---|
| GLM-5.3 (v1) | 3.5 %, +0.4 term. | 24 %, −1.0 term. | 54 %, +0.2 term. | 75 %, 0.0 term. |
| DeepSeek V4 Flash | 3 %, −1.9 | 12 %, −11.7 | 57 %, −44.1 | 67 %, −53.1 |
| Claude Haiku 4.5 | 0 %, −0.9 | 0 %, −12.4 (n=89) | 36 %, −33.3 (n=33) | – |
| gpt-4.1-mini | 0 %, −3.6 | 84 %, −67.0 (n=109) | 86 %, −65.6 (n=90) | 98 %, −77.8 (n=90) |
| GPT-5.6 Luna (off / on) | 0 %, −1.4 / +0.7 | 0 %, −3.0 / +0.7 | 0 %, +1.7 (n=232) / +0.2 | 0 %, −6.4 / −0.9 |

- **The ordering Lojban ≥ Ithkuil > Toki Pona > Esperanto holds on every
  model that runs away at all.** It tracks how much of the language the model
  has *produced* in training (Esperanto ≫ Toki Pona > Lojban ≈ Ithkuil), not
  the language's precision: Lojban, the most unambiguous, is the worst;
  Toki Pona, the vaguest, runs away 12–13× more than English while being
  written correctly.
- **On samples where the model did finish, accuracy is close to English:**
  GLM shows an exact null (+0.2, 0.0, −1.0, +0.4); DeepSeek loses 3–6 points
  in Toki Pona and Lojban; Haiku loses 12 in Toki Pona without a single
  runaway. So a weaker model pays a small accuracy tax even when it finishes,
  a stronger one pays only in termination.
- **Runaways are loops, not long reasoning.** Measured by zlib compression
  and distinct words per response (`analyze.py` REPETITION section):
  gpt-4.1-mini's Ithkuil runaways contain a median of 3 distinct words
  (`Klaţţh-ţhâlţh-ţhâlţh-…` to the 32k-token limit); constructed-language
  runaways across models compress to 0.014–0.017 of their size with 25–39
  distinct words, against 0.44–0.57 for terminated answers and 0.07–0.13 for
  the rare natural-language runaways. Even terminated conlang answers have a
  lower distinct-word ratio than English.
- **Cost:** Esperanto 1.4×, Toki Pona 6×, Ithkuil 33×, Lojban 66× the output
  tokens of English on GLM, for no accuracy gain anywhere.
- **GPT-5.6 Luna's zero runaways are non-compliance, not robustness.** With
  reasoning off it answers with the bare label on 20–42 % of conlang samples
  and refuses 26 % of Ithkuil outright, and its uncapped Lojban run is −6.4
  (p < 10⁻⁴); with reasoning on, its hidden channel is active on 100 % of
  samples, the visible Lojban is a write-up of a decision already made, and
  Ithkuil/Lojban are +0.2/−0.9 on 1,048 samples.
- An Ithkuil IV parser (christian-oudard/ithkuil) cannot score fidelity: 84 %
  of the words in *English* reasoning parse as valid Ithkuil. Recorded as a
  negative result in `ithkuil_fidelity.py`.

## 3. "Use every language at once" does not help either

- **wildcard** (mixing permitted, optimise purely for correctness): every
  model answers in plain English (0 of ~2,000 GLM and gpt-4.1-mini responses
  contain non-Latin script) and scores −1 to −5 vs English.
- **polyglot** (mixing *required* within every sentence; the instruction is
  itself written in nine languages and gives the rationale): DeepSeek
  complies, mixing English, German, French, Spanish, Chinese and logic
  notation, and scores −3.5 (p = 0.005); gpt-4.1-mini complies best (mixed
  script in 55 % of responses) and scores −10.9 (p < 10⁻⁴); Luna −1.1 and
  Luna-think −0.1 (both n.s., and Luna-off mixes scripts in only 8 %).
  Runaways stay at the English rate everywhere. Prediction before the run
  was "equal to English, more runaways"; instead the cost is accuracy, in
  proportion to how much the model actually complies.
- The broadest possible vocabulary buys nothing because expressiveness was
  never the bottleneck: accuracy is flat across 20 languages and notations on
  every capable model. What the language changes is how far the model is
  from its most-practised production register, and that shows up as
  termination failures and a small accuracy tax, never as a gain.

## Bottom line

1. On current models, the language you ask for the chain of thought in is
   irrelevant to accuracy within ±3 points, English included. Older/weaker
   models lose up to 9 points in non-English or notational reasoning.
2. Languages the model recognises but rarely produces (Toki Pona, Lojban,
   Ithkuil) cause looping non-termination in 12–98 % of samples on models
   that don't refuse, at 6–66× the token cost; the answers that do come back
   are about as good as English.
3. Forcing or allowing multilingual mixing is never a gain: neutral on the
   strong models, −3.5 on DeepSeek, −11 on gpt-4.1-mini.
4. Skipping the chain of thought entirely costs at most ~2 points and on one
   model nothing at all; the language of that chain of thought is worth
   nothing on top.

## Caveats

- Different models were run under different prompt versions (GLM v1, the
  rest v2). The conlang findings replicate across both.
- Ithkuil/Lojban/Toki Pona on gpt-4.1-mini, Ithkuil on Haiku and Ithkuil on
  Luna-off are capped or partial subsets (33–232 samples), because runaways
  bill the full output limit or credit ran out.
- DeepSeek runaways stop at 8,192 tokens (server default), Haiku and
  gpt-4.1-mini at 32k; runaway *rates* are comparable across models,
  lengths are not.
- One run per cell at temperature 0; no repeat-sampling variance.
- Full audit trail, including what was wrong with earlier versions of this
  pipeline, is in ASSESSMENT.md.
