# LegalBench Chain-of-Thought Language Experiment

## Research Question

Does the language used for internal reasoning (chain-of-thought) affect LLM reasoning quality? Is English optimal, or do other languages, formal notations, or unconstrained polyglot reasoning produce better results?

## Experimental Design

**21 conditions** (16 natural languages across 8 families + 3 abstract formats + 1 wildcard + 1 no-CoT control) × **6 models** (Claude, GPT-4o, Mistral, DeepSeek, Qwen, Gemini) × **10 LegalBench tasks** × **3 runs** for variance.

### Conditions

| # | Condition | Family | Rationale |
|---|-----------|--------|-----------|
| 1 | English | Indo-European | Baseline |
| 2 | German | Indo-European | Compound words, V2 clause structure |
| 3 | Russian | Indo-European | Case system, flexible word order |
| 4 | Hindi | Indo-European | SOV, postpositions, ergative |
| 5 | Mandarin | Sino-Tibetan | High information density per character |
| 6 | Arabic | Afroasiatic | Root-based morphology |
| 7 | Hebrew | Afroasiatic | Root-based, different script |
| 8 | Japanese | Japonic | SOV, agglutinative, mixed scripts |
| 9 | Korean | Koreanic | SOV, agglutinative |
| 10 | Turkish | Turkic | Agglutinative, regular grammar |
| 11 | Finnish | Uralic | 15 grammatical cases, agglutinative |
| 12 | Hungarian | Uralic | Agglutinative, different structure |
| 13 | Indonesian | Austronesian | Simple morphology, affixation |
| 14 | Vietnamese | Austroasiatic | Tonal, analytic |
| 15 | Formal Logic | Abstract | Propositional/predicate logic |
| 16 | Pseudocode | Abstract | Code-like reasoning |
| 17 | Emergent | Abstract | Model invents own notation |
| 18 | Wildcard | Wildcard | Unconstrained – use anything |
| 19 | No CoT | Control | Direct answer, no reasoning |

### Models

| Model | Provider | Origin | Rationale |
|-------|----------|--------|-----------|
| Claude Sonnet 4 | Anthropic (Azure) | USA | Baseline |
| GPT-4o | OpenAI (Azure) | USA | Industry standard benchmark |
| Mistral Large | Mistral AI (Azure) | France | French training data bias |
| DeepSeek V3 | DeepSeek (Azure) | China | Mandarin training data bias |
| Qwen Max | Alibaba (direct) | China | Second Chinese-origin model |
| Gemini 2.5 Flash | Google (direct) | USA | Broadest multilingual training |

## Setup

```bash
pip install -r requirements.txt --break-system-packages
cp .env.template .env
# Fill in your API keys in .env
source .env  # or use dotenv
```

## Running

```bash
# Dry run – see the matrix without API calls
python run_experiment.py --dry-run

# Full experiment (all 21 conditions × 6 models × 3 runs)
python run_experiment.py

# Subset of models/conditions
python run_experiment.py --models claude-sonnet,gpt-4o --conditions english,mandarin,wildcard

# Single run (no variance measurement)
python run_experiment.py --runs 1

# Adjust concurrency (default 5)
python run_experiment.py --concurrency 10
```

## Analysis

```bash
python analyze.py
# Exports: results/results_matrix.csv + printed report
```

## Output Structure

```
results/
├── experiment_summary.json          # aggregate results
├── results_matrix.csv               # for external visualization
├── claude-sonnet__english__run0.jsonl    # detailed per-sample results
├── claude-sonnet__mandarin__run0.jsonl
├── ...
```

## Key Metrics

- **Accuracy**: correct answers / total per cell
- **Token efficiency**: accuracy per 1000 reasoning tokens
- **Origin advantage**: does model X reason better in language Y associated with its training data?
- **Wildcard analysis**: does unconstrained reasoning outperform any single language?

## Cost Estimate

Full run (21 × 6 × ~200 samples × 3 runs): approximately $600–800 USD.
