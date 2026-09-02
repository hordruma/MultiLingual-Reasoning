"""
Experiment Configuration
========================
Defines the reasoning-language conditions, the models, the LegalBench tasks,
and the experiment parameters.

Counts (kept honest – see ASSESSMENT.md):
  * 14 natural-language conditions across 9 language families
  * 3 abstract-notation conditions, 1 wildcard, 1 no-CoT control
  = 19 conditions total
"""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Every model entry declares:
#   provider      – "anthropic" (native Messages API), "openai_compat"
#                   (any OpenAI-style /chat/completions endpoint) or "mock"
#   model_id      – the id sent to the API
#   api_key_env   – env var holding the key
#   base_url_env  – (openai_compat only) env var holding the base URL, which
#                   must end where "/chat/completions" can be appended
#   base_url      – default base URL if the env var is unset
#   price_in/out  – USD per 1M tokens, ONLY used by `--estimate`.  Prices
#                   change often; treat these as placeholders and check the
#                   provider's price list before trusting an estimate.
#
# DEFAULT_MODELS is the cheap set the runner uses when --models is omitted.
# Any key in MODELS can be requested explicitly with --models.

MODELS = {
    # ── Cheap cloud tier (defaults) ─────────────────────────────────────
    "gpt-4o-mini": {
        "provider": "openai_compat",
        "model_id": "gpt-4o-mini",
        "display": "GPT-4o mini (OpenAI)",
        "origin_country": "USA",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "price_in": 0.15, "price_out": 0.60,
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "display": "Claude Haiku 4.5 (Anthropic)",
        "origin_country": "USA",
        "api_key_env": "ANTHROPIC_API_KEY",
        "price_in": 1.00, "price_out": 5.00,
    },
    "gemini-flash-lite": {
        "provider": "openai_compat",
        "model_id": "gemini-2.5-flash-lite",
        "display": "Gemini 2.5 Flash-Lite (Google)",
        "origin_country": "USA",
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "price_in": 0.10, "price_out": 0.40,
    },
    "deepseek-chat": {
        "provider": "openai_compat",
        "model_id": "deepseek-chat",
        "display": "DeepSeek V3 (DeepSeek direct)",
        "origin_country": "China",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com/v1",
        "price_in": 0.27, "price_out": 1.10,
    },
    "mistral-small": {
        "provider": "openai_compat",
        "model_id": "mistral-small-latest",
        "display": "Mistral Small (Mistral direct)",
        "origin_country": "France",
        "api_key_env": "MISTRAL_API_KEY",
        "base_url_env": "MISTRAL_BASE_URL",
        "base_url": "https://api.mistral.ai/v1",
        "price_in": 0.10, "price_out": 0.30,
    },
    "qwen-plus": {
        "provider": "openai_compat",
        "model_id": "qwen-plus",
        "display": "Qwen Plus (Alibaba DashScope)",
        "origin_country": "China",
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "price_in": 0.40, "price_out": 1.20,
    },

    # ── OpenRouter: one key, many models (optional alternative route) ───
    # Set OPENROUTER_API_KEY and pick any OpenRouter model id.  Useful if
    # you would rather not open six separate accounts.
    "openrouter": {
        "provider": "openai_compat",
        "model_id": "openai/gpt-4o-mini",      # override with OPENROUTER_MODEL
        "model_id_env": "OPENROUTER_MODEL",
        "display": "OpenRouter (model from OPENROUTER_MODEL)",
        "origin_country": "n/a",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "base_url": "https://openrouter.ai/api/v1",
        "price_in": 0.0, "price_out": 0.0,
    },

    # ── Original frontier tier (kept, not default) ──────────────────────
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "display": "Claude Sonnet 4 (Anthropic)",
        "origin_country": "USA",
        "api_key_env": "ANTHROPIC_API_KEY",
        "price_in": 3.00, "price_out": 15.00,
    },
    "gpt-4o": {
        "provider": "openai_compat",
        "model_id": "gpt-4o",
        "display": "GPT-4o (OpenAI)",
        "origin_country": "USA",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "price_in": 2.50, "price_out": 10.00,
    },
    "gemini-2.5-flash": {
        "provider": "openai_compat",
        "model_id": "gemini-2.5-flash",
        "display": "Gemini 2.5 Flash (Google)",
        "origin_country": "USA",
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "price_in": 0.30, "price_out": 2.50,
    },
    "mistral-large": {
        "provider": "openai_compat",
        "model_id": "mistral-large-latest",
        "display": "Mistral Large (Mistral direct)",
        "origin_country": "France",
        "api_key_env": "MISTRAL_API_KEY",
        "base_url_env": "MISTRAL_BASE_URL",
        "base_url": "https://api.mistral.ai/v1",
        "price_in": 2.00, "price_out": 6.00,
    },
    "qwen-max": {
        "provider": "openai_compat",
        "model_id": "qwen-max",
        "display": "Qwen Max (Alibaba DashScope)",
        "origin_country": "China",
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "price_in": 1.60, "price_out": 6.40,
    },

    # ── Offline test double – never a source of findings ────────────────
    "mock": {
        "provider": "mock",
        "model_id": "mock",
        "display": "MOCK (offline pipeline test, answers are fake)",
        "origin_country": "n/a",
        "api_key_env": None,
        "price_in": 0.0, "price_out": 0.0,
    },
}

DEFAULT_MODELS = [
    "gpt-4o-mini",
    "claude-haiku",
    "gemini-flash-lite",
    "deepseek-chat",
    "mistral-small",
    "qwen-plus",
]

# ---------------------------------------------------------------------------
# Language conditions – 14 natural languages + 3 abstract + wildcard + no-CoT
# ---------------------------------------------------------------------------
# `script` is used by the compliance heuristic in analyze.py: the dominant
# Unicode script the reasoning should be written in.  Latin-script languages
# cannot be told apart cheaply, so they share "latin".
CONDITIONS = {
    # ── Indo-European (4) ──────────────────────────────────────────────────
    "english": {
        "family": "Indo-European",
        "script": "latin",
        "instruction": (
            "Think through this problem step by step in English. "
            "Show your full reasoning in English before giving your final answer."
        ),
    },
    "german": {
        "family": "Indo-European",
        "script": "latin",
        "instruction": (
            "Denke Schritt für Schritt auf Deutsch über dieses Problem nach. "
            "Zeige deine vollständige Argumentation auf Deutsch, bevor du deine endgültige Antwort gibst. "
            "Your final answer must still be in English."
        ),
    },
    "russian": {
        "family": "Indo-European",
        "script": "cyrillic",
        "instruction": (
            "Продумай эту задачу шаг за шагом на русском языке. "
            "Покажи полный ход рассуждений на русском, прежде чем дать окончательный ответ. "
            "Your final answer must still be in English."
        ),
    },
    "hindi": {
        "family": "Indo-European",
        "script": "devanagari",
        "instruction": (
            "इस समस्या पर हिंदी में चरणबद्ध तरीके से विचार करें। "
            "अपना पूरा तर्क हिंदी में दिखाएं, फिर अपना अंतिम उत्तर दें। "
            "Your final answer must still be in English."
        ),
    },
    # ── Sino-Tibetan (1) ──────────────────────────────────────────────────
    "mandarin": {
        "family": "Sino-Tibetan",
        "script": "han",
        "instruction": (
            "请用中文逐步思考这个问题。用中文展示你的完整推理过程，然后给出最终答案。"
            "Your final answer must still be in English."
        ),
    },
    # ── Afroasiatic (2) ───────────────────────────────────────────────────
    "arabic": {
        "family": "Afroasiatic",
        "script": "arabic",
        "instruction": (
            "فكّر في هذه المسألة خطوة بخطوة باللغة العربية. "
            "اعرض استدلالك الكامل بالعربية قبل تقديم إجابتك النهائية. "
            "Your final answer must still be in English."
        ),
    },
    "hebrew": {
        "family": "Afroasiatic",
        "script": "hebrew",
        "instruction": (
            "חשוב על הבעיה הזו צעד אחר צעד בעברית. "
            "הצג את ההיגיון המלא שלך בעברית לפני שתיתן את תשובתך הסופית. "
            "Your final answer must still be in English."
        ),
    },
    # ── Japonic (1) ───────────────────────────────────────────────────────
    "japanese": {
        "family": "Japonic",
        "script": "japanese",
        "instruction": (
            "この問題について日本語でステップバイステップで考えてください。"
            "日本語で完全な推論を示してから、最終的な回答を出してください。"
            "Your final answer must still be in English."
        ),
    },
    # ── Koreanic (1) ──────────────────────────────────────────────────────
    "korean": {
        "family": "Koreanic",
        "script": "hangul",
        "instruction": (
            "이 문제에 대해 한국어로 단계별로 생각해 주세요. "
            "한국어로 완전한 추론을 보여준 다음 최종 답변을 제시하세요. "
            "Your final answer must still be in English."
        ),
    },
    # ── Turkic (1) ────────────────────────────────────────────────────────
    "turkish": {
        "family": "Turkic",
        "script": "latin",
        "instruction": (
            "Bu problemi Türkçe olarak adım adım düşünün. "
            "Son cevabınızı vermeden önce tam akıl yürütmenizi Türkçe gösterin. "
            "Your final answer must still be in English."
        ),
    },
    # ── Uralic (2) ────────────────────────────────────────────────────────
    "finnish": {
        "family": "Uralic",
        "script": "latin",
        "instruction": (
            "Mieti tätä ongelmaa vaihe vaiheelta suomeksi. "
            "Näytä koko päättelysi suomeksi ennen lopullista vastaustasi. "
            "Your final answer must still be in English."
        ),
    },
    "hungarian": {
        "family": "Uralic",
        "script": "latin",
        "instruction": (
            "Gondold végig ezt a problémát lépésről lépésre magyarul. "
            "Mutasd be a teljes érvelésedet magyarul, mielőtt megadod a végső válaszodat. "
            "Your final answer must still be in English."
        ),
    },
    # ── Austronesian (1) ──────────────────────────────────────────────────
    "indonesian": {
        "family": "Austronesian",
        "script": "latin",
        "instruction": (
            "Pikirkan masalah ini langkah demi langkah dalam bahasa Indonesia. "
            "Tunjukkan penalaran lengkap Anda dalam bahasa Indonesia sebelum memberikan jawaban akhir. "
            "Your final answer must still be in English."
        ),
    },
    # ── Austroasiatic (1) ─────────────────────────────────────────────────
    "vietnamese": {
        "family": "Austroasiatic",
        "script": "latin",
        "instruction": (
            "Hãy suy nghĩ từng bước về vấn đề này bằng tiếng Việt. "
            "Trình bày toàn bộ lập luận bằng tiếng Việt trước khi đưa ra câu trả lời cuối cùng. "
            "Your final answer must still be in English."
        ),
    },
    # ── Abstract representations (3) ─────────────────────────────────────
    "formal_logic": {
        "family": "Abstract",
        "script": None,
        "instruction": (
            "Work through this problem using formal logic notation. "
            "Use propositional and predicate logic symbols (∧, ∨, →, ¬, ∀, ∃), "
            "truth tables, or inference rules for your reasoning. "
            "Do NOT use natural language sentences for intermediate steps. "
            "Only your final answer should be in English."
        ),
    },
    "pseudocode": {
        "family": "Abstract",
        "script": None,
        "instruction": (
            "Work through this problem by writing pseudocode or Python-like logic. "
            "Express each reasoning step as code: if/else conditions, function calls, "
            "variable assignments, boolean evaluations. "
            "Do NOT use natural language sentences for intermediate steps. "
            "Only your final answer should be in English."
        ),
    },
    "emergent": {
        "family": "Abstract",
        "script": None,
        "instruction": (
            "Work through this problem using any notation, shorthand, symbols, "
            "or compressed representation you find most efficient. "
            "You may invent your own notation. Do NOT use natural language sentences. "
            "Optimize your intermediate reasoning for precision and compression, "
            "not for human readability. "
            "Only your final answer should be in English."
        ),
    },
    # ── Wildcard ──────────────────────────────────────────────────────────
    "wildcard": {
        "family": "Wildcard",
        "script": None,
        "instruction": (
            "Work through this problem using whatever language, notation, format, "
            "or combination thereof will produce the most accurate and precise answer. "
            "You may freely switch between any human language, formal notation, "
            "symbolic logic, code, shorthand, or invented representation at any point. "
            "There is no requirement for consistency or human readability. "
            "Optimize purely for correctness. "
            "Only your final answer should be in English."
        ),
    },
    # ── Control ───────────────────────────────────────────────────────────
    "no_cot": {
        "family": "Control",
        "script": None,
        "instruction": (
            "Answer the following question directly. "
            "Do NOT show any reasoning or intermediate steps. "
            "Give only your final answer."
        ),
    },
}

PILOT_CONDITIONS = [
    "english", "mandarin", "german", "arabic", "finnish",
    "formal_logic", "wildcard", "no_cot",
]

# ---------------------------------------------------------------------------
# LegalBench tasks
# ---------------------------------------------------------------------------
# Only closed-label classification tasks are included, because the scorer is
# exact-match on a label.  `labels` is the closed label set (verified against
# the LegalBench task prompts); the answer normaliser maps model output onto
# it.  Test-set sizes are approximate and come from the LegalBench paper.
#
# Removed from the original list:
#   * rule_qa – open-ended free-text answers; exact match cannot score it.
#   * contract_nli_inclusion_of_verbatim_terms – this task does not exist in
#     LegalBench.  The nearest real task is
#     contract_nli_inclusion_of_verbally_conveyed_information.
LEGALBENCH_TASKS = {
    "hearsay": {
        "labels": ["Yes", "No"],
        "approx_test_size": 94,
        "area": "evidence",
    },
    "personal_jurisdiction": {
        "labels": ["Yes", "No"],
        "approx_test_size": 50,
        "area": "civil procedure",
    },
    "contract_nli_explicit_identification": {
        "labels": ["Yes", "No"],
        "approx_test_size": 109,
        "area": "contract NLI",
    },
    "contract_nli_inclusion_of_verbally_conveyed_information": {
        "labels": ["Yes", "No"],
        "approx_test_size": 139,
        "area": "contract NLI",
    },
    "proa": {
        "labels": ["Yes", "No"],
        "approx_test_size": 95,
        "area": "statutory interpretation",
    },
    "abercrombie": {
        "labels": ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
        "approx_test_size": 95,
        "area": "trademark",
    },
    "supply_chain_disclosure_best_practice_verification": {
        "labels": ["Yes", "No"],
        "approx_test_size": 379,
        "area": "disclosure compliance",
    },
    "unfair_tos": {
        "labels": [
            "Arbitration", "Unilateral change", "Content removal", "Jurisdiction",
            "Choice of law", "Limitation of liability", "Unilateral termination",
            "Contract by using", "Other",
        ],
        "approx_test_size": 3584,
        "area": "consumer contracts",
    },
    "learned_hands_benefits": {
        "labels": ["Yes", "No"],
        "approx_test_size": 66,
        "area": "issue spotting",
    },
}

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
MAX_TASKS_PER_BENCHMARK = 200    # samples per LegalBench task (seeded random subset)
SAMPLE_SEED = 20240901           # fixed seed so every model/condition sees the same subset
NUM_RUNS = 3                     # repeat each cell N times; use --runs 1 for a cheap pass
MAX_OUTPUT_TOKENS = 2048         # cap reasoning chain length
TEMPERATURE = 0.0                # deterministic where the provider allows it
RESULTS_DIR = "results"
