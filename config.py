"""
Experiment Configuration
========================
Defines all 21 reasoning language conditions, 6 models, and selected LegalBench tasks.
"""

# ---------------------------------------------------------------------------
# Models – each entry carries the provider key used by the provider adapter
# ---------------------------------------------------------------------------
MODELS = {
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "display": "Claude Sonnet 4 (Anthropic)",
        "origin_country": "USA",
    },
    "gpt-4o": {
        "provider": "azure_openai",
        "model_id": "gpt-4o",
        "display": "GPT-4o (OpenAI via Azure)",
        "origin_country": "USA",
    },
    "mistral-large": {
        "provider": "azure_mistral",
        "model_id": "mistral-large-latest",
        "display": "Mistral Large (Mistral AI via Azure)",
        "origin_country": "France",
    },
    "deepseek-v3": {
        "provider": "azure_deepseek",
        "model_id": "deepseek-v3",
        "display": "DeepSeek V3 (via Azure)",
        "origin_country": "China",
    },
    "qwen-max": {
        "provider": "qwen",           # user hooks in manually
        "model_id": "qwen-max",
        "display": "Qwen Max (Alibaba)",
        "origin_country": "China",
    },
    "gemini-2.5-flash": {
        "provider": "gemini",          # user hooks in manually
        "model_id": "gemini-2.5-flash",
        "display": "Gemini 2.5 Flash (Google)",
        "origin_country": "USA",
    },
}

# ---------------------------------------------------------------------------
# Language conditions – 16 natural languages + 3 abstract + 1 wildcard + 1 no-CoT
# ---------------------------------------------------------------------------
CONDITIONS = {
    # ── Indo-European (4) ──────────────────────────────────────────────────
    "english": {
        "family": "Indo-European",
        "instruction": (
            "Think through this problem step by step in English. "
            "Show your full reasoning in English before giving your final answer."
        ),
    },
    "german": {
        "family": "Indo-European",
        "instruction": (
            "Denke Schritt für Schritt auf Deutsch über dieses Problem nach. "
            "Zeige deine vollständige Argumentation auf Deutsch, bevor du deine endgültige Antwort gibst. "
            "Your final answer must still be in English."
        ),
    },
    "russian": {
        "family": "Indo-European",
        "instruction": (
            "Продумай эту задачу шаг за шагом на русском языке. "
            "Покажи полный ход рассуждений на русском, прежде чем дать окончательный ответ. "
            "Your final answer must still be in English."
        ),
    },
    "hindi": {
        "family": "Indo-European",
        "instruction": (
            "इस समस्या पर हिंदी में चरणबद्ध तरीके से विचार करें। "
            "अपना पूरा तर्क हिंदी में दिखाएं, फिर अपना अंतिम उत्तर दें। "
            "Your final answer must still be in English."
        ),
    },
    # ── Sino-Tibetan (1) ──────────────────────────────────────────────────
    "mandarin": {
        "family": "Sino-Tibetan",
        "instruction": (
            "请用中文逐步思考这个问题。用中文展示你的完整推理过程，然后给出最终答案。"
            "Your final answer must still be in English."
        ),
    },
    # ── Afroasiatic (2) ───────────────────────────────────────────────────
    "arabic": {
        "family": "Afroasiatic",
        "instruction": (
            "فكّر في هذه المسألة خطوة بخطوة باللغة العربية. "
            "اعرض استدلالك الكامل بالعربية قبل تقديم إجابتك النهائية. "
            "Your final answer must still be in English."
        ),
    },
    "hebrew": {
        "family": "Afroasiatic",
        "instruction": (
            "חשוב על הבעיה הזו צעד אחר צעד בעברית. "
            "הצג את ההיגיון המלא שלך בעברית לפני שתיתן את תשובתך הסופית. "
            "Your final answer must still be in English."
        ),
    },
    # ── Japonic (1) ───────────────────────────────────────────────────────
    "japanese": {
        "family": "Japonic",
        "instruction": (
            "この問題について日本語でステップバイステップで考えてください。"
            "日本語で完全な推論を示してから、最終的な回答を出してください。"
            "Your final answer must still be in English."
        ),
    },
    # ── Koreanic (1) ──────────────────────────────────────────────────────
    "korean": {
        "family": "Koreanic",
        "instruction": (
            "이 문제에 대해 한국어로 단계별로 생각해 주세요. "
            "한국어로 완전한 추론을 보여준 다음 최종 답변을 제시하세요. "
            "Your final answer must still be in English."
        ),
    },
    # ── Turkic (1) ────────────────────────────────────────────────────────
    "turkish": {
        "family": "Turkic",
        "instruction": (
            "Bu problemi Türkçe olarak adım adım düşünün. "
            "Son cevabınızı vermeden önce tam akıl yürütmenizi Türkçe gösterin. "
            "Your final answer must still be in English."
        ),
    },
    # ── Uralic (2) ────────────────────────────────────────────────────────
    "finnish": {
        "family": "Uralic",
        "instruction": (
            "Mieti tätä ongelmaa vaihe vaiheelta suomeksi. "
            "Näytä koko päättelysi suomeksi ennen lopullista vastaustasi. "
            "Your final answer must still be in English."
        ),
    },
    "hungarian": {
        "family": "Uralic",
        "instruction": (
            "Gondold végig ezt a problémát lépésről lépésre magyarul. "
            "Mutasd be a teljes érvelésedet magyarul, mielőtt megadod a végső válaszodat. "
            "Your final answer must still be in English."
        ),
    },
    # ── Austronesian (1) ──────────────────────────────────────────────────
    "indonesian": {
        "family": "Austronesian",
        "instruction": (
            "Pikirkan masalah ini langkah demi langkah dalam bahasa Indonesia. "
            "Tunjukkan penalaran lengkap Anda dalam bahasa Indonesia sebelum memberikan jawaban akhir. "
            "Your final answer must still be in English."
        ),
    },
    # ── Austroasiatic (1) ─────────────────────────────────────────────────
    "vietnamese": {
        "family": "Austroasiatic",
        "instruction": (
            "Hãy suy nghĩ từng bước về vấn đề này bằng tiếng Việt. "
            "Trình bày toàn bộ lập luận bằng tiếng Việt trước khi đưa ra câu trả lời cuối cùng. "
            "Your final answer must still be in English."
        ),
    },
    # ── Abstract representations (3) ─────────────────────────────────────
    "formal_logic": {
        "family": "Abstract",
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
        "instruction": (
            "Answer the following question directly. "
            "Do NOT show any reasoning or intermediate steps. "
            "Give only your final answer."
        ),
    },
}

# ---------------------------------------------------------------------------
# LegalBench tasks – selected for reasoning intensity & clean binary signal
# ---------------------------------------------------------------------------
LEGALBENCH_TASKS = [
    "hearsay",                       # evidence rule application
    "personal_jurisdiction",         # constitutional law reasoning
    "rule_qa",                       # rule comprehension
    "contract_nli_explicit_identification",   # contract NLI
    "contract_nli_inclusion_of_verbatim_terms",
    "proa",                          # statutory interpretation
    "abercrombie",                    # trademark classification
    "supply_chain_disclosure_best_practice_verification",
    "unfair_tos",                    # consumer contract analysis
    "learned_hands_benefits",        # legal issue spotting
]

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
MAX_TASKS_PER_BENCHMARK = 200    # samples per LegalBench task per condition
NUM_RUNS = 3                     # repeat each cell N times for variance
MAX_OUTPUT_TOKENS = 2048         # cap reasoning chain length
TEMPERATURE = 0.0                # deterministic where possible
RESULTS_DIR = "results"
