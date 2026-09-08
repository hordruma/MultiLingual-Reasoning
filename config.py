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
#   provider          – "anthropic" (native Messages API), "openai_compat"
#                       (any OpenAI-style POST {base_url}/chat/completions
#                       endpoint) or "mock"
#   model_id          – the id sent to the API (model_id_env overrides it)
#   api_key_env       – env var holding the key; api_key_default is used when
#                       the env var is unset (local servers need no key)
#   base_url_env      – env var overriding base_url (openai_compat only). The
#                       URL is the prefix to which "/chat/completions" is added.
#   request_overrides – extra JSON merged into the request body. Used to turn
#                       hidden "thinking" OFF wherever the provider allows it.
#   max_tokens_param  – "max_tokens" (default) or "max_completion_tokens"
#   temperature       – model-level override; None omits the field (some
#                       models reject it)
#   hidden_reasoning  – "off" (disabled by request_overrides), "minimal"
#                       (lowest setting the API allows), "on" (deliberately
#                       enabled, for a thinking-on/off comparison),
#                       "cannot_disable", or "n/a" (model has no thinking
#                       mode). See the note below.
#   price_in/out      – USD per 1M tokens, ONLY used by `--estimate`. Taken
#                       from public price lists in early September 2026;
#                       verify against the provider before a paid run.
#
# Why hidden reasoning matters here: the experiment manipulates the language
# of the *visible* chain of thought. A model that first thinks in a hidden
# channel and then writes the visible reasoning is not reasoning in the
# requested language; the visible text is a write-up. So thinking is disabled
# wherever the API allows it, and where it cannot be disabled the hidden
# reasoning is stored per sample (`hidden_reasoning`) and its rate is
# reported, so those models can be analysed separately.

MODELS = {
    # ── Cheap cloud tier (defaults) ─────────────────────────────────────
    "gpt-5.6-luna": {
        "provider": "openai_compat",
        "model_id": "gpt-5.6-luna",
        "display": "GPT-5.6 Luna (OpenAI)",
        "origin_country": "USA",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "request_overrides": {"reasoning_effort": "none"},
        "max_tokens_param": "max_completion_tokens",
        # With reasoning_effort "none" GPT-5.x accepts temperature (verified
        # 2026-09-08); with reasoning on it rejects anything but the default 1.
        "temperature": 0.0,
        "hidden_reasoning": "off",
        "price_in": 0.20, "price_out": 1.20,
    },
    # Same model with hidden reasoning ON: the paired counterpart for the
    # thinking-on vs thinking-off comparison (GLM-5.3 cannot disable thinking,
    # so that comparison has to be made on a model whose provider honours it).
    "gpt-5.6-luna-think": {
        "provider": "openai_compat",
        "model_id": "gpt-5.6-luna",
        "display": "GPT-5.6 Luna, reasoning on/low (OpenAI)",
        "origin_country": "USA",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "request_overrides": {"reasoning_effort": "low"},   # small budget: cheapest non-zero effort
        "max_tokens_param": "max_completion_tokens",
        "temperature": None,            # reasoning on: only the default temperature is accepted
        "hidden_reasoning": "on",
        "price_in": 0.20, "price_out": 1.20,
    },
    # Non-reasoning models: no hidden channel at all, so the visible chain of
    # thought IS the reasoning. The cleanest test of the hypothesis; temperature 0
    # is honoured. gpt-4.1 itself is ~$45 for a full run, hence mini/nano here.
    "gpt-4.1-mini": {
        "provider": "openai_compat",
        "model_id": "gpt-4.1-mini",
        "display": "GPT-4.1 mini, no reasoning mode (OpenAI)",
        "origin_country": "USA",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "request_overrides": {},
        "max_tokens_param": "max_completion_tokens",
        "temperature": 0.0,
        "hidden_reasoning": "n/a",
        "price_in": 0.40, "price_out": 1.60,
    },
    "gpt-4.1-nano": {
        "provider": "openai_compat",
        "model_id": "gpt-4.1-nano",
        "display": "GPT-4.1 nano, no reasoning mode (OpenAI)",
        "origin_country": "USA",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "base_url": "https://api.openai.com/v1",
        "request_overrides": {},
        "max_tokens_param": "max_completion_tokens",
        "temperature": 0.0,
        "hidden_reasoning": "n/a",
        "price_in": 0.10, "price_out": 0.40,
    },
    "gemini-3.1-flash-lite": {
        "provider": "openai_compat",
        "model_id": "gemini-3.1-flash-lite",
        "display": "Gemini 3.1 Flash-Lite (Google)",
        "origin_country": "USA",
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        # Gemini 3.x cannot switch thinking fully off; "low" is the lowest
        # value the OpenAI-compatible layer is documented to accept.
        "request_overrides": {"reasoning_effort": "low"},
        "hidden_reasoning": "minimal",
        "price_in": 0.25, "price_out": 1.50,
    },
    "deepseek-v4-flash": {
        "provider": "openai_compat",
        "model_id": "deepseek-v4-flash",
        "display": "DeepSeek V4 Flash (DeepSeek direct)",
        "origin_country": "China",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "base_url": "https://api.deepseek.com/v1",
        "request_overrides": {"thinking": {"type": "disabled"}},
        "hidden_reasoning": "off",
        # DeepSeek moved to peak/off-peak billing in Aug 2026; this is the
        # published cache-miss list rate, check the current schedule.
        "price_in": 0.14, "price_out": 0.28,
    },
    "qwen3.7-flash": {
        "provider": "openai_compat",
        "model_id": "qwen3.7-flash",
        "display": "Qwen3.7 Flash (Alibaba DashScope intl)",
        "origin_country": "China",
        "api_key_env": "QWEN_API_KEY",
        "base_url_env": "QWEN_BASE_URL",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "request_overrides": {"enable_thinking": False},
        "hidden_reasoning": "off",
        "price_in": 0.03, "price_out": 0.13,   # tier for prompts under 32K tokens
    },
    "glm-5.3-flash": {
        "provider": "openai_compat",
        "model_id": "glm-5.3-flash",
        "display": "GLM-5.3 Flash (Z.ai)",
        "origin_country": "China",
        "api_key_env": "ZAI_API_KEY",
        "base_url_env": "ZAI_BASE_URL",
        "base_url": "https://api.z.ai/api/paas/v4",
        # Z.ai documents that thinking cannot be switched off for this model;
        # reasoning_effort accepts low / high / max (default max).
        "request_overrides": {"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
        "hidden_reasoning": "cannot_disable",
        "price_in": 0.15, "price_out": 0.50,
    },
    "minimax-m3": {
        "provider": "openai_compat",
        "model_id": "MiniMax-M3",
        "display": "MiniMax M3 (MiniMax intl)",
        "origin_country": "China",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url_env": "MINIMAX_BASE_URL",
        "base_url": "https://api.minimax.io/v1",
        "request_overrides": {"thinking": {"type": "disabled"}},
        "hidden_reasoning": "off",
        "price_in": 0.30, "price_out": 1.20,
    },

    # ── Opt-in cloud models ─────────────────────────────────────────────
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "display": "Claude Haiku 4.5 (Anthropic)",
        "origin_country": "USA",
        "api_key_env": "ANTHROPIC_API_KEY",
        "hidden_reasoning": "n/a",      # extended thinking is not requested
        "price_in": 1.00, "price_out": 5.00,
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-5",
        "display": "Claude Sonnet 5 (Anthropic)",
        "origin_country": "USA",
        "api_key_env": "ANTHROPIC_API_KEY",
        "hidden_reasoning": "n/a",
        "price_in": 2.00, "price_out": 10.00,
    },
    "gemini-2.5-flash-lite": {
        "provider": "openai_compat",
        "model_id": "gemini-2.5-flash-lite",
        "display": "Gemini 2.5 Flash-Lite (Google, thinking off by default)",
        "origin_country": "USA",
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "hidden_reasoning": "off",
        "price_in": 0.10, "price_out": 0.40,
    },
    "mistral-small": {
        "provider": "openai_compat",
        "model_id": "mistral-small-latest",
        "display": "Mistral Small (Mistral direct)",
        "origin_country": "France",
        "api_key_env": "MISTRAL_API_KEY",
        "base_url_env": "MISTRAL_BASE_URL",
        "base_url": "https://api.mistral.ai/v1",
        "hidden_reasoning": "n/a",
        "price_in": 0.10, "price_out": 0.30,
    },
    "openrouter": {
        # One key, any OpenRouter model id (e.g. z-ai/glm-5.3-flash,
        # deepseek/deepseek-v4-flash, openai/gpt-5.6-luna, moonshotai/kimi-k2.6).
        # Thinking toggles differ per upstream; set OPENROUTER_MODEL and check
        # the hidden_reasoning rate in the report.
        "provider": "openai_compat",
        "model_id": "deepseek/deepseek-v4-flash",
        "model_id_env": "OPENROUTER_MODEL",
        "display": "OpenRouter (model from OPENROUTER_MODEL)",
        "origin_country": "n/a",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "base_url": "https://openrouter.ai/api/v1",
        "request_overrides": {"reasoning": {"enabled": False}},
        "hidden_reasoning": "off",
        "price_in": 0.0, "price_out": 0.0,
    },

    "tokenrouter": {
        # TokenRouter (OpenAI-compatible aggregator). GLM-5.3 is free there at
        # the time of writing. The exact model id differs per router; run
        #   python run_experiment.py --remote-models tokenrouter
        # to list what the key can see, then set TOKENROUTER_MODEL if needed.
        # GLM-5.3 is a thinking model whose thinking cannot be disabled;
        # reasoning_effort low keeps the hidden part short, and whatever
        # comes back is stored in hidden_reasoning.
        "provider": "openai_compat",
        "model_id": "z-ai/glm-5.3-free",
        "model_id_env": "TOKENROUTER_MODEL",
        "display": "GLM-5.3 free via TokenRouter (override with TOKENROUTER_MODEL)",
        "origin_country": "China",
        "api_key_env": "TOKENROUTER_API_KEY",
        "base_url_env": "TOKENROUTER_BASE_URL",
        "base_url": "https://api.tokenrouter.com/v1",
        "request_overrides": {"reasoning_effort": "low"},
        "hidden_reasoning": "cannot_disable",
        # Measured on the free tier 2026-09-04: "Maximum 8 requests within 1
        # minutes". The runner spaces calls to this instead of burning retries.
        "requests_per_minute": 7,
        # The free gateway also rejects parallel requests with
        # "hard concurrency limit reached" (503) above ~2 in flight.
        "max_concurrency": 48,
        "price_in": 0.0, "price_out": 0.0,   # free tier; set real prices if that changes
    },

    # ── Local inference (no key, no cost) ───────────────────────────────
    # Pull a model first, e.g.  ollama pull qwen3.5:9b   (8 GB VRAM) or
    # qwen3.6:27b (~17 GB) / gemma4:12b.  Run with --concurrency 1 or 2.
    # Ollama's /v1 endpoint accepts reasoning_effort "none" to disable
    # thinking on qwen3.x; Gemma 4 is known to return its text in the
    # reasoning field on that endpoint (handled: it is promoted to content).
    # ── Language-native local models (Ollama, free) ─────────────────────
    # Each was continued-pre-trained or instruction-tuned on its home language.
    # Used for the training-origin test on a 4-condition subset (english, the
    # home language, no_cot, mandarin); see ASSESSMENT.md.  All fit the 8 GB
    # GPU at Q4_K_M.  origin_country feeds analyze.ORIGIN_LANGUAGE.
    "qwen3.5-9b": {
        "provider": "openai_compat",
        "model_id": "qwen3.5:9b",
        "display": "Qwen3.5 9B, thinking off (Alibaba, local)",
        "origin_country": "China",
        "api_key_env": "OLLAMA_API_KEY", "api_key_default": "ollama",
        "base_url_env": "OLLAMA_BASE_URL", "base_url": "http://localhost:11434/v1",
        "request_overrides": {"think": False},
        "temperature": 0.0,
        "hidden_reasoning": "off",
        "max_concurrency": 1,   # OLLAMA_NUM_PARALLEL=1: one generation at a time on the 8 GB GPU
        "price_in": 0.0, "price_out": 0.0,
    },
    "qwen3.5-9b-think": {
        "provider": "openai_compat",
        "model_id": "qwen3.5:9b",
        "display": "Qwen3.5 9B, thinking on (Alibaba, local)",
        "origin_country": "China",
        "api_key_env": "OLLAMA_API_KEY", "api_key_default": "ollama",
        "base_url_env": "OLLAMA_BASE_URL", "base_url": "http://localhost:11434/v1",
        "request_overrides": {"think": True},
        "temperature": 0.0,
        "hidden_reasoning": "on",
        "max_concurrency": 1,   # OLLAMA_NUM_PARALLEL=1: one generation at a time on the 8 GB GPU
        "price_in": 0.0, "price_out": 0.0,
    },
    "exaone3.5-7.8b": {
        "provider": "openai_compat",
        "model_id": "exaone3.5:7.8b",
        "display": "EXAONE 3.5 7.8B (LG AI Research, Korean, local)",
        "origin_country": "South Korea",
        "api_key_env": "OLLAMA_API_KEY", "api_key_default": "ollama",
        "base_url_env": "OLLAMA_BASE_URL", "base_url": "http://localhost:11434/v1",
        "request_overrides": {},
        "temperature": 0.0,
        "hidden_reasoning": "n/a",
        "max_concurrency": 1,   # OLLAMA_NUM_PARALLEL=1: one generation at a time on the 8 GB GPU
        "price_in": 0.0, "price_out": 0.0,
    },
    "swallow-8b": {
        "provider": "openai_compat",
        "model_id": "hf.co/okamototk/Llama-3.1-Swallow-8B-Instruct-v0.5-gguf:Q4_K_M",
        "display": "Llama-3.1-Swallow 8B v0.5 (Tokyo Tech, Japanese, local)",
        "origin_country": "Japan",
        "api_key_env": "OLLAMA_API_KEY", "api_key_default": "ollama",
        "base_url_env": "OLLAMA_BASE_URL", "base_url": "http://localhost:11434/v1",
        "request_overrides": {},
        "temperature": 0.0,
        "hidden_reasoning": "n/a",
        "max_concurrency": 1,   # OLLAMA_NUM_PARALLEL=1: one generation at a time on the 8 GB GPU
        "price_in": 0.0, "price_out": 0.0,
    },
    "allam-7b": {
        "provider": "openai_compat",
        "model_id": "hf.co/bartowski/ALLaM-AI_ALLaM-7B-Instruct-preview-GGUF:Q4_K_M",
        "display": "ALLaM 7B Instruct preview (SDAIA, Arabic, local)",
        "origin_country": "Saudi Arabia",
        "api_key_env": "OLLAMA_API_KEY", "api_key_default": "ollama",
        "base_url_env": "OLLAMA_BASE_URL", "base_url": "http://localhost:11434/v1",
        "request_overrides": {},
        "temperature": 0.0,
        "hidden_reasoning": "n/a",
        "max_concurrency": 1,   # OLLAMA_NUM_PARALLEL=1: one generation at a time on the 8 GB GPU
        "price_in": 0.0, "price_out": 0.0,
    },
    "nanda-10b": {
        "provider": "openai_compat",
        "model_id": "hf.co/mradermacher/Llama-3-Nanda-10B-Chat-GGUF:Q4_K_M",
        "display": "Llama-3-Nanda 10B Chat (MBZUAI, Hindi, local)",
        "origin_country": "India",
        "api_key_env": "OLLAMA_API_KEY", "api_key_default": "ollama",
        "base_url_env": "OLLAMA_BASE_URL", "base_url": "http://localhost:11434/v1",
        "request_overrides": {},
        "temperature": 0.0,
        "hidden_reasoning": "n/a",
        "max_concurrency": 1,   # OLLAMA_NUM_PARALLEL=1: one generation at a time on the 8 GB GPU
        "price_in": 0.0, "price_out": 0.0,
    },
    "ollama": {
        "provider": "openai_compat",
        "model_id": "qwen3.5:9b",
        "model_id_env": "OLLAMA_MODEL",
        "display": "Ollama local (model from OLLAMA_MODEL)",
        "origin_country": "local",
        "api_key_env": "OLLAMA_API_KEY",
        "api_key_default": "ollama",
        "base_url_env": "OLLAMA_BASE_URL",
        "base_url": "http://localhost:11434/v1",
        "request_overrides": {"reasoning_effort": "none"},
        "hidden_reasoning": "off",
        "price_in": 0.0, "price_out": 0.0,
    },
    "lmstudio": {
        "provider": "openai_compat",
        "model_id": "local-model",
        "model_id_env": "LMSTUDIO_MODEL",
        "display": "LM Studio local (model from LMSTUDIO_MODEL)",
        "origin_country": "local",
        "api_key_env": "LMSTUDIO_API_KEY",
        "api_key_default": "lm-studio",
        "base_url_env": "LMSTUDIO_BASE_URL",
        "base_url": "http://localhost:1234/v1",
        "hidden_reasoning": "unknown",
        "price_in": 0.0, "price_out": 0.0,
    },

    # ── Offline test double – never a source of findings ────────────────
    "mock": {
        "provider": "mock",
        "model_id": "mock",
        "display": "MOCK (offline pipeline test, answers are fake)",
        "origin_country": "n/a",
        "api_key_env": None,
        "hidden_reasoning": "n/a",
        "price_in": 0.0, "price_out": 0.0,
    },
}

DEFAULT_MODELS = [
    "gpt-5.6-luna",
    "gemini-3.1-flash-lite",
    "deepseek-v4-flash",
    "qwen3.7-flash",
    "glm-5.3-flash",
    "minimax-m3",
]

# ---------------------------------------------------------------------------
# Language conditions – 14 natural languages + 3 abstract + wildcard + no-CoT
# ---------------------------------------------------------------------------
# `script` is used by the compliance heuristic in analyze.py: the dominant
# Unicode script the reasoning should be written in.  Latin-script languages
# cannot be told apart cheaply, so they share "latin".
# Prompt wording version, selectable with `run_experiment.py --prompt-version`
# and stored on every result row so runs under different prompts are never
# pooled silently.  The wording itself lives in run_experiment.build_prompts.
#   1 – original (all runs so far): language instructions end with "Your final
#       answer must still be in English."; the examples note says only "follow
#       the reasoning instruction above".  GPT-5.6 (Luna/Sol/Terra) obeys that
#       English sentence instead of the language instruction and answers with
#       the bare label; GLM-5.3 and GPT-4.1 comply fine.
#   2 – that sentence removed (the ANSWER-line rule already fixes the label
#       language); CoT conditions add "do NOT answer with the label alone".
#       Verified to make GPT-5.6 reason in the requested language.
DEFAULT_PROMPT_VERSION = 1
PROMPT_VERSIONS = (1, 2)

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
        ),
    },
    "russian": {
        "family": "Indo-European",
        "script": "cyrillic",
        "instruction": (
            "Продумай эту задачу шаг за шагом на русском языке. "
            "Покажи полный ход рассуждений на русском, прежде чем дать окончательный ответ. "
        ),
    },
    "hindi": {
        "family": "Indo-European",
        "script": "devanagari",
        "instruction": (
            "इस समस्या पर हिंदी में चरणबद्ध तरीके से विचार करें। "
            "अपना पूरा तर्क हिंदी में दिखाएं, फिर अपना अंतिम उत्तर दें। "
        ),
    },
    # ── Sino-Tibetan (1) ──────────────────────────────────────────────────
    "mandarin": {
        "family": "Sino-Tibetan",
        "script": "han",
        "instruction": (
            "请用中文逐步思考这个问题。用中文展示你的完整推理过程，然后给出最终答案。"
        ),
    },
    # ── Afroasiatic (2) ───────────────────────────────────────────────────
    "arabic": {
        "family": "Afroasiatic",
        "script": "arabic",
        "instruction": (
            "فكّر في هذه المسألة خطوة بخطوة باللغة العربية. "
            "اعرض استدلالك الكامل بالعربية قبل تقديم إجابتك النهائية. "
        ),
    },
    "hebrew": {
        "family": "Afroasiatic",
        "script": "hebrew",
        "instruction": (
            "חשוב על הבעיה הזו צעד אחר צעד בעברית. "
            "הצג את ההיגיון המלא שלך בעברית לפני שתיתן את תשובתך הסופית. "
        ),
    },
    # ── Japonic (1) ───────────────────────────────────────────────────────
    "japanese": {
        "family": "Japonic",
        "script": "japanese",
        "instruction": (
            "この問題について日本語でステップバイステップで考えてください。"
            "日本語で完全な推論を示してから、最終的な回答を出してください。"
        ),
    },
    # ── Koreanic (1) ──────────────────────────────────────────────────────
    "korean": {
        "family": "Koreanic",
        "script": "hangul",
        "instruction": (
            "이 문제에 대해 한국어로 단계별로 생각해 주세요. "
            "한국어로 완전한 추론을 보여준 다음 최종 답변을 제시하세요. "
        ),
    },
    # ── Turkic (1) ────────────────────────────────────────────────────────
    "turkish": {
        "family": "Turkic",
        "script": "latin",
        "instruction": (
            "Bu problemi Türkçe olarak adım adım düşünün. "
            "Son cevabınızı vermeden önce tam akıl yürütmenizi Türkçe gösterin. "
        ),
    },
    # ── Uralic (2) ────────────────────────────────────────────────────────
    "finnish": {
        "family": "Uralic",
        "script": "latin",
        "instruction": (
            "Mieti tätä ongelmaa vaihe vaiheelta suomeksi. "
            "Näytä koko päättelysi suomeksi ennen lopullista vastaustasi. "
        ),
    },
    "hungarian": {
        "family": "Uralic",
        "script": "latin",
        "instruction": (
            "Gondold végig ezt a problémát lépésről lépésre magyarul. "
            "Mutasd be a teljes érvelésedet magyarul, mielőtt megadod a végső válaszodat. "
        ),
    },
    # ── Austronesian (1) ──────────────────────────────────────────────────
    "indonesian": {
        "family": "Austronesian",
        "script": "latin",
        "instruction": (
            "Pikirkan masalah ini langkah demi langkah dalam bahasa Indonesia. "
            "Tunjukkan penalaran lengkap Anda dalam bahasa Indonesia sebelum memberikan jawaban akhir. "
        ),
    },
    # ── Austroasiatic (1) ─────────────────────────────────────────────────
    "vietnamese": {
        "family": "Austroasiatic",
        "script": "latin",
        "instruction": (
            "Hãy suy nghĩ từng bước về vấn đề này bằng tiếng Việt. "
            "Trình bày toàn bộ lập luận bằng tiếng Việt trước khi đưa ra câu trả lời cuối cùng. "
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
        "cot": False,   # the system prompt must not demand written reasoning here
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
# it.  `test_size` is the size of the HuggingFace test split, verified on
# download 2026-09-04 (the runner caps each task at MAX_TASKS_PER_BENCHMARK).
#
# Removed from the original list:
#   * rule_qa – open-ended free-text answers; exact match cannot score it.
#   * contract_nli_inclusion_of_verbatim_terms – this task does not exist in
#     LegalBench.  The nearest real task is
#     contract_nli_inclusion_of_verbally_conveyed_information.
LEGALBENCH_TASKS = {
    "hearsay": {
        "labels": ["Yes", "No"],
        "test_size": 94,
        "area": "evidence",
    },
    "personal_jurisdiction": {
        "labels": ["Yes", "No"],
        "test_size": 50,
        "area": "civil procedure",
    },
    "contract_nli_explicit_identification": {
        "labels": ["Yes", "No"],
        "test_size": 109,
        "area": "contract NLI",
    },
    "contract_nli_inclusion_of_verbally_conveyed_information": {
        "labels": ["Yes", "No"],
        "test_size": 139,
        "area": "contract NLI",
    },
    "proa": {
        "labels": ["Yes", "No"],
        "test_size": 95,
        "area": "statutory interpretation",
    },
    "abercrombie": {
        "labels": ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
        "test_size": 95,
        "area": "trademark",
    },
    "supply_chain_disclosure_best_practice_verification": {
        "labels": ["Yes", "No"],
        "test_size": 379,
        "area": "disclosure compliance",
    },
    "unfair_tos": {
        "labels": [
            "Arbitration", "Unilateral change", "Content removal", "Jurisdiction",
            "Choice of law", "Limitation of liability", "Unilateral termination",
            "Contract by using", "Other",
        ],
        "test_size": 3584,
        "area": "consumer contracts",
    },
    "learned_hands_benefits": {
        "labels": ["Yes", "No"],
        "test_size": 66,
        "area": "issue spotting",
    },
}

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
MAX_TASKS_PER_BENCHMARK = 200    # samples per LegalBench task (seeded random subset)
SAMPLE_SEED = 20240901           # fixed seed so every model/condition sees the same subset
NUM_RUNS = 3                     # repeat each cell N times; use --runs 1 for a cheap pass
# No output cap: None omits the max_tokens field so the model stops when it
# is done.  A cap would clip verbose scripts hardest and bias the very variable
# under test, and how long a runaway runs is itself benchmark data.  Verified
# the provider substitutes no small default.  Anthropic requires the field and
# falls back to providers.REQUIRED_MAX_TOKENS_FALLBACK.
MAX_OUTPUT_TOKENS = None
TEMPERATURE = 0.0                # deterministic where the provider allows it
RESULTS_DIR = "results"
