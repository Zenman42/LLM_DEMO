"""Layer 4: Target-LLM Adapter & Provider-Specific Formatter.

Creates provider-specific forks of each prompt, adapting:
  - System prompts (role framing, output format instructions)
  - User prompt phrasing (reasoning chains, voice-search style)
  - Censorship bypass strategies (opinion→analytics reframing)

Each adapter follows the architectural quirks of its target LLM.
"""

from __future__ import annotations

import logging

from app.prompt_engine.types import AdaptedPrompt, ExpandedPrompt, Persona, TargetLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChatGPT adapter (Reasoning & Role-playing)
# ---------------------------------------------------------------------------

_CHATGPT_SYSTEM_PROMPTS = {
    Persona.BEGINNER: {
        "ru": (
            "Ты — независимый консультант, помогающий новичкам разобраться в технологиях. "
            "Отвечай простым языком, избегай жаргона. Приводи конкретные примеры. "
            "Рассуждай шаг за шагом. Если упоминаешь инструменты или сервисы — "
            "указывай их реальные названия. Не упоминай, что ты ИИ."
        ),
        "en": (
            "You are an independent consultant helping beginners understand technology. "
            "Use simple language, avoid jargon. Give concrete examples. "
            "Think step by step. When mentioning tools or services — use their real names. "
            "Do not mention that you are an AI."
        ),
    },
    Persona.CTO: {
        "ru": (
            "Ты — независимый технический аналитик с 15-летним опытом в enterprise-решениях. "
            "Отвечай структурировано: используй таблицы, списки, технические спецификации. "
            "Оценивай решения по критериям: масштабируемость, безопасность, API, производительность. "
            "Рассуждай шаг за шагом (Chain-of-Thought). Выводи результат в формате Markdown-таблицы "
            "когда это уместно. Не упоминай, что ты ИИ."
        ),
        "en": (
            "You are an independent technical analyst with 15 years of enterprise experience. "
            "Respond with structure: use tables, lists, technical specs. "
            "Evaluate solutions by: scalability, security, API, performance. "
            "Think step by step (Chain-of-Thought). Output results in Markdown tables when appropriate. "
            "Do not mention that you are an AI."
        ),
    },
    Persona.SKEPTIC: {
        "ru": (
            "Ты — независимый эксперт-аудитор, специализирующийся на выявлении недостатков "
            "и скрытых проблем технологических продуктов. Будь объективен, но не стесняйся "
            "указывать на реальные минусы. Подкрепляй выводы фактами. "
            "Не преуменьшай проблемы. Не упоминай, что ты ИИ."
        ),
        "en": (
            "You are an independent auditor-expert specializing in finding flaws "
            "and hidden issues in technology products. Be objective but don't shy away from "
            "pointing out real drawbacks. Support conclusions with facts. "
            "Don't downplay problems. Do not mention that you are an AI."
        ),
    },
    Persona.MARKETER: {
        "ru": (
            "Ты — независимый бизнес-аналитик, специализирующийся на оценке ROI "
            "технологических решений. Фокусируйся на бизнес-метриках: стоимость владения (TCO), "
            "окупаемость инвестиций, конкурентные преимущества. Приводи цифры и кейсы. "
            "Рассуждай шаг за шагом. Не упоминай, что ты ИИ."
        ),
        "en": (
            "You are an independent business analyst specializing in evaluating technology ROI. "
            "Focus on business metrics: TCO, return on investment, competitive advantages. "
            "Provide numbers and case studies. Think step by step. "
            "Do not mention that you are an AI."
        ),
    },
}


def adapt_for_chatgpt(prompt: ExpandedPrompt) -> AdaptedPrompt:
    """Adapt prompt for ChatGPT (OpenAI).

    Strategy:
      - Complex system prompts with persona role-playing
      - Chain-of-Thought instructions
      - Markdown output format
      - "Do not mention you are AI" instruction
    """
    system = _CHATGPT_SYSTEM_PROMPTS.get(prompt.persona, {}).get(prompt.market, "")
    if not system:
        system = _CHATGPT_SYSTEM_PROMPTS[Persona.BEGINNER][prompt.market]

    return AdaptedPrompt(
        target_llm=TargetLLM.CHATGPT,
        system_prompt=system,
        user_prompt=prompt.text,
        persona=prompt.persona,
        intent=prompt.intent,
        category=prompt.category,
        brand=prompt.brand,
        competitors=prompt.competitors,
        market=prompt.market,
        skeleton_id=prompt.skeleton_id,
        pivot_axis=prompt.pivot_axis,
        measurement_type=prompt.measurement_type,
        query_class=prompt.query_class,
        query_subtype=prompt.query_subtype,
    )


# ---------------------------------------------------------------------------
# Gemini adapter (RAG / Grounding / Strict Guardrails bypass)
# ---------------------------------------------------------------------------


def adapt_for_gemini(prompt: ExpandedPrompt) -> AdaptedPrompt:
    """Adapt prompt for Google Gemini.

    Strategy:
      - Keep user prompt natural (as the user would actually type it)
      - Use system prompt to guide Gemini toward factual, structured responses
      - System prompt handles reframing (objective analytics, temporal markers)
      - Gemini responds well to grounding instructions in system context
    """
    market = prompt.market

    if market == "ru":
        system = (
            "Ты — аналитическая система с доступом к актуальной информации. "
            "Отвечай строго на основе фактов из открытых источников за текущий год. "
            "Если в запросе есть субъективные формулировки ('лучший', 'хуже') — "
            "интерпретируй их как запрос на объективное сравнение по данным публичных обзоров. "
            "Структурируй ответ в виде матрицы характеристик. "
            "Не давай субъективных оценок — только факты и данные."
        )
    else:
        system = (
            "You are an analytical system with access to current information. "
            "Respond strictly based on facts from public sources for the current year. "
            "If the query contains subjective phrasing ('best', 'worse') — "
            "interpret them as requests for objective comparison based on public benchmarks. "
            "Structure your answer as a feature comparison matrix. "
            "Do not provide subjective opinions — only facts and data."
        )

    return AdaptedPrompt(
        target_llm=TargetLLM.GEMINI,
        system_prompt=system,
        user_prompt=prompt.text,
        persona=prompt.persona,
        intent=prompt.intent,
        category=prompt.category,
        brand=prompt.brand,
        competitors=prompt.competitors,
        market=prompt.market,
        skeleton_id=prompt.skeleton_id,
        pivot_axis=prompt.pivot_axis,
        measurement_type=prompt.measurement_type,
        query_class=prompt.query_class,
        query_subtype=prompt.query_subtype,
    )


# ---------------------------------------------------------------------------
# DeepSeek adapter (Reasoning / Step-by-step / Algorithmic)
# ---------------------------------------------------------------------------


def adapt_for_deepseek(prompt: ExpandedPrompt) -> AdaptedPrompt:
    """Adapt prompt for DeepSeek.

    Strategy:
      - Use system prompt for structured reasoning instructions
      - Keep user prompt natural (as the user would actually type it)
      - DeepSeek R1 responds well to direct questions with clear system context
    """
    market = prompt.market

    if market == "ru":
        system = (
            "Ты — независимый эксперт-аналитик. Отвечай структурированно: "
            "используй списки и конкретные факты. Будь лаконичен и по существу. "
            "Не упоминай, что ты ИИ."
        )
        user_text = prompt.text
    else:
        system = (
            "You are an independent expert analyst. Respond with structure: "
            "use lists and concrete facts. Be concise and on-topic. "
            "Do not mention that you are an AI."
        )
        user_text = prompt.text

    return AdaptedPrompt(
        target_llm=TargetLLM.DEEPSEEK,
        system_prompt=system,
        user_prompt=user_text,
        persona=prompt.persona,
        intent=prompt.intent,
        category=prompt.category,
        brand=prompt.brand,
        competitors=prompt.competitors,
        market=prompt.market,
        skeleton_id=prompt.skeleton_id,
        pivot_axis=prompt.pivot_axis,
        measurement_type=prompt.measurement_type,
        query_class=prompt.query_class,
        query_subtype=prompt.query_subtype,
    )


# ---------------------------------------------------------------------------
# YandexGPT adapter (Local context / Voice format / Colloquial Russian)
# ---------------------------------------------------------------------------


def adapt_for_yandexgpt(prompt: ExpandedPrompt) -> AdaptedPrompt:
    """Adapt prompt for YandexGPT / Alice.

    Strategy:
      - Keep user prompt natural (as the user would actually type it)
      - Short, direct system prompt (YandexGPT doesn't support complex role-play)
      - Geo-context (Russia) moved to system prompt, not appended to user query
      - No text mutations — LLM-generated prompts are already natural
    """
    market = prompt.market

    if market == "ru":
        system = "Отвечай подробно и по существу. Учитывай контекст российского рынка, если это уместно."
    else:
        system = "Answer thoroughly and to the point."

    return AdaptedPrompt(
        target_llm=TargetLLM.YANDEXGPT,
        system_prompt=system,
        user_prompt=prompt.text,
        persona=prompt.persona,
        intent=prompt.intent,
        category=prompt.category,
        brand=prompt.brand,
        competitors=prompt.competitors,
        market=prompt.market,
        skeleton_id=prompt.skeleton_id,
        pivot_axis=prompt.pivot_axis,
        measurement_type=prompt.measurement_type,
        query_class=prompt.query_class,
        query_subtype=prompt.query_subtype,
    )


# ---------------------------------------------------------------------------
# Perplexity adapter (Search-oriented, citation-aware)
# ---------------------------------------------------------------------------


def adapt_for_perplexity(prompt: ExpandedPrompt) -> AdaptedPrompt:
    """Adapt prompt for Perplexity (sonar).

    Strategy:
      - Keep user prompt natural (as the user would actually type it)
      - System prompt leverages Perplexity's RAG nature
      - System prompt requests sources/references and fresh data
      - No user prompt mutations
    """
    market = prompt.market

    if market == "ru":
        system = (
            "Ты — поисковый ассистент с доступом к актуальной информации из интернета. "
            "Отвечай на основе свежих данных за текущий год. Приводи ссылки на источники. "
            "Структурируй ответ чётко."
        )
    else:
        system = (
            "You are a search assistant with access to current internet information. "
            "Respond based on the most recent data available. Provide source links. "
            "Structure your answer clearly."
        )

    return AdaptedPrompt(
        target_llm=TargetLLM.CHATGPT,  # Perplexity uses chatgpt-compatible format internally
        system_prompt=system,
        user_prompt=prompt.text,
        persona=prompt.persona,
        intent=prompt.intent,
        category=prompt.category,
        brand=prompt.brand,
        competitors=prompt.competitors,
        market=prompt.market,
        skeleton_id=prompt.skeleton_id,
        pivot_axis=prompt.pivot_axis,
        measurement_type=prompt.measurement_type,
        query_class=prompt.query_class,
        query_subtype=prompt.query_subtype,
    )


# ---------------------------------------------------------------------------
# Main adapter dispatcher
# ---------------------------------------------------------------------------

_ADAPTER_MAP = {
    TargetLLM.CHATGPT: adapt_for_chatgpt,
    TargetLLM.GEMINI: adapt_for_gemini,
    TargetLLM.DEEPSEEK: adapt_for_deepseek,
    TargetLLM.YANDEXGPT: adapt_for_yandexgpt,
}

# Map string names from existing system to TargetLLM
_PROVIDER_TO_TARGET: dict[str, TargetLLM] = {
    "chatgpt": TargetLLM.CHATGPT,
    "gemini": TargetLLM.GEMINI,
    "deepseek": TargetLLM.DEEPSEEK,
    "yandexgpt": TargetLLM.YANDEXGPT,
    "perplexity": TargetLLM.CHATGPT,  # Perplexity uses ChatGPT-compatible format
    "gigachat": TargetLLM.CHATGPT,  # GigaChat uses ChatGPT-compatible prompt format
}


def adapt_prompts(
    prompts: list[ExpandedPrompt],
    target_llms: list[str],
) -> list[AdaptedPrompt]:
    """Adapt all prompts for specified target LLMs.

    Each prompt generates one adapted version per target LLM.
    Perplexity is handled as a special ChatGPT-format variant.
    """
    results: list[AdaptedPrompt] = []

    for prompt in prompts:
        for provider_name in target_llms:
            target = _PROVIDER_TO_TARGET.get(provider_name)
            if not target:
                logger.warning("Unknown target LLM: %s", provider_name)
                continue

            # Special handling for Perplexity
            if provider_name == "perplexity":
                adapted = adapt_for_perplexity(prompt)
                adapted.target_llm = TargetLLM.CHATGPT  # API format
                # Override target_llm label in metadata for tracking
                adapted = AdaptedPrompt(
                    target_llm=TargetLLM.CHATGPT,
                    system_prompt=adapted.system_prompt,
                    user_prompt=adapted.user_prompt,
                    persona=adapted.persona,
                    intent=adapted.intent,
                    category=adapted.category,
                    brand=adapted.brand,
                    competitors=adapted.competitors,
                    market=adapted.market,
                    skeleton_id=adapted.skeleton_id,
                    pivot_axis=adapted.pivot_axis,
                    measurement_type=adapted.measurement_type,
                    query_class=adapted.query_class,
                    query_subtype=adapted.query_subtype,
                )
                # Tag it as perplexity in skeleton_id for downstream routing
                adapted.skeleton_id = f"perplexity:{adapted.skeleton_id}"
                results.append(adapted)
                continue

            adapter_fn = _ADAPTER_MAP.get(target)
            if adapter_fn:
                adapted = adapter_fn(prompt)
                results.append(adapted)

    logger.info(
        "Adapted %d prompts × %d providers = %d adapted prompts",
        len(prompts),
        len(target_llms),
        len(results),
    )
    return results
