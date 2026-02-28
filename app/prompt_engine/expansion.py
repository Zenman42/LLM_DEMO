"""Layer 3: Semantic Expansion & Pivot Engine.

Takes rigid skeletons from the Matrix Engine and:
  1. Applies Semantic Pivots — substitutes quality adjectives
  2. Optionally calls a lightweight LLM to generate natural-language variations
  3. Falls back to rule-based expansion when LLM is unavailable
"""

from __future__ import annotations

import logging
import re

from app.prompt_engine.types import AuditProfile, ExpandedPrompt, PromptSkeleton

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Semantic Pivot dictionaries — adjective substitutions
# ---------------------------------------------------------------------------

_PIVOTS_RU: dict[str, list[str]] = {
    "лучший": [
        "самый надёжный",
        "самый бюджетный",
        "самый простой в освоении",
        "самый функциональный",
        "самый популярный",
    ],
    "лучшее": ["самое надёжное", "самое бюджетное", "самое простое в освоении", "самое функциональное"],
    "лучшую": ["самую надёжную", "самую бюджетную", "самую простую в освоении", "самую функциональную"],
    "лучших": ["самых надёжных", "самых бюджетных", "самых популярных", "самых функциональных"],
    "лучше": ["надёжнее", "дешевле", "проще в использовании", "быстрее"],
    "Топ-5": ["Рейтинг", "Обзор лучших", "Список топовых", "Подборка"],
    "дешевле": ["выгоднее", "экономичнее", "доступнее по цене"],
}

_PIVOTS_EN: dict[str, list[str]] = {
    "best": ["most reliable", "cheapest", "easiest to learn", "most feature-rich", "most popular"],
    "better": ["more reliable", "cheaper", "easier to use", "faster"],
    "Top 5": ["Ranking of", "Best overview of", "Top list of", "Curated list of"],
    "cheaper": ["more affordable", "more cost-effective", "better value for money"],
    "top": ["leading", "most popular", "highest-rated"],
}

# Conversational rewrites for variation without LLM
_CONVERSATIONAL_PREFIXES_RU = [
    "Подскажи, ",
    "Помоги разобраться: ",
    "Хочу понять — ",
    "Интересует вопрос: ",
    "",  # no prefix = original
]

_CONVERSATIONAL_PREFIXES_EN = [
    "Can you help me understand: ",
    "I'd like to know — ",
    "Quick question: ",
    "Help me figure out: ",
    "",
]

_CONVERSATIONAL_SUFFIXES_RU = [
    "",
    " Объясни подробно.",
    " Приведи примеры.",
    " Дай конкретные рекомендации.",
]

_CONVERSATIONAL_SUFFIXES_EN = [
    "",
    " Please explain in detail.",
    " Give specific examples.",
    " Provide concrete recommendations.",
]


def apply_semantic_pivots(
    skeletons: list[PromptSkeleton],
    market: str = "ru",
) -> list[ExpandedPrompt]:
    """Apply semantic pivots to skeletons.

    For each skeleton, checks if any pivot keywords appear in the text.
    If found, generates N additional variants with substituted adjectives.
    """
    pivots = _PIVOTS_RU if market == "ru" else _PIVOTS_EN
    results: list[ExpandedPrompt] = []

    for skel in skeletons:
        text = skel.render()

        # Always include the original
        results.append(
            ExpandedPrompt(
                text=text,
                persona=skel.persona,
                intent=skel.intent,
                category=skel.category,
                brand=skel.brand,
                competitors=skel.competitors,
                market=skel.market,
                pivot_axis="original",
            )
        )

        # Apply pivots
        for keyword, replacements in pivots.items():
            if keyword.lower() in text.lower():
                for replacement in replacements:
                    pivoted = re.sub(
                        re.escape(keyword),
                        replacement,
                        text,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    if pivoted != text:
                        results.append(
                            ExpandedPrompt(
                                text=pivoted,
                                persona=skel.persona,
                                intent=skel.intent,
                                category=skel.category,
                                brand=skel.brand,
                                competitors=skel.competitors,
                                market=skel.market,
                                pivot_axis=replacement,
                            )
                        )

    logger.info("Semantic pivots: %d skeletons → %d expanded prompts", len(skeletons), len(results))
    return results


def apply_conversational_variations(
    prompts: list[ExpandedPrompt],
    market: str = "ru",
    max_variations: int = 3,
) -> list[ExpandedPrompt]:
    """Generate conversational variations of prompts (rule-based, no LLM).

    Adds prefixes and suffixes to make prompts sound more natural,
    simulating different user communication styles.
    """
    prefixes = _CONVERSATIONAL_PREFIXES_RU if market == "ru" else _CONVERSATIONAL_PREFIXES_EN

    results: list[ExpandedPrompt] = []
    variation_count = 0

    for prompt in prompts:
        # Always keep the original
        results.append(prompt)

        # Only generate variations for "original" pivot to avoid combinatorial explosion
        if prompt.pivot_axis != "original":
            continue

        generated = 0
        for prefix in prefixes:
            if not prefix:  # skip empty prefix (that's the original)
                continue
            if generated >= max_variations:
                break

            varied_text = prefix + prompt.text[0].lower() + prompt.text[1:]
            results.append(
                ExpandedPrompt(
                    text=varied_text,
                    persona=prompt.persona,
                    intent=prompt.intent,
                    category=prompt.category,
                    brand=prompt.brand,
                    competitors=prompt.competitors,
                    market=prompt.market,
                    pivot_axis=f"conversational:{prefix.strip().rstrip(':')}",
                )
            )
            generated += 1
            variation_count += 1

    logger.info("Conversational variations: added %d new prompts (total %d)", variation_count, len(results))
    return results


# ---------------------------------------------------------------------------
# LLM-based expansion (optional, requires API key)
# ---------------------------------------------------------------------------

_EXPANSION_SYSTEM_PROMPT_RU = """Ты — специалист по формулированию поисковых запросов.
Тебе дан шаблонный запрос. Создай {count} естественных вариаций этого запроса.
Требования:
- Каждая вариация должна сохранять тот же смысл и намерение
- Используй разный стиль: от разговорного до профессионального
- Варьируй длину: от коротких до развёрнутых
- Некоторые могут содержать сленг или неформальные обороты
- Сохраняй все упомянутые бренды и продукты

Верни только список вариаций, по одной на строку, без нумерации и пояснений."""

_EXPANSION_SYSTEM_PROMPT_EN = """You are a search query formulation specialist.
Given a template query, create {count} natural variations of this query.
Requirements:
- Each variation must preserve the same meaning and intent
- Use different styles: from conversational to professional
- Vary the length: from short to elaborate
- Some may include slang or informal phrasing
- Preserve all mentioned brands and products

Return only the list of variations, one per line, without numbering or explanations."""


async def expand_with_llm(
    prompts: list[ExpandedPrompt],
    api_key: str,
    market: str = "ru",
    variations_per_prompt: int = 5,
    model: str = "gpt-4o-mini",
) -> list[ExpandedPrompt]:
    """Use a lightweight LLM to generate natural-language variations.

    Only expands 'original' pivots to avoid excessive API costs.
    Falls back gracefully if the API call fails.
    """
    import httpx

    system_prompt = (_EXPANSION_SYSTEM_PROMPT_RU if market == "ru" else _EXPANSION_SYSTEM_PROMPT_EN).format(
        count=variations_per_prompt
    )

    originals = [p for p in prompts if p.pivot_axis == "original"]
    results = list(prompts)  # Start with all existing prompts
    expanded_count = 0

    for prompt in originals:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt.text},
                ],
                "temperature": 0.8,
                "max_tokens": 1024,
            }

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            text = data["choices"][0]["message"]["content"] or ""
            variations = [
                line.strip().lstrip("0123456789.-) ")
                for line in text.strip().split("\n")
                if line.strip() and len(line.strip()) > 10
            ]

            for variation in variations[:variations_per_prompt]:
                results.append(
                    ExpandedPrompt(
                        text=variation,
                        persona=prompt.persona,
                        intent=prompt.intent,
                        category=prompt.category,
                        brand=prompt.brand,
                        competitors=prompt.competitors,
                        market=prompt.market,
                        pivot_axis="llm_expansion",
                    )
                )
                expanded_count += 1

        except Exception as e:
            logger.warning("LLM expansion failed for prompt (falling back): %s", e)
            continue

    logger.info("LLM expansion: generated %d new variations (total %d)", expanded_count, len(results))
    return results


def expand_skeletons(
    skeletons: list[PromptSkeleton],
    profile: AuditProfile,
    enable_pivots: bool = True,
    enable_conversational: bool = True,
) -> list[ExpandedPrompt]:
    """Main expansion entry point (rule-based only, no LLM).

    For LLM-based expansion, call expand_with_llm() separately after this.
    """
    if enable_pivots:
        expanded = apply_semantic_pivots(skeletons, market=profile.market)
    else:
        # Just convert skeletons to ExpandedPrompts without pivots
        expanded = [
            ExpandedPrompt(
                text=skel.render(),
                persona=skel.persona,
                intent=skel.intent,
                category=skel.category,
                brand=skel.brand,
                competitors=skel.competitors,
                market=skel.market,
                pivot_axis="original",
            )
            for skel in skeletons
        ]

    if enable_conversational:
        expanded = apply_conversational_variations(expanded, market=profile.market)

    return expanded
