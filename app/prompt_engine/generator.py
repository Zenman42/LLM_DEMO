"""V2 LLM-based prompt generation with two query classes.

Architecture:
  - THEMATIC queries (no brand in query) → measure which brands LLM recalls
  - BRANDED queries (brand in query) → measure LLM's opinion about brand

Each class has its own meta-prompt and post-filtering logic.
Uses gpt-4o for higher quality generation.
Falls back to template-based generation when LLM is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher

import httpx

from app.prompt_engine.types import (
    AuditProfile,
    BrandedSubtype,
    BRANDED_SUBTYPE_DESCRIPTIONS,
    ExpandedPrompt,
    Intent,
    Persona,
    PERSONA_DESCRIPTIONS,
    QueryClass,
    ScenarioConfig,
    ThematicSubtype,
    THEMATIC_SUBTYPE_DESCRIPTIONS,
)

logger = logging.getLogger(__name__)

# Default model for V2 generation
DEFAULT_MODEL = "gemini-3-flash-preview"
_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# ═══════════════════════════════════════════════════════════════════════════
# Meta-prompts: THEMATIC (brand-free queries)
# ═══════════════════════════════════════════════════════════════════════════

_THEMATIC_SYSTEM_RU = """Ты — генератор реалистичных запросов к AI-ассистентам (ChatGPT, YandexGPT, Gemini, DeepSeek).

Задача: сгенерировать запросы, на которые AI-ассистент ОБЯЗАТЕЛЬНО ответит с упоминанием конкретных компаний/брендов из категории.

Контекст:
- Бизнес-категория: {business_category}
{brand_description_block}- Рынок: {market}{geo_part}

Персона пользователя: {persona_description}

Тип запроса: {subtype_description}

АБСОЛЮТНЫЙ ЗАПРЕТ:
- В тексте запроса НЕ ДОЛЖНО быть названий НИКАКИХ брендов, компаний, продуктов: {all_brands_ban_list}.
- Если бренд пролез в запрос — это провал. Запрос будет отброшен.

ОБЯЗАТЕЛЬНОЕ ТРЕБОВАНИЕ:
- Каждый запрос должен быть сформулирован так, чтобы ПОЛНЫЙ ответ AI-ассистента ОБЯЗАТЕЛЬНО содержал названия конкретных компаний/брендов из этой категории.
- Запросы типа «стоит ли покупать квартиру» — ЗАПРЕЩЕНЫ, потому что на них можно ответить без упоминания брендов.
- Правильный подход: «какой застройщик лучше в москве» — на это НЕВОЗМОЖНО ответить без конкретных названий.

Правила:
1. Запросы должны звучать как ЖИВАЯ РЕЧЬ реального человека в чате с AI.
2. Варьируй стиль: разговорный, полуформальный, с лёгкими неточностями.
3. Варьируй длину: от коротких (5-8 слов) до развёрнутых (15-25 слов).
4. НЕ начинай запросы со слов «Выполни», «Проведи», «Составь таблицу».
5. Каждый запрос УНИКАЛЕН по смыслу — не генерируй вариации одного запроса.
6. КРИТИЧНО — естественный русский язык:
   - Используй ПРАВИЛЬНЫЕ падежи, предлоги, согласования
   - НЕ калькируй с английского — пиши как носитель русского языка
   - Категория — это ТЕМА запроса, а не фраза для вставки. Перефразируй её своими словами
   - ПЛОХО: «посоветуй вариант в сфере покупка квартиры бизнес-класса»
   - ХОРОШО: «посоветуй, где лучше купить квартиру бизнес-класса»

Сгенерируй {count} запросов.

Верни JSON: {{"queries": ["запрос 1", "запрос 2", ...]}}"""

_THEMATIC_SYSTEM_EN = """You are a generator of realistic queries for AI assistants (ChatGPT, Gemini, DeepSeek).

Task: generate queries that AI assistants will NECESSARILY answer by naming specific companies/brands in the category.

Context:
- Business category: {business_category}
{brand_description_block}- Market: {market}{geo_part}

User persona: {persona_description}

Query type: {subtype_description}

ABSOLUTE BAN:
- The query text MUST NOT contain ANY brand, company, or product names: {all_brands_ban_list}.
- If a brand leaks into the query — it's a failure. The query will be discarded.

MANDATORY REQUIREMENT:
- Each query must be formulated so that a COMPLETE AI response MUST name specific companies/brands from this category.
- Queries like "is it worth buying an apartment" are FORBIDDEN because they can be answered without naming brands.
- Correct approach: "which developer is best in Moscow" — IMPOSSIBLE to answer without specific names.

Rules:
1. Queries must sound like REAL HUMAN SPEECH in an AI chat.
2. Vary the style: casual, semi-formal, with slight imperfections.
3. Vary the length: from short (5-8 words) to detailed (15-25 words).
4. Do NOT start queries with "Perform", "Execute", "Create a table".
5. Each query must be UNIQUE in meaning — no variations of the same query.

Generate {count} queries.

Return JSON: {{"queries": ["query 1", "query 2", ...]}}"""


# ═══════════════════════════════════════════════════════════════════════════
# Meta-prompts: BRANDED (brand-in-query)
# ═══════════════════════════════════════════════════════════════════════════

_BRANDED_SYSTEM_RU = """Ты — генератор реалистичных запросов к AI-ассистентам (ChatGPT, YandexGPT, Gemini, DeepSeek).

Задача: сгенерировать запросы, в которых пользователь спрашивает AI про КОНКРЕТНЫЙ бренд.

Контекст:
- Бренд: {brand}
{brand_description_block}- Конкуренты: {competitors}
- Бизнес-категория: {business_category}
- Рынок: {market}{geo_part}

Персона пользователя: {persona_description}

Тип запроса: {subtype_description}

ОБЯЗАТЕЛЬНО:
- Каждый запрос ДОЛЖЕН содержать название бренда «{brand}» (или его алиасы: {brand_aliases}).
{branded_extra_instructions}

Правила:
1. Запросы должны звучать как ЖИВАЯ РЕЧЬ реального человека в чате с AI.
2. Варьируй стиль: разговорный, полуформальный, с лёгкими неточностями.
3. Варьируй длину: от коротких (5-8 слов) до развёрнутых (15-25 слов).
4. НЕ начинай запросы со слов «Выполни», «Проведи», «Составь таблицу».
5. Каждый запрос УНИКАЛЕН по смыслу — не генерируй вариации одного запроса.
6. КРИТИЧНО — естественный русский язык:
   - Используй ПРАВИЛЬНЫЕ падежи, предлоги, согласования
   - НЕ калькируй с английского — пиши как носитель русского языка
   - Категория — это ТЕМА запроса, а не фраза для вставки. Перефразируй её своими словами
   - ПЛОХО: «расскажи про {brand} в сфере покупка квартиры бизнес-класса»
   - ХОРОШО: «расскажи про {brand} — стоит ли покупать у них квартиру бизнес-класса»

Сгенерируй {count} запросов.

Верни JSON: {{"queries": ["запрос 1", "запрос 2", ...]}}"""

_BRANDED_SYSTEM_EN = """You are a generator of realistic queries for AI assistants (ChatGPT, Gemini, DeepSeek).

Task: generate queries where the user asks the AI about a SPECIFIC brand.

Context:
- Brand: {brand}
{brand_description_block}- Competitors: {competitors}
- Business category: {business_category}
- Market: {market}{geo_part}

User persona: {persona_description}

Query type: {subtype_description}

MANDATORY:
- Each query MUST contain the brand name "{brand}" (or its aliases: {brand_aliases}).
{branded_extra_instructions}

Rules:
1. Queries must sound like REAL HUMAN SPEECH in an AI chat.
2. Vary the style: casual, semi-formal, with slight imperfections.
3. Vary the length: from short (5-8 words) to detailed (15-25 words).
4. Do NOT start queries with "Perform", "Execute", "Create a table".
5. Each query must be UNIQUE in meaning — no variations of the same query.

Generate {count} queries.

Return JSON: {{"queries": ["query 1", "query 2", ...]}}"""


# ═══════════════════════════════════════════════════════════════════════════
# Extra instructions per branded subtype
# ═══════════════════════════════════════════════════════════════════════════

_BRANDED_EXTRA_RU = {
    BrandedSubtype.REPUTATION: (
        "- Запросы должны спрашивать мнение AI о бренде: стоит ли выбирать, что за компания, можно ли доверять.\n"
        "- НЕ сравнивай с конкурентами — это отдельный тип."
    ),
    BrandedSubtype.COMPARISON: (
        "- Каждый запрос сравнивает {brand} с 1-2 конкурентами из списка: {competitors}.\n"
        "- ЧЕРЕДУЙ порядок брендов: не всегда ставь {brand} первым.\n"
        "- Используй формулировки: «X или Y что лучше», «X vs Y», «чем X отличается от Y»."
    ),
    BrandedSubtype.NEGATIVE: (
        "- Запросы о НЕДОСТАТКАХ, проблемах, негативных отзывах о {brand}.\n"
        "- Половина запросов про {brand}, половина про конкурентов ({competitors}).\n"
        "- Формулировки: «минусы X», «проблемы X», «почему не стоит», «жалобы на X»."
    ),
    BrandedSubtype.FACT_CHECK: (
        "- Запросы проверяют КОНКРЕТНЫЕ факты о бренде {brand}.\n"
        "- Цель: узнать, знает ли AI реальные факты или выдумывает.\n"
        "{golden_facts_instruction}"
        "- Формулировки: «правда что у X...?», «у X есть...?», «X действительно...?»."
    ),
}

_BRANDED_EXTRA_EN = {
    BrandedSubtype.REPUTATION: (
        "- Queries should ask the AI's opinion about the brand: is it worth choosing, what kind of company, can you trust it.\n"
        "- Do NOT compare with competitors — that's a separate type."
    ),
    BrandedSubtype.COMPARISON: (
        "- Each query compares {brand} with 1-2 competitors from: {competitors}.\n"
        "- ALTERNATE brand order: don't always put {brand} first.\n"
        "- Use phrases: 'X vs Y', 'X compared to Y', 'difference between X and Y'."
    ),
    BrandedSubtype.NEGATIVE: (
        "- Queries about DRAWBACKS, problems, negative reviews of {brand}.\n"
        "- Half about {brand}, half about competitors ({competitors}).\n"
        "- Phrases: 'downsides of X', 'problems with X', 'why not choose X', 'complaints about X'."
    ),
    BrandedSubtype.FACT_CHECK: (
        "- Queries verify SPECIFIC facts about brand {brand}.\n"
        "- Goal: find out if the AI knows real facts or hallucinates.\n"
        "{golden_facts_instruction}"
        "- Phrases: 'is it true that X...?', 'does X have...?', 'does X really...?'."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Subtype → legacy Intent mapping (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

SUBTYPE_TO_INTENT = {
    # Thematic
    ThematicSubtype.CATEGORY: Intent.INFORMATIONAL,
    ThematicSubtype.ATTRIBUTE: Intent.COMPARATIVE,
    ThematicSubtype.SCENARIO: Intent.TRANSACTIONAL,
    ThematicSubtype.EVENT: Intent.INFORMATIONAL,
    # Branded
    BrandedSubtype.REPUTATION: Intent.NAVIGATIONAL,
    BrandedSubtype.COMPARISON: Intent.COMPARATIVE,
    BrandedSubtype.NEGATIVE: Intent.NAVIGATIONAL,
    BrandedSubtype.FACT_CHECK: Intent.INFORMATIONAL,
}


# ═══════════════════════════════════════════════════════════════════════════
# Prompt building helpers
# ═══════════════════════════════════════════════════════════════════════════


def _build_thematic_prompt(
    profile: AuditProfile,
    persona: Persona,
    subtype: ThematicSubtype,
    scenario: ScenarioConfig | str,
    count: int,
) -> str:
    """Build meta-prompt for thematic query generation."""
    market = profile.market
    template = _THEMATIC_SYSTEM_RU if market == "ru" else _THEMATIC_SYSTEM_EN

    # Support both ScenarioConfig and legacy string
    if isinstance(scenario, str):
        category_label = scenario
    else:
        category_label = scenario.rich_label

    logger.info(
        "BUILD_THEMATIC type=%s name=%r desc=%r hints=%r → rich_label=%r",
        type(scenario).__name__,
        getattr(scenario, "name", "?"),
        getattr(scenario, "description", "?"),
        getattr(scenario, "context_hints", "?"),
        category_label,
    )

    persona_desc = PERSONA_DESCRIPTIONS.get(persona, {}).get(market, str(persona.value))
    subtype_desc = THEMATIC_SUBTYPE_DESCRIPTIONS.get(subtype, {}).get(market, "")
    geo_part = f", Гео: {profile.geo}" if profile.geo else ""

    # Build ban list: all brands + aliases that must NOT appear in queries
    all_brands = [profile.brand] + profile.brand_aliases + profile.competitors
    ban_list = ", ".join(f"«{b}»" for b in all_brands)

    brand_description_block = ""
    if profile.brand_description:
        label = "О компании" if market == "ru" else "About the company"
        brand_description_block = f"- {label}: {profile.brand_description}\n"

    return template.format(
        business_category=category_label,
        brand_description_block=brand_description_block,
        market=market,
        geo_part=geo_part,
        persona_description=persona_desc,
        subtype_description=subtype_desc,
        all_brands_ban_list=ban_list,
        count=count,
    )


def _build_branded_prompt(
    profile: AuditProfile,
    persona: Persona,
    subtype: BrandedSubtype,
    scenario: ScenarioConfig | str,
    count: int,
) -> str:
    """Build meta-prompt for branded query generation."""
    market = profile.market
    template = _BRANDED_SYSTEM_RU if market == "ru" else _BRANDED_SYSTEM_EN

    # Support both ScenarioConfig and legacy string
    if isinstance(scenario, str):
        category_label = scenario
    else:
        category_label = scenario.rich_label

    persona_desc = PERSONA_DESCRIPTIONS.get(persona, {}).get(market, str(persona.value))
    subtype_desc = BRANDED_SUBTYPE_DESCRIPTIONS.get(subtype, {}).get(market, "")
    competitors_str = (
        ", ".join(profile.competitors) if profile.competitors else ("не указаны" if market == "ru" else "not specified")
    )
    aliases_str = ", ".join(profile.brand_aliases) if profile.brand_aliases else profile.brand
    geo_part = f", Гео: {profile.geo}" if profile.geo else ""

    brand_description_block = ""
    if profile.brand_description:
        label = "О компании" if market == "ru" else "About the company"
        brand_description_block = f"- {label}: {profile.brand_description}\n"

    # Build extra instructions for this subtype
    extras = _BRANDED_EXTRA_RU if market == "ru" else _BRANDED_EXTRA_EN
    extra_template = extras.get(subtype, "")

    # Golden facts instruction for fact_check subtype
    golden_facts_instruction = ""
    if subtype == BrandedSubtype.FACT_CHECK and profile.golden_facts:
        facts_str = "\n".join(f"  - {f}" for f in profile.golden_facts[:10])
        if market == "ru":
            golden_facts_instruction = (
                f"- Используй эти РЕАЛЬНЫЕ факты о бренде как основу для запросов:\n{facts_str}\n"
            )
        else:
            golden_facts_instruction = f"- Use these REAL facts about the brand as basis for queries:\n{facts_str}\n"

    branded_extra = extra_template.format(
        brand=profile.brand,
        competitors=competitors_str,
        golden_facts_instruction=golden_facts_instruction,
    )

    return template.format(
        brand=profile.brand,
        brand_description_block=brand_description_block,
        competitors=competitors_str,
        business_category=category_label,
        market=market,
        geo_part=geo_part,
        persona_description=persona_desc,
        subtype_description=subtype_desc,
        brand_aliases=aliases_str,
        branded_extra_instructions=branded_extra,
        count=count,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LLM call + response parsing
# ═══════════════════════════════════════════════════════════════════════════


def _parse_generation_output(text: str) -> list[str]:
    """Parse LLM output: try JSON first, fall back to line-per-prompt."""
    text = text.strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "queries" in parsed:
            return [str(q).strip() for q in parsed["queries"] if str(q).strip() and len(str(q).strip()) > 10]
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip() and len(str(q).strip()) > 10]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    json_match = re.search(r'\{.*"queries"\s*:\s*\[.*\].*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if "queries" in parsed:
                return [str(q).strip() for q in parsed["queries"] if str(q).strip() and len(str(q).strip()) > 10]
        except json.JSONDecodeError:
            pass

    # Try to extract just an array
    array_match = re.search(r"\[.*\]", text, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip() and len(str(q).strip()) > 10]
        except json.JSONDecodeError:
            pass

    # Last resort: line-per-prompt
    lines = []
    for line in text.split("\n"):
        cleaned = line.strip().lstrip("0123456789.-) \"'").rstrip("\"',")
        if cleaned and len(cleaned) > 10:
            lines.append(cleaned)
    return lines


async def _call_llm(
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    user_msg: str,
    api_key: str,
    model: str,
    count: int,
) -> list[str]:
    """Make one LLM call via Gemini API and return parsed query list."""
    async with semaphore:
        url = _GEMINI_API_URL.format(model=model)

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": user_msg}]},
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        url,
                        json=payload,
                        params={"key": api_key},
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code == 429 and attempt < 2:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    resp.raise_for_status()
                break
            except httpx.TimeoutException:
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                raise

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            logger.warning("Gemini returned no candidates: %s", list(data.keys()))
            return []

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
        return _parse_generation_output(text)[:count]


# ═══════════════════════════════════════════════════════════════════════════
# Post-filtering
# ═══════════════════════════════════════════════════════════════════════════


def _contains_any_brand(text: str, brands: list[str]) -> bool:
    """Check if text contains any of the brand names (case-insensitive)."""
    text_lower = text.lower()
    for brand in brands:
        if brand.lower() in text_lower:
            return True
    return False


def _filter_thematic(queries: list[str], profile: AuditProfile) -> list[str]:
    """Remove thematic queries that accidentally contain brand names."""
    all_brands = [profile.brand] + profile.brand_aliases + profile.competitors
    filtered = []
    for q in queries:
        if _contains_any_brand(q, all_brands):
            logger.debug("Thematic filter: removed query with brand: %s", q[:80])
            continue
        filtered.append(q)
    return filtered


def _filter_branded(queries: list[str], profile: AuditProfile) -> list[str]:
    """Remove branded queries that don't contain the target brand."""
    brand_names = [profile.brand] + profile.brand_aliases
    filtered = []
    for q in queries:
        if _contains_any_brand(q, brand_names):
            filtered.append(q)
        else:
            logger.debug("Branded filter: removed query without brand: %s", q[:80])
    return filtered


def _fuzzy_dedup(
    prompts: list[ExpandedPrompt],
    threshold: float = 0.75,
) -> list[ExpandedPrompt]:
    """Remove near-duplicate prompts using SequenceMatcher."""
    if len(prompts) <= 1:
        return prompts

    kept: list[ExpandedPrompt] = []
    kept_texts: list[str] = []

    for prompt in prompts:
        text_lower = prompt.text.strip().lower()

        is_dup = False
        for existing in kept_texts:
            if SequenceMatcher(None, text_lower, existing).ratio() > threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(prompt)
            kept_texts.append(text_lower)

    if len(prompts) != len(kept):
        logger.debug(
            "Fuzzy dedup: %d → %d prompts (removed %d)",
            len(prompts),
            len(kept),
            len(prompts) - len(kept),
        )
    return kept


# ═══════════════════════════════════════════════════════════════════════════
# Generation orchestrators per class
# ═══════════════════════════════════════════════════════════════════════════


async def _generate_thematic_batch(
    semaphore: asyncio.Semaphore,
    profile: AuditProfile,
    persona: Persona,
    subtype: ThematicSubtype,
    scenario: ScenarioConfig | str,
    api_key: str,
    count: int,
    model: str,
) -> list[ExpandedPrompt]:
    """Generate one batch of thematic prompts."""
    system_prompt = _build_thematic_prompt(profile, persona, subtype, scenario, count)
    market = profile.market
    cat_name = scenario.name if isinstance(scenario, ScenarioConfig) else scenario
    cat_desc = scenario.description if isinstance(scenario, ScenarioConfig) and scenario.description else ""

    if market == "ru":
        if cat_desc:
            user_msg = (
                f"Вот ситуация: {cat_desc}\n\n"
                f"Напиши {count} разных сообщений, которые этот человек отправит в ChatGPT/YandexGPT. "
                f"Тема: «{cat_name}»."
            )
        else:
            user_msg = (
                f"Напиши {count} разных сообщений, которые реальный человек отправит в ChatGPT/YandexGPT. "
                f"Тема: «{cat_name}»."
            )
    else:
        if cat_desc:
            user_msg = (
                f"Here's the situation: {cat_desc}\n\n"
                f"Write {count} different messages this person would send to ChatGPT/Gemini. "
                f'Topic: "{cat_name}".'
            )
        else:
            user_msg = (
                f'Write {count} different messages a real person would send to ChatGPT/Gemini. Topic: "{cat_name}".'
            )

    logger.info(
        "THEMATIC prompt [%s/%s/%s]\n=== SYSTEM ===\n%s\n=== USER ===\n%s",
        persona.value,
        subtype.value,
        cat_name,
        system_prompt[:2000],
        user_msg,
    )

    raw_queries = await _call_llm(semaphore, system_prompt, user_msg, api_key, model, count)
    filtered = _filter_thematic(raw_queries, profile)

    intent = SUBTYPE_TO_INTENT.get(subtype, Intent.INFORMATIONAL)

    return [
        ExpandedPrompt(
            text=q,
            persona=persona,
            intent=intent,
            category=cat_name,
            brand=profile.brand,
            competitors=profile.competitors,
            market=market,
            pivot_axis="llm_generated_v2",
            query_class=QueryClass.THEMATIC.value,
            query_subtype=subtype.value,
        )
        for q in filtered
    ]


async def _generate_branded_batch(
    semaphore: asyncio.Semaphore,
    profile: AuditProfile,
    persona: Persona,
    subtype: BrandedSubtype,
    scenario: ScenarioConfig | str,
    api_key: str,
    count: int,
    model: str,
) -> list[ExpandedPrompt]:
    """Generate one batch of branded prompts."""
    # Skip fact_check if no golden facts
    if subtype == BrandedSubtype.FACT_CHECK and not profile.golden_facts:
        return []

    system_prompt = _build_branded_prompt(profile, persona, subtype, scenario, count)
    market = profile.market
    cat_name = scenario.name if isinstance(scenario, ScenarioConfig) else scenario
    cat_desc = scenario.description if isinstance(scenario, ScenarioConfig) and scenario.description else ""

    if market == "ru":
        if cat_desc:
            user_msg = (
                f"Вот ситуация: {cat_desc}\n\n"
                f"Напиши {count} разных сообщений, которые этот человек отправит в ChatGPT/YandexGPT про «{profile.brand}». "
                f"Тема: «{cat_name}»."
            )
        else:
            user_msg = (
                f"Напиши {count} разных сообщений, которые реальный человек отправит в ChatGPT/YandexGPT про «{profile.brand}». "
                f"Тема: «{cat_name}»."
            )
    else:
        if cat_desc:
            user_msg = (
                f"Here's the situation: {cat_desc}\n\n"
                f'Write {count} different messages this person would send to ChatGPT/Gemini about "{profile.brand}". '
                f'Topic: "{cat_name}".'
            )
        else:
            user_msg = (
                f'Write {count} different messages a real person would send to ChatGPT/Gemini about "{profile.brand}". '
                f'Topic: "{cat_name}".'
            )

    logger.info(
        "BRANDED prompt [%s/%s/%s]\n=== SYSTEM ===\n%s\n=== USER ===\n%s",
        persona.value,
        subtype.value,
        cat_name,
        system_prompt[:2000],
        user_msg,
    )

    raw_queries = await _call_llm(semaphore, system_prompt, user_msg, api_key, model, count)
    filtered = _filter_branded(raw_queries, profile)

    intent = SUBTYPE_TO_INTENT.get(subtype, Intent.NAVIGATIONAL)

    return [
        ExpandedPrompt(
            text=q,
            persona=persona,
            intent=intent,
            category=cat_name,
            brand=profile.brand,
            competitors=profile.competitors,
            market=market,
            pivot_axis="llm_generated_v2",
            query_class=QueryClass.BRANDED.value,
            query_subtype=subtype.value,
        )
        for q in filtered
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


async def generate_prompts_v2(
    profile: AuditProfile,
    api_key: str,
    personas: list[Persona] | None = None,
    thematic_subtypes: list[ThematicSubtype] | None = None,
    branded_subtypes: list[BrandedSubtype] | None = None,
    prompts_per_scenario: int = 10,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 10,
) -> list[ExpandedPrompt]:
    """Generate prompts using V2 architecture.

    Simple model: 1 scenario = 1 thematic LLM call (+ 1 branded if scenario has branded_subtypes).
    Persona/style is embedded in the scenario description, not multiplied as combinations.

    Args:
        profile: Brand audit profile
        api_key: OpenAI API key
        personas: Legacy param, kept for backward compat (ignored in new logic)
        thematic_subtypes: Legacy param (ignored — scenario decides)
        branded_subtypes: Legacy param (ignored — scenario decides)
        prompts_per_scenario: How many prompts per scenario (default: 10)
        model: OpenAI model to use (default: gpt-4o)
        max_concurrent: Max concurrent LLM calls

    Returns:
        List of ExpandedPrompt objects with query_class and query_subtype set.
    """
    # Thematic prompts: use full count; branded: ~30% of that, minimum 2
    thematic_count = prompts_per_scenario
    branded_count = max(2, prompts_per_scenario // 3)

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []

    thematic_scenarios = 0
    branded_scenarios = 0

    for scenario in profile.scenarios:
        # Every scenario gets 1 thematic generation call
        # Use SCENARIO subtype as default — it best matches "person in a situation"
        tasks.append(
            _generate_thematic_batch(
                semaphore=semaphore,
                profile=profile,
                persona=Persona.BEGINNER,  # Neutral default; actual persona is in scenario desc
                subtype=ThematicSubtype.SCENARIO,
                scenario=scenario,
                api_key=api_key,
                count=thematic_count,
                model=model,
            )
        )
        thematic_scenarios += 1

        # Branded only if scenario explicitly has branded_subtypes
        if scenario.branded_subtypes:
            # Pick the first branded subtype as representative
            branded_subtype_val = scenario.branded_subtypes[0]
            try:
                branded_subtype = BrandedSubtype(branded_subtype_val)
            except ValueError:
                branded_subtype = BrandedSubtype.REPUTATION

            # Skip fact_check if no golden facts
            if branded_subtype == BrandedSubtype.FACT_CHECK and not profile.golden_facts:
                pass
            else:
                tasks.append(
                    _generate_branded_batch(
                        semaphore=semaphore,
                        profile=profile,
                        persona=Persona.BEGINNER,
                        subtype=branded_subtype,
                        scenario=scenario,
                        api_key=api_key,
                        count=branded_count,
                        model=model,
                    )
                )
                branded_scenarios += 1

    total_tasks = len(tasks)
    logger.info(
        "V2 generation: %d scenarios → %d thematic + %d branded = %d LLM calls",
        len(profile.scenarios),
        thematic_scenarios,
        branded_scenarios,
        total_tasks,
    )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_prompts: list[ExpandedPrompt] = []
    errors = 0
    for result in results:
        if isinstance(result, Exception):
            errors += 1
            logger.warning("V2 generation failed for one combination: %s", result)
            continue
        all_prompts.extend(result)

    # Deduplicate
    deduped = _fuzzy_dedup(all_prompts, threshold=0.75)

    logger.info(
        "V2 generation complete: %d tasks, %d raw, %d after dedup, %d errors",
        total_tasks,
        len(all_prompts),
        len(deduped),
        errors,
    )

    # If ALL tasks failed, raise so the caller can fall back to templates
    if errors > 0 and not deduped:
        first_error = next(
            (r for r in results if isinstance(r, Exception)), RuntimeError("All V2 generation tasks failed")
        )
        raise RuntimeError(f"V2 generation produced 0 prompts ({errors}/{total_tasks} tasks failed): {first_error}")

    return deduped


# ═══════════════════════════════════════════════════════════════════════════
# Legacy API — kept for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════


async def generate_prompts_llm(
    profile: AuditProfile,
    api_key: str,
    personas: list[Persona] | None = None,
    intents: list[Intent] | None = None,
    prompts_per_scenario: int = 10,
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 10,
) -> list[ExpandedPrompt]:
    """Legacy API — redirects to V2 generation.

    Maps old intent-based interface to new class/subtype system.
    """
    return await generate_prompts_v2(
        profile=profile,
        api_key=api_key,
        personas=personas,
        prompts_per_scenario=prompts_per_scenario,
        model=model,
        max_concurrent=max_concurrent,
    )


def generate_prompts_fallback(
    profile: AuditProfile,
    personas: list[Persona] | None = None,
    intents: list[Intent] | None = None,
) -> list[ExpandedPrompt]:
    """Rule-based fallback when LLM generation is unavailable."""
    from app.prompt_engine.expansion import expand_skeletons
    from app.prompt_engine.matrix import build_skeletons

    skeletons = build_skeletons(profile, personas=personas, intents=intents)
    return expand_skeletons(
        skeletons,
        profile,
        enable_pivots=True,
        enable_conversational=True,
    )
