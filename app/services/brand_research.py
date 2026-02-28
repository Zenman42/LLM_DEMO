"""Brand research service — auto-generates company profiles, competitor and category suggestions.

Used in the onboarding wizard to:
  1. Research brand → comprehensive product/service catalog (Step 1)
  2. Suggest competitors based on the brand profile (Step 2)
  3. Suggest product categories / audience use-cases (Step 3)
  4. Suggest sub-brands / product lines (Steps 1 & 2)

Strategy for all calls:
  1. Gemini 3.1 Pro (if key available) — best quality + Google Search grounding
  2. Perplexity (if key available) — has real-time web access, good quality
  3. OpenAI gpt-4o (fallback) — strong model, no web access but good knowledge
"""

from __future__ import annotations

import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-3.1-pro-preview"
_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# ---------------------------------------------------------------------------
# Shared LLM call helper (Gemini → Perplexity → OpenAI fallback)
# ---------------------------------------------------------------------------


async def _call_llm(
    prompt: str,
    openai_api_key: str,
    perplexity_api_key: str | None = None,
    gemini_api_key: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    use_grounding: bool = True,
) -> str:
    """Call LLM with Gemini-first, Perplexity-second, OpenAI-fallback strategy.

    Returns raw text response from the model.
    """
    logger.info(
        "[_call_llm] Keys: gemini=%s perplexity=%s openai=%s",
        bool(gemini_api_key),
        bool(perplexity_api_key),
        bool(openai_api_key),
    )

    # Strategy 1: Gemini 3.1 Pro (Google Search grounding)
    if gemini_api_key:
        try:
            result = await _call_gemini(
                prompt,
                gemini_api_key,
                max_tokens,
                temperature,
                use_grounding=use_grounding,
            )
            if result:
                logger.info("[_call_llm] Success via Gemini: %d chars", len(result))
                return result
            logger.warning("[_call_llm] Gemini returned empty, falling back")
        except Exception as e:
            logger.warning("[_call_llm] Gemini failed: %s: %s, falling back", type(e).__name__, e)

    # Strategy 2: Perplexity (web search)
    if perplexity_api_key:
        try:
            result = await _call_perplexity(prompt, perplexity_api_key, max_tokens, temperature)
            if result:
                logger.info("[_call_llm] Success via Perplexity: %d chars", len(result))
                return result
            logger.warning("[_call_llm] Perplexity returned empty, falling back")
        except Exception as e:
            logger.warning("[_call_llm] Perplexity failed: %s: %s, falling back to OpenAI", type(e).__name__, e)

    # Strategy 3: OpenAI (fallback)
    if openai_api_key:
        try:
            result = await _call_openai(prompt, openai_api_key, max_tokens, temperature)
            logger.info("[_call_llm] Success via OpenAI: %d chars", len(result))
            return result
        except Exception as e:
            logger.warning("[_call_llm] OpenAI failed: %s: %s", type(e).__name__, e)

    logger.error("[_call_llm] All strategies failed, returning empty")
    return ""


def _trim_to_last_sentence(text: str) -> str:
    """Trim text to the last complete sentence boundary.

    Looks for the last sentence-ending punctuation (. ! ? or their Russian
    equivalents) and cuts there.  If no sentence boundary is found, returns
    the original text unchanged.
    """
    # Find last sentence-ending punctuation followed by optional whitespace
    m = re.search(r"[.!?…]\s*$", text)
    if m:
        return text  # already ends with punctuation

    # Search backwards for the last sentence boundary
    idx = -1
    for ch in (".", "!", "?", "…"):
        pos = text.rfind(ch)
        if pos > idx:
            idx = pos

    if idx > 0:
        return text[: idx + 1]

    return text


async def _call_gemini(
    prompt: str,
    api_key: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    model: str = _GEMINI_MODEL,
    use_grounding: bool = True,
) -> str:
    """Call Gemini API (native), optionally with Google Search grounding."""
    url = _GEMINI_API_URL.format(model=model)

    gen_config: dict = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        # Minimize thinking — it consumes output token budget and
        # produces truncated answers for short-form tasks like descriptions
        # Note: "minimal" is Flash-only; Pro models minimum is "low"
        "thinkingConfig": {"thinkingLevel": "low"},
    }

    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    if use_grounding:
        payload["tools"] = [{"google_search": {}}]

    logger.info("[gemini] Calling model=%s grounding=%s maxTokens=%d", model, use_grounding, max_tokens)
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            url,
            json=payload,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            logger.error("[gemini] Error %d: %s", resp.status_code, resp.text[:500])
            resp.raise_for_status()

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        logger.warning("[gemini] No candidates in response. Keys: %s", list(data.keys()))
        return ""

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason", "")
    if finish_reason and finish_reason not in ("STOP", "END_TURN"):
        logger.warning("[gemini] finishReason=%s (may be truncated)", finish_reason)

    parts = candidate.get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    result = "".join(text_parts).strip()
    logger.info("[gemini] Response length=%d chars", len(result))
    return result


async def _call_perplexity(
    prompt: str,
    api_key: str,
    max_tokens: int = 4000,
    temperature: float = 0.1,
) -> str:
    """Call Perplexity API (has real-time web search)."""
    payload = {
        "model": "sonar",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.perplexity.ai/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()

    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


async def _call_openai(
    prompt: str,
    api_key: str,
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> str:
    """Call OpenAI API (no web access, relies on training data)."""
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
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
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def _parse_json_list(text: str, key: str) -> list[str]:
    """Extract a list of strings from LLM JSON response.

    Handles common LLM quirks: markdown code fences, trailing commas, etc.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and key in data:
            return [str(item).strip() for item in data[key] if str(item).strip()]
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON object in the text
    m = re.search(r'\{[^{}]*"' + re.escape(key) + r'"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return [str(item).strip() for item in data.get(key, []) if str(item).strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: extract quoted strings from truncated JSON
    # Handles cases like: {"competitors": ["A", "B", "C  (cut off by MAX_TOKENS)
    noise = {key.lower(), "competitors", "categories", "json", ""}
    items = re.findall(r'"([^"]{2,})"', text)
    items = [i.strip() for i in items if i.strip().lower() not in noise]
    if items:
        return items

    # Last resort: split by newlines, filter empties
    lines = [line.strip().lstrip("- \u2022\u00b70123456789.)").strip() for line in text.split("\n")]
    return [line for line in lines if line and len(line) > 1]


# ---------------------------------------------------------------------------
# Prompt templates — Brand research
# ---------------------------------------------------------------------------

_RESEARCH_PROMPT_RU = """Ты — бизнес-аналитик с доступом к интернету. \
Найди и составь подробный профиль компании \u00ab{brand_name}\u00bb.

{domain_hint}

Используй поиск, чтобы найти АКТУАЛЬНУЮ информацию о компании. \
Этот текст будет использоваться для генерации поисковых запросов к AI-ассистентам, \
поэтому чем больше конкретных фактов, продуктов и услуг ты найдёшь \u2014 тем лучше.

Обязательно укажи:

1. ОСНОВНАЯ ДЕЯТЕЛЬНОСТЬ: чем занимается компания, год основания, штаб-квартира (1-3 предложения).

2. ПОЛНЫЙ КАТАЛОГ ПРОДУКТОВ/УСЛУГ: перечисли ВСЕ основные продукты, услуги, \
товарные категории и подкатегории. Не ограничивайся 3-5 пунктами \u2014 пиши столько, сколько найдёшь. \
Для каждого продукта/услуги укажи конкретное название, а не общую категорию. \
Например: не \u00abкровельные материалы\u00bb, а \u00abметаллочерепица, профнастил, мягкая кровля, \
водосточные системы, мансардные окна, кровельные аксессуары\u00bb.

3. БРЕНДЫ И ЛИНЕЙКИ: собственные бренды, торговые марки, линейки продукции, \
ключевые продуктовые названия.

4. КЛЮЧЕВЫЕ ФАКТЫ: год основания, количество сотрудников, выручка (если публичная), \
количество клиентов/пользователей, награды, лицензии, сертификаты.

5. ЦЕЛЕВОЙ РЫНОК: география присутствия, основные сегменты клиентов (B2B/B2C), \
типичные покупатели и их сценарии использования.

6. ПОЗИЦИОНИРОВАНИЕ: лидер рынка / средний сегмент / нишевой игрок, \
ценовой сегмент (премиум/средний/эконом), главные конкурентные преимущества.

Правила:
- Ищи РЕАЛЬНУЮ информацию из интернета. Не придумывай продукты, которых нет.
- Если точных данных о каком-то аспекте нет \u2014 пропусти его, но не выдумывай.
- Если точных данных нет \u2014 сделай обоснованное предположение на основе названия{domain_hint_short}, \
но явно отметь это.
- Пиши фактологично, без маркетинговых клише.
- Только текст, без заголовков и маркированных списков."""

_RESEARCH_PROMPT_EN = """You are a business analyst with internet access. \
Research and create a detailed company profile for "{brand_name}".

{domain_hint}

Use web search to find CURRENT information about the company. \
This text will be used to generate search queries for AI assistants, \
so the more specific facts, products and services you find \u2014 the better.

You must include:

1. CORE BUSINESS: what the company does, year founded, headquarters (1-3 sentences).

2. FULL PRODUCT/SERVICE CATALOG: list ALL major products, services, \
product categories and subcategories. Don't limit yourself to 3-5 items \u2014 write as many as you find. \
For each product/service, use specific names, not generic categories. \
For example: not "roofing materials", but "metal tiles, corrugated sheets, soft roofing, \
gutter systems, skylight windows, roofing accessories".

3. BRANDS AND PRODUCT LINES: own brands, trademarks, product lines, \
key product names.

4. KEY FACTS: year founded, employee count, revenue (if public), \
number of customers/users, awards, licenses, certifications.

5. TARGET MARKET: geography, main customer segments (B2B/B2C), \
typical buyers and their use cases.

6. POSITIONING: market leader / mid-segment / niche player, \
price segment (premium/mid/economy), main competitive advantages.

Rules:
- Find REAL information from the internet. Don't invent products that don't exist.
- If precise data about some aspect is unavailable \u2014 skip it, but don't make things up.
- If precise data is unavailable \u2014 make a reasonable inference from the name{domain_hint_short}, \
but explicitly note it.
- Write factually, without marketing clich\u00e9s.
- Plain text only, no headers or bullet lists."""


# ---------------------------------------------------------------------------
# Prompt templates — Competitor suggestions
# ---------------------------------------------------------------------------

_COMPETITORS_PROMPT_RU = """Назови 5-10 прямых конкурентов компании \u00ab{brand_name}\u00bb на том же рынке.

О компании: {brand_description}

{exclude_block}

Верни JSON-объект: {{"competitors": ["Конкурент 1", "Конкурент 2", ...]}}

Правила:
- Только прямые конкуренты в том же сегменте рынка
- Названия компаний/брендов, как их знают потребители
- Не включай сам бренд \u00ab{brand_name}\u00bb
- Не придумывай несуществующие компании
- Только JSON, без пояснений"""

_COMPETITORS_PROMPT_EN = """List 5-10 direct competitors of "{brand_name}" in the same market.

About the company: {brand_description}

{exclude_block}

Return a JSON object: {{"competitors": ["Competitor 1", "Competitor 2", ...]}}

Rules:
- Only direct competitors in the same market segment
- Use brand/company names as consumers know them
- Do not include "{brand_name}" itself
- Do not invent non-existent companies
- JSON only, no explanations"""


# ---------------------------------------------------------------------------
# Prompt templates — Category suggestions
# ---------------------------------------------------------------------------

_CATEGORIES_PROMPT_RU = """Придумай 8-12 пользовательских сценариев для аудита видимости бренда «{brand_name}» в AI-ассистентах (ChatGPT, YandexGPT, DeepSeek).

О компании: {brand_description}

Каждый сценарий — это РЕАЛЬНАЯ ЖИЗНЕННАЯ СИТУАЦИЯ, в которой человек обращается к AI-ассистенту за советом, рекомендацией или помощью в выборе. Сценарий описывает КОНКРЕТНУЮ потребность, контекст и мотивацию пользователя.

{exclude_block}

Верни JSON-объект: {{"categories": ["Сценарий 1", "Сценарий 2", ...]}}

Правила:
- Сценарий — это описание ситуации от лица пользователя, 7-15 слов
- Должно звучать как МЫСЛЬ или ЗАДАЧА реального человека, а НЕ как запрос в поисковике
- Примеры ХОРОШИХ сценариев (для застройщика): «Выбираю новостройку для семьи с детьми рядом с метро», «Сравниваю застройщиков по надёжности и срокам сдачи», «Переживаю что застройщик задержит сдачу дома»
- Примеры ПЛОХИХ: «ЖК бизнес класса» (ключевик), «Паркинги» (продукт), «Покупка квартиры» (слишком общо, нет контекста)
- Охвати РАЗНЫЕ аспекты: выбор, сравнение, страхи/опасения, конкретные потребности, ценовые вопросы, отзывы, послепродажное обслуживание
- Каждый сценарий на русском
- Не придумывай нерелевантные сценарии
- НЕ упоминай название бренда «{brand_name}» в сценариях
- Только JSON, без пояснений"""

_CATEGORIES_PROMPT_EN = """Create 8-12 user scenarios for auditing brand visibility of "{brand_name}" in AI assistants (ChatGPT, Gemini, DeepSeek).

About the company: {brand_description}

Each scenario is a REAL-LIFE SITUATION where a person turns to an AI assistant for advice, recommendations, or help making a decision. The scenario describes a SPECIFIC need, context, and user motivation.

{exclude_block}

Return a JSON object: {{"categories": ["Scenario 1", "Scenario 2", ...]}}

Rules:
- A scenario describes a situation from the user's perspective, 7-15 words
- Must sound like a real person's THOUGHT or TASK, NOT a search engine query
- Good examples (for a homebuilder): "Choosing a new build apartment for a family with kids near the subway", "Comparing developers by reliability and completion timelines", "Worried the developer might delay handover of my apartment"
- BAD examples: "Luxury apartments" (keyword), "Parking" (product), "Buying a flat" (too generic, no context)
- Cover DIFFERENT aspects: choosing, comparing, fears/concerns, specific needs, pricing, reviews, after-sales service
- Each scenario in English
- Don't invent irrelevant scenarios
- Do NOT mention the brand name "{brand_name}" in scenarios
- JSON only, no explanations"""


# ---------------------------------------------------------------------------
# Public API — Brand research
# ---------------------------------------------------------------------------


async def research_brand(
    brand_name: str,
    api_key: str,
    domain: str | None = None,
    market: str = "ru",
    model: str = "gpt-4o",
    perplexity_api_key: str | None = None,
    gemini_api_key: str | None = None,
) -> str:
    """Research a brand and return a comprehensive company profile.

    The profile focuses on a complete product/service catalog so that
    the prompt generator can anchor every query to the company's real
    offerings.

    Returns:
        Plain text company profile with detailed product catalog.

    Raises:
        ValueError: If the response is empty.
    """
    # Build domain hint
    if domain:
        if market == "ru":
            domain_hint = f"Домен компании: {domain}"
            domain_hint_short = " и домена"
        else:
            domain_hint = f"Company domain: {domain}"
            domain_hint_short = " and domain"
    else:
        domain_hint = ""
        domain_hint_short = ""

    template = _RESEARCH_PROMPT_RU if market == "ru" else _RESEARCH_PROMPT_EN
    user_prompt = template.format(
        brand_name=brand_name,
        domain_hint=domain_hint,
        domain_hint_short=domain_hint_short,
    )

    description = await _call_llm(
        prompt=user_prompt,
        openai_api_key=api_key,
        perplexity_api_key=perplexity_api_key,
        gemini_api_key=gemini_api_key,
        max_tokens=8000,
    )

    if not description:
        raise ValueError("Empty response from LLM for brand research")

    # Ensure the free-text description ends on a complete sentence
    # (Gemini may stop mid-sentence if MAX_TOKENS is hit)
    description = _trim_to_last_sentence(description)

    # Hard upper bound (keep sane)
    if len(description) > 15000:
        description = _trim_to_last_sentence(description[:15000])

    return description


# ---------------------------------------------------------------------------
# Public API — Competitor suggestions
# ---------------------------------------------------------------------------


async def suggest_competitors(
    brand_name: str,
    brand_description: str,
    api_key: str,
    market: str = "ru",
    perplexity_api_key: str | None = None,
    gemini_api_key: str | None = None,
    existing: list[str] | None = None,
) -> list[str]:
    """Suggest direct competitors for a brand."""
    exclude_block = ""
    if existing:
        names = ", ".join(existing)
        if market == "ru":
            exclude_block = f"Уже добавлены (НЕ повторяй): {names}"
        else:
            exclude_block = f"Already added (do NOT repeat): {names}"

    template = _COMPETITORS_PROMPT_RU if market == "ru" else _COMPETITORS_PROMPT_EN
    prompt = template.format(
        brand_name=brand_name,
        brand_description=brand_description or brand_name,
        exclude_block=exclude_block,
    )

    raw = await _call_llm(
        prompt=prompt,
        openai_api_key=api_key,
        perplexity_api_key=perplexity_api_key,
        gemini_api_key=gemini_api_key,
        max_tokens=2000,
        temperature=0.3,
        use_grounding=False,
    )

    if not raw:
        return []

    result = _parse_json_list(raw, "competitors")

    # Filter out the brand itself and existing competitors
    brand_lower = brand_name.lower()
    existing_lower = {x.lower() for x in (existing or [])}
    result = [c for c in result if c.lower() != brand_lower and c.lower() not in existing_lower]

    return result[:10]


# ---------------------------------------------------------------------------
# Public API — Category suggestions
# ---------------------------------------------------------------------------


async def suggest_categories(
    brand_name: str,
    brand_description: str,
    api_key: str,
    market: str = "ru",
    perplexity_api_key: str | None = None,
    gemini_api_key: str | None = None,
    existing: list[str] | None = None,
) -> list[str]:
    """Suggest product categories / user scenarios for brand audit."""
    exclude_block = ""
    if existing:
        names = ", ".join(existing)
        if market == "ru":
            exclude_block = f"Уже добавлены (НЕ повторяй): {names}"
        else:
            exclude_block = f"Already added (do NOT repeat): {names}"

    template = _CATEGORIES_PROMPT_RU if market == "ru" else _CATEGORIES_PROMPT_EN
    prompt = template.format(
        brand_name=brand_name,
        brand_description=brand_description or brand_name,
        exclude_block=exclude_block,
    )

    raw = await _call_llm(
        prompt=prompt,
        openai_api_key=api_key,
        perplexity_api_key=perplexity_api_key,
        gemini_api_key=gemini_api_key,
        max_tokens=2000,
        temperature=0.3,
        use_grounding=False,
    )

    if not raw:
        return []

    result = _parse_json_list(raw, "categories")

    # Filter out existing categories
    existing_lower = {x.lower() for x in (existing or [])}
    result = [c for c in result if c.lower() not in existing_lower]

    return result[:12]


# ---------------------------------------------------------------------------
# Prompt templates — Sub-brand suggestions
# ---------------------------------------------------------------------------

_SUB_BRANDS_PROMPT_RU = """Назови суб-бренды, дочерние бренды, продуктовые линейки или отдельные продукты компании «{brand_name}», \
которые имеют собственное имя и узнаваемость на рынке.

О компании: {brand_description}

{exclude_block}

Верни JSON-объект: {{"sub_brands": [{{"name": "Суб-бренд 1", "aliases": ["Алиас 1", "Алиас 2"]}}, ...]}}

Правила:
- Только реальные суб-бренды / продуктовые линейки с собственным названием
- Алиасы — альтернативные написания, сокращения, варианты названий (если есть)
- Максимум 15 суб-брендов
- Не включай головную компанию «{brand_name}» как суб-бренд
- Только JSON, без пояснений"""

_SUB_BRANDS_PROMPT_EN = """List sub-brands, child brands, product lines, or standalone products of "{brand_name}" \
that have their own name and market recognition.

About the company: {brand_description}

{exclude_block}

Return a JSON object: {{"sub_brands": [{{"name": "Sub-brand 1", "aliases": ["Alias 1", "Alias 2"]}}, ...]}}

Rules:
- Only real sub-brands / product lines with their own name
- Aliases are alternative spellings, abbreviations, name variations (if any)
- Maximum 15 sub-brands
- Do NOT include the parent company "{brand_name}" as a sub-brand
- JSON only, no explanations"""


def _parse_json_sub_brands(text: str) -> list[dict]:
    """Parse sub-brand suggestions from LLM response.

    Expected format: {"sub_brands": [{"name": "...", "aliases": [...]}, ...]}
    Falls back to _parse_json_list if structured format not found.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        items = data.get("sub_brands", data) if isinstance(data, dict) else data
        if isinstance(items, list):
            result = []
            for item in items:
                if isinstance(item, dict) and "name" in item:
                    result.append(
                        {
                            "name": str(item["name"]).strip(),
                            "aliases": [str(a).strip() for a in item.get("aliases", []) if str(a).strip()],
                        }
                    )
                elif isinstance(item, str):
                    result.append({"name": item.strip(), "aliases": []})
            return result
    except json.JSONDecodeError:
        pass
    # Fallback: try simple list parsing
    names = _parse_json_list(text, "sub_brands")
    return [{"name": n, "aliases": []} for n in names]


# ---------------------------------------------------------------------------
# Public API — Sub-brand suggestions
# ---------------------------------------------------------------------------


async def suggest_sub_brands(
    brand_name: str,
    brand_description: str,
    api_key: str,
    market: str = "ru",
    perplexity_api_key: str | None = None,
    gemini_api_key: str | None = None,
    existing: list[str] | None = None,
) -> list[dict]:
    """Suggest sub-brands / product lines for a brand."""
    exclude_block = ""
    if existing:
        names = ", ".join(existing)
        if market == "ru":
            exclude_block = f"Уже добавлены (НЕ повторяй): {names}"
        else:
            exclude_block = f"Already added (do NOT repeat): {names}"

    template = _SUB_BRANDS_PROMPT_RU if market == "ru" else _SUB_BRANDS_PROMPT_EN
    prompt = template.format(
        brand_name=brand_name,
        brand_description=brand_description or brand_name,
        exclude_block=exclude_block,
    )

    raw = await _call_llm(
        prompt=prompt,
        openai_api_key=api_key,
        perplexity_api_key=perplexity_api_key,
        gemini_api_key=gemini_api_key,
        max_tokens=2000,
        temperature=0.3,
        use_grounding=True,
    )

    if not raw:
        return []

    result = _parse_json_sub_brands(raw)

    # Filter out existing sub-brands
    existing_lower = {x.lower() for x in (existing or [])}
    brand_lower = brand_name.lower()
    result = [sb for sb in result if sb["name"].lower() not in existing_lower and sb["name"].lower() != brand_lower]

    return result[:15]


# ---------------------------------------------------------------------------
# Prompt templates — Scenario description generation
# ---------------------------------------------------------------------------

_SCENARIO_DESC_PROMPT_RU = """Напиши развёрнутое описание пользовательского сценария для мониторинга видимости бренда в AI-ассистентах.

Отрасль: {brand_description}
Сценарий: «{scenario_name}»

Требования к описанию:
1. Пиши ОТ ЛИЦА ПОЛЬЗОВАТЕЛЯ — кто он, какая у него жизненная ситуация, что он ищет
2. Перечисли 4-6 конкретных вопросов/тем, которые пользователь задаёт AI-ассистенту
3. НЕ упоминай название «{brand_name}» и конкретные продукты компании
4. Описание — РОВНО 3 предложения. Каждое предложение ОБЯЗАТЕЛЬНО заканчивается точкой
5. Максимум 500 символов

Пример для сценария «Ремонт квартиры» (строительный магазин):
«Пользователь планирует ремонт квартиры и обращается к AI за помощью на каждом этапе — от выбора материалов для стен и пола до поиска надёжных подрядчиков. Спрашивает какие бренды ламината и краски лучше по соотношению цена-качество, где выгоднее покупать стройматериалы оптом, читает ли AI отзывы на строительные магазины. Интересуется последовательностью ремонтных работ, нормами расхода материалов, сравнением сетевых гипермаркетов и локальных строительных баз.»

Пример для сценария «Выбор новостройки для семьи» (застройщик):
«Семья с детьми выбирает квартиру в новостройке и спрашивает AI-ассистента о лучших жилых комплексах в своём городе — сравнивает застройщиков по надёжности, срокам сдачи, качеству строительства и наличию социальной инфраструктуры. Интересуется рейтингами застройщиков, отзывами жильцов уже сданных домов, наличием школ и детских садов рядом, транспортной доступностью. Спрашивает про ипотечные программы от застройщиков, Trade-in квартир, рассрочку и акции.»

Теперь напиши такое же подробное описание для сценария «{scenario_name}». Начинай сразу с текста описания, БЕЗ заголовков и преамбул:"""

_SCENARIO_DESC_PROMPT_EN = """Write a detailed user scenario description for brand visibility monitoring in AI assistants.

Industry: {brand_description}
Scenario: "{scenario_name}"

Requirements:
1. Write FROM THE USER'S PERSPECTIVE — who they are, their life situation, what they're looking for
2. List 4-6 specific questions/topics the user asks an AI assistant about
3. Do NOT mention "{brand_name}" or specific company products
4. Description — EXACTLY 3 sentences. Each sentence MUST end with a period
5. Maximum 500 characters

Example for "Home Renovation" (hardware store):
"A homeowner is planning a renovation and turns to AI for help at every stage — from choosing wall and floor materials to finding reliable contractors. They ask which laminate and paint brands offer the best value for money, where to buy building materials in bulk at the best price, and whether AI has insights on store reviews. They're interested in the proper sequence of renovation work, material consumption rates, and comparisons between chain stores and local building supply centers."

Example for "Choosing a New Build for a Family" (developer):
"A family with children is choosing an apartment in a new development and asks an AI assistant about the best residential complexes in their city — comparing developers by reliability, delivery timelines, construction quality, and available social infrastructure. They're interested in developer ratings, reviews from residents of completed buildings, nearby schools and kindergartens, and transport accessibility. They ask about developer mortgage programs, trade-in options, installment plans, and promotions."

Now write an equally detailed description for the scenario "{scenario_name}". Start directly with the description text, NO headers or preambles:"""


# ---------------------------------------------------------------------------
# Public API — Scenario description generation
# ---------------------------------------------------------------------------


async def suggest_scenario_description(
    brand_name: str,
    brand_description: str,
    scenario_name: str,
    api_key: str,
    market: str = "ru",
    perplexity_api_key: str | None = None,
    gemini_api_key: str | None = None,
) -> str:
    """Generate a 1-2 sentence description for a scenario.

    Uses the brand context to create a relevant description that
    explains what the scenario covers.
    """
    template = _SCENARIO_DESC_PROMPT_RU if market == "ru" else _SCENARIO_DESC_PROMPT_EN
    prompt = template.format(
        brand_name=brand_name,
        brand_description=brand_description or brand_name,
        scenario_name=scenario_name,
    )

    raw = await _call_llm(
        prompt=prompt,
        openai_api_key=api_key,
        perplexity_api_key=perplexity_api_key,
        gemini_api_key=gemini_api_key,
        max_tokens=2000,
        temperature=0.7,
        use_grounding=True,
    )

    if not raw:
        return ""

    result = _clean_scenario_description(raw)
    return result


def _clean_scenario_description(raw: str) -> str:
    """Clean LLM output for scenario description: strip headers, markdown, quotes."""
    text = raw.strip()

    # Remove markdown headers (## Title, **Title:** etc.)
    # Remove lines that look like headers/labels before the actual content
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            continue
        # Skip markdown headers
        if stripped.startswith("#"):
            continue
        # Skip bold labels like "**Описание:**" or "**Description:**"
        if re.match(r"^\*\*[^*]+:\*\*\s*$", stripped):
            continue
        # Strip leading bold label from a content line: "**Label:** actual text" → "actual text"
        stripped = re.sub(r"^\*\*[^*]+:\*\*\s*", "", stripped)
        # Strip other markdown formatting
        stripped = stripped.strip("*").strip("_").strip()
        if stripped:
            cleaned_lines.append(stripped)

    text = " ".join(cleaned_lines)

    # Remove surrounding quotes
    text = text.strip().strip('"').strip("'").strip("\u00ab\u00bb").strip()

    # Trim to last complete sentence
    text = _trim_to_last_sentence(text)

    return text
