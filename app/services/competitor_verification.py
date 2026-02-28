"""Competitor verification service — checks if discovered entities are real competitors.

Uses Perplexity API (preferred, has web search) with OpenAI fallback.
Leverages the same LLM call pattern as brand_research.py.
"""

from __future__ import annotations

import json
import logging
import re
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.encryption import decrypt_value
from app.models.tenant import Tenant
from app.schemas.competitor import VerifyResult
from app.services.brand_research import _call_llm

logger = logging.getLogger(__name__)


_VERIFY_PROMPT_RU = """Бренд: {brand_name}
Описание: {brand_description}
Категории: {categories}
Известные конкуренты: {competitors}
Известные суббренды: {sub_brands}

Для каждой из следующих сущностей определи её тип:
{entity_list}

Ответь строго в JSON формате (массив):
[{{"entity": "...", "is_competitor": true/false, "is_sub_brand": true/false, "parent_brand": "имя родительского бренда или null", "is_alias": true/false, "alias_of": "имя бренда/конкурента/суббренда, алиасом которого является, или null", "explanation": "краткое пояснение"}}]

Правила:
- is_competitor=true — если это самостоятельная компания-конкурент в том же сегменте рынка, которой НЕТ в списке конкурентов выше
- is_sub_brand=true + parent_brand="X" — если это продукт, проект, услуга, подбренд, ЖК, линейка, сервис, дочерняя компания, инвестиционное подразделение ЛЮБОГО из известных брендов (бренд «{brand_name}» или ЛЮБОЙ из конкурентов)
- is_alias=true + alias_of="X" — если это альтернативное название (алиас) бренда «{brand_name}», любого из конкурентов или любого из суббрендов. Например: аббревиатура, сокращение, написание на другом языке, старое название, разговорное название, юридическое название
- ВАЖНО: при сопоставлении НЕ обращай внимания на организационно-правовую форму и кавычки. Например «ГК "МонАрх"» и «МонАрх» — это одна и та же компания. «ПИК» и «ГК ПИК» — одна компания. «Группа ЛСР» и «ЛСР» — одна компания. В таких случаях ставь is_alias=true
- Учитывай что «Компания-Инвест», «Компания-Девелопмент», «Компания Строй» и т.п. часто являются дочерними суббрендами основной «Компания»
- В parent_brand и alias_of пиши имя ТОЧНО как в списке конкурентов/суббрендов выше (копируй из списка!)
- Если сущность одновременно похожа на алиас — предпочитай is_alias (алиас важнее суббренда)
- Если это поставщик, партнёр, нерелевантная компания — is_competitor=false, is_sub_brand=false, is_alias=false
- Только JSON, без пояснений вне массива"""

_VERIFY_PROMPT_EN = """Brand: {brand_name}
Description: {brand_description}
Categories: {categories}
Known competitors: {competitors}
Known sub-brands: {sub_brands}

For each of the following entities, determine its type:
{entity_list}

Reply strictly in JSON format (array):
[{{"entity": "...", "is_competitor": true/false, "is_sub_brand": true/false, "parent_brand": "parent brand name or null", "is_alias": true/false, "alias_of": "name of brand/competitor/sub-brand this is an alias of, or null", "explanation": "brief explanation"}}]

Rules:
- is_competitor=true — if this is an independent competitor company in the same market segment that is NOT already in the competitors list above
- is_sub_brand=true + parent_brand="X" — if this is a product, project, service, sub-brand, subsidiary, investment arm, product line, or offering of ANY of the known brands ("{brand_name}" or ANY of the competitors)
- is_alias=true + alias_of="X" — if this is an alternative name (alias) for the brand "{brand_name}", any competitor, or any sub-brand. Examples: abbreviation, acronym, name in another language, old name, colloquial name, legal name
- IMPORTANT: ignore legal form prefixes when matching. E.g. "GK MonArch" and "MonArch" are the same company. "PIK Group" and "PIK" are the same. Match by core name, not by exact prefix. In such cases set is_alias=true
- Names like "Company Invest", "Company Development" etc. are often subsidiaries of "Company"
- parent_brand and alias_of must be copied EXACTLY from the competitors/sub-brands list above (use the exact string from the list!)
- If entity looks like both an alias and a sub-brand — prefer is_alias (alias takes priority over sub-brand)
- If it's a supplier, partner, or irrelevant company — is_competitor=false, is_sub_brand=false, is_alias=false
- JSON only, no explanations outside the array"""


def _parse_verify_response(raw: str) -> list[dict]:
    """Parse LLM verification response into list of dicts."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in text
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return []


async def verify_competitors(
    brand_name: str,
    brand_description: str,
    categories: list[str],
    entities: list[str],
    tenant_id: UUID,
    db: AsyncSession,
    competitors: list[str] | None = None,
    sub_brands: list[str] | None = None,
) -> list[VerifyResult]:
    """Verify whether discovered entities are real competitors or sub-brands.

    Uses Gemini (preferred), Perplexity, OpenAI fallback.
    """
    # Load tenant API keys
    tenant_q = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = tenant_q.scalar_one_or_none()
    if not tenant:
        return [VerifyResult(entity=e, is_competitor=False, explanation="Tenant not found") for e in entities]

    openai_key = decrypt_value(tenant.openai_api_key) if tenant.openai_api_key else None
    perplexity_key = decrypt_value(tenant.perplexity_api_key) if tenant.perplexity_api_key else None
    gemini_key = decrypt_value(tenant.gemini_api_key) if getattr(tenant, "gemini_api_key", None) else None

    logger.info(
        "[verify] Keys available: gemini=%s perplexity=%s openai=%s",
        bool(gemini_key),
        bool(perplexity_key),
        bool(openai_key),
    )

    if not openai_key and not perplexity_key and not gemini_key:
        return [VerifyResult(entity=e, is_competitor=False, explanation="No API keys configured") for e in entities]

    # Detect language from brand description
    is_ru = bool(re.search(r"[а-яА-Я]", brand_description[:200])) if brand_description else False
    template = _VERIFY_PROMPT_RU if is_ru else _VERIFY_PROMPT_EN

    # categories may be list of strings or list of dicts with "name" key
    cat_names = []
    for c in (categories or [])[:10]:
        if isinstance(c, str):
            cat_names.append(c)
        elif isinstance(c, dict):
            cat_names.append(c.get("name", c.get("label", str(c))))
        else:
            cat_names.append(str(c))
    categories_str = ", ".join(cat_names) if cat_names else "N/A"
    entity_list = "\n".join(f"- {e}" for e in entities)

    competitors_str = ", ".join(competitors) if competitors else "N/A"
    sub_brands_str = ", ".join(sub_brands) if sub_brands else "N/A"
    logger.info("[verify] Full competitors list (%d): %s", len(competitors) if competitors else 0, competitors)
    logger.info("[verify] Competitors string for prompt: %s", competitors_str)
    logger.info("[verify] Sub-brands list (%d): %s", len(sub_brands) if sub_brands else 0, sub_brands)

    prompt = template.format(
        brand_name=brand_name,
        brand_description=(brand_description[:500] if brand_description else brand_name),
        categories=categories_str,
        competitors=competitors_str,
        sub_brands=sub_brands_str,
        entity_list=entity_list,
    )

    logger.info("[verify] Calling LLM for %d entities: %s", len(entities), entities[:5])

    raw = await _call_llm(
        prompt=prompt,
        openai_api_key=openai_key or "",
        perplexity_api_key=perplexity_key,
        gemini_api_key=gemini_key,
        max_tokens=2000,
        temperature=0.1,
    )

    logger.info("[verify] LLM response length=%d, first 300 chars: %s", len(raw) if raw else 0, (raw or "")[:300])

    if not raw:
        return [VerifyResult(entity=e, is_competitor=False, explanation="Empty LLM response") for e in entities]

    parsed = _parse_verify_response(raw)
    logger.info("[verify] Parsed %d results from LLM response", len(parsed))

    # Build valid parents set for sub-brand validation
    valid_parents = {brand_name} | set(competitors or [])
    # Build lowercase lookup for fuzzy matching parent names
    parent_lookup = {p.lower().strip(): p for p in valid_parents}

    # Build valid alias targets (brand + competitors + sub-brands)
    valid_alias_targets = {brand_name} | set(competitors or []) | set(sub_brands or [])
    alias_lookup = {t.lower().strip(): t for t in valid_alias_targets}

    def _fuzzy_match(name: str, lookup: dict[str, str]) -> str | None:
        """Try to fuzzy-match a name against a lookup dict."""
        clean = name.lower().strip().strip("«»\"'")
        matched = lookup.get(clean)
        if matched:
            return matched
        # Substring match
        for key, val in lookup.items():
            if clean in key or key in clean:
                return val
        return None

    # Build results, ensuring all entities are covered
    results_map = {}
    for item in parsed:
        if isinstance(item, dict) and "entity" in item:
            is_sub = bool(item.get("is_sub_brand", False))
            parent = item.get("parent_brand") or None
            is_alias = bool(item.get("is_alias", False))
            alias_of = item.get("alias_of") or None

            # Validate alias_of target
            if is_alias and alias_of:
                if alias_of not in valid_alias_targets:
                    matched = _fuzzy_match(alias_of, alias_lookup)
                    if matched:
                        logger.info(
                            "[verify] Fuzzy-matched alias_of %r -> %r for %r", alias_of, matched, item["entity"]
                        )
                        alias_of = matched
                    else:
                        logger.info(
                            "[verify] Rejected alias_of %r for %r (not in valid targets)", alias_of, item["entity"]
                        )
                        alias_of = None
                        is_alias = False

            # Validate parent_brand is one of the known brands
            if is_sub and parent and parent not in valid_parents:
                matched = _fuzzy_match(parent, parent_lookup)
                if matched:
                    logger.info("[verify] Fuzzy-matched parent %r -> %r for %r", parent, matched, item["entity"])
                    parent = matched
                else:
                    logger.info(
                        "[verify] Rejected parent %r for %r (not in valid_parents: %s)",
                        parent,
                        item["entity"],
                        valid_parents,
                    )
                    parent = None
                    is_sub = False

            # If alias is set, clear sub_brand/competitor to avoid confusion
            if is_alias and alias_of:
                is_sub = False
                parent = None

            results_map[item["entity"].lower()] = VerifyResult(
                entity=item["entity"],
                is_competitor=bool(item.get("is_competitor", False)) and not is_alias,
                is_sub_brand=is_sub,
                parent_brand=parent if is_sub else None,
                is_alias=is_alias,
                alias_of=alias_of if is_alias else None,
                explanation=str(item.get("explanation", "")),
            )

    results = []
    for e in entities:
        if e.lower() in results_map:
            results.append(results_map[e.lower()])
        else:
            results.append(VerifyResult(entity=e, is_competitor=False, explanation="Not found in LLM response"))

    return results
