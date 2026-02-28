"""Layer 2: Combinatorial Matrix Engine.

Builds prompt skeletons by crossing Persona x Intent axes for each category.
Uses hardcoded template taxonomies for each (Persona, Intent) pair.
"""

from __future__ import annotations

import hashlib
import logging
from itertools import product

from app.prompt_engine.types import AuditProfile, Intent, Persona, PromptSkeleton

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template registry: (Persona, Intent, Market) → list of template strings
#
# Placeholders:
#   {brand}       — target brand name
#   {competitor}  — single competitor (iterated)
#   {competitors} — comma-joined competitor list
#   {category}    — product/service category
#   {geo}         — geographic context
# ---------------------------------------------------------------------------

_TEMPLATES_RU: dict[tuple[Persona, Intent], list[str]] = {
    # ── BEGINNER ──────────────────────────────────────────────────────────
    (Persona.BEGINNER, Intent.INFORMATIONAL): [
        "Объясни простыми словами, что такое {category} и зачем это нужно",
        "Расскажи как для новичка, что такое {category} и как этим пользоваться",
        "Я ничего не знаю про {category}, расскажи основы простым языком",
    ],
    (Persona.BEGINNER, Intent.NAVIGATIONAL): [
        "Где найти официальный сайт {brand} и посмотреть условия — {category}",
        "Как обратиться в {brand}, чтобы узнать про {category}",
        "Где посмотреть цены {brand} на {category}",
    ],
    (Persona.BEGINNER, Intent.COMPARATIVE): [
        "Посоветуй лучший вариант в сфере {category} для начинающего",
        "Что лучше выбрать — {brand} или {competitor} в сфере {category}",
        "Топ-5 лучших вариантов {category}",
        "Какие лучшие варианты {category} для тех, кто только начинает разбираться",
    ],
    (Persona.BEGINNER, Intent.TRANSACTIONAL): [
        "С чего начать, если хочу попробовать {category}",
        "Какой самый доступный вариант {category} для начинающего",
        "Есть ли у {brand} что-то для новичков в {category}",
    ],
    # ── CTO ───────────────────────────────────────────────────────────────
    (Persona.CTO, Intent.INFORMATIONAL): [
        "Расскажи подробно, как устроена сфера {category} и какие ключевые игроки на рынке",
        "Какие стандарты качества и сертификации существуют в сфере {category}",
        "Какие ключевые характеристики нужно оценивать при выборе {category}",
    ],
    (Persona.CTO, Intent.NAVIGATIONAL): [
        "Где найти подробную информацию о {brand} в сфере {category}",
        "Какие гарантии даёт {brand} в сфере {category}",
        "Какие возможности и особенности есть у {brand} в сфере {category}",
    ],
    (Persona.CTO, Intent.COMPARATIVE): [
        "Сравни {brand} и {competitor} в сфере {category} — по качеству, надёжности, ценам",
        "Что лучше для крупного бизнеса — {brand} или {competitor} — в сфере {category}",
        "Составь сравнение топ-5 лидеров в сфере {category} по ключевым параметрам",
        "Какой вариант {category} обеспечивает лучшее качество",
    ],
    (Persona.CTO, Intent.TRANSACTIONAL): [
        "Какие условия предлагает {brand} в сфере {category} для крупных клиентов",
        "Сколько стоят услуги лидеров рынка в сфере {category}",
        "Как заказать {category} с индивидуальными условиями",
    ],
    # ── SKEPTIC ───────────────────────────────────────────────────────────
    (Persona.SKEPTIC, Intent.INFORMATIONAL): [
        "Какие реальные проблемы и ограничения есть в сфере {category}",
        "Что обычно замалчивают компании в сфере {category}",
        "Какие скрытые расходы бывают при выборе {category}",
    ],
    (Persona.SKEPTIC, Intent.NAVIGATIONAL): [
        "Где найти честные отзывы о {brand} в сфере {category}",
        "Есть ли судебные иски или скандалы у {brand} в сфере {category}",
        "Покажи негативные отзывы о {brand} — {category}",
    ],
    (Persona.SKEPTIC, Intent.COMPARATIVE): [
        "Какие недостатки у {brand} по сравнению с {competitor} — {category}",
        "Что лучше НЕ выбирать в сфере {category} и почему",
        "Назови главные минусы топ-5 компаний в сфере {category}",
        "На какие подводные камни обратить внимание при выборе {category}: {brand} vs {competitor}",
    ],
    (Persona.SKEPTIC, Intent.TRANSACTIONAL): [
        "Какие скрытые доплаты есть у {brand} в сфере {category}",
        "На чём экономят компании в сфере {category} за счёт клиентов",
        "Стоит ли переплачивать за {brand} — {category} — или есть более выгодные альтернативы",
    ],
    # ── MARKETER ──────────────────────────────────────────────────────────
    (Persona.MARKETER, Intent.INFORMATIONAL): [
        "Какие тренды в сфере {category} актуальны в текущем году",
        "Какие результаты можно получить от {category}",
        "На какие показатели обращать внимание при выборе {category}",
    ],
    (Persona.MARKETER, Intent.NAVIGATIONAL): [
        "Где найти реальные примеры и кейсы {brand} в сфере {category}",
        "Где посмотреть результаты и отзывы клиентов {brand} — {category}",
        "Какие награды и достижения есть у {brand} в сфере {category}",
    ],
    (Persona.MARKETER, Intent.COMPARATIVE): [
        "Что лучше по соотношению цена/качество — {brand} или {competitor} — {category}",
        "Сравни выгоду от {brand} и {competitor} в сфере {category}",
        "Топ-5 лучших вариантов {category} по соотношению цены и качества",
        "Что выбрать в сфере {category} для максимальной выгоды",
    ],
    (Persona.MARKETER, Intent.TRANSACTIONAL): [
        "Где выгоднее заказать {category}: у {brand} или {competitor}",
        "Какие скидки и акции есть у компаний в сфере {category} сейчас",
        "Какое предложение {brand} в сфере {category} лучше всего подходит для среднего бизнеса",
    ],
}

_TEMPLATES_EN: dict[tuple[Persona, Intent], list[str]] = {
    # ── BEGINNER ──────────────────────────────────────────────────────────
    (Persona.BEGINNER, Intent.INFORMATIONAL): [
        "Explain in simple terms what {category} is and why someone would need it",
        "As a complete beginner, tell me the basics of {category}",
        "I know nothing about {category}. Explain it like I'm five",
    ],
    (Persona.BEGINNER, Intent.NAVIGATIONAL): [
        "Where can I find {brand}'s official information about {category}",
        "How do I get started with {brand} for {category}",
        "Show me where to find {brand} pricing for {category}",
    ],
    (Persona.BEGINNER, Intent.COMPARATIVE): [
        "What is the best option for {category} for a complete beginner",
        "Which is better for a beginner: {brand} or {competitor} in {category}",
        "Top 5 most beginner-friendly options for {category}",
        "Rank the best {category} options for someone just getting started",
    ],
    (Persona.BEGINNER, Intent.TRANSACTIONAL): [
        "How do I get started with {category}",
        "What is the most affordable {category} option for beginners",
        "Does {brand} have something for newcomers in {category}",
    ],
    # ── CTO ───────────────────────────────────────────────────────────────
    (Persona.CTO, Intent.INFORMATIONAL): [
        "Explain how the {category} industry works and who the key players are",
        "What quality standards and certifications exist for {category}",
        "What key characteristics should I evaluate when choosing {category}",
    ],
    (Persona.CTO, Intent.NAVIGATIONAL): [
        "Where to find detailed information about {brand} in {category}",
        "What guarantees does {brand} offer for {category}",
        "What features and capabilities does {brand} have in {category}",
    ],
    (Persona.CTO, Intent.COMPARATIVE): [
        "Compare {brand} vs {competitor} in {category}: quality, reliability, pricing",
        "Which is better for large business: {brand} or {competitor} in {category}",
        "Create a comparison of the top 5 {category} leaders by key parameters",
        "Which {category} option provides the best quality",
    ],
    (Persona.CTO, Intent.TRANSACTIONAL): [
        "What terms does {brand} offer for {category} for large clients",
        "How much do market leaders charge for {category}",
        "Where to order {category} with customized terms",
    ],
    # ── SKEPTIC ───────────────────────────────────────────────────────────
    (Persona.SKEPTIC, Intent.INFORMATIONAL): [
        "What are the real problems and limitations of {category}",
        "What do companies in {category} usually hide from customers",
        "What hidden costs should I expect when choosing {category}",
    ],
    (Persona.SKEPTIC, Intent.NAVIGATIONAL): [
        "Where to find honest reviews about {brand} in {category}",
        "Are there any lawsuits or scandals involving {brand} in {category}",
        "Show me negative reviews about {brand} for {category}",
    ],
    (Persona.SKEPTIC, Intent.COMPARATIVE): [
        "What are {brand}'s disadvantages compared to {competitor} in {category}",
        "What should I NOT choose in {category} and why",
        "Name the top drawbacks of the top 5 {category} companies",
        "What pitfalls to watch for when choosing between {brand} vs {competitor} for {category}",
    ],
    (Persona.SKEPTIC, Intent.TRANSACTIONAL): [
        "What hidden fees does {brand} have in {category}",
        "What do {category} companies cut corners on to save money",
        "Is it worth overpaying for {brand} in {category} or are there better alternatives",
    ],
    # ── MARKETER ──────────────────────────────────────────────────────────
    (Persona.MARKETER, Intent.INFORMATIONAL): [
        "What are the current trends in {category} this year",
        "What results can you expect from {category}",
        "What indicators matter most when choosing {category}",
    ],
    (Persona.MARKETER, Intent.NAVIGATIONAL): [
        "Where to find real examples and case studies of {brand} in {category}",
        "Where to see client results and reviews for {brand} in {category}",
        "What awards and achievements does {brand} have in {category}",
    ],
    (Persona.MARKETER, Intent.COMPARATIVE): [
        "Which {category} offers the best price-to-quality ratio: {brand} or {competitor}",
        "Compare the value from {brand} vs {competitor} in {category}",
        "Top 5 best {category} options by price-to-quality ratio",
        "What to choose in {category} for the best value",
    ],
    (Persona.MARKETER, Intent.TRANSACTIONAL): [
        "Where is it more cost-effective to get {category}: {brand} or {competitor}",
        "What discounts and promotions do {category} companies offer right now",
        "Which {brand} offering in {category} is best for mid-size business",
    ],
}


def _skeleton_id(persona: Persona, intent: Intent, category: str, template_idx: int) -> str:
    """Generate a deterministic skeleton ID."""
    raw = f"{persona.value}:{intent.value}:{category}:{template_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_skeletons(
    profile: AuditProfile,
    personas: list[Persona] | None = None,
    intents: list[Intent] | None = None,
) -> list[PromptSkeleton]:
    """Build prompt skeletons by crossing all dimensions.

    For each (Persona, Intent, Category) triple:
      - Look up template strings
      - For templates with {competitor}, iterate over each competitor
      - For templates without {competitor}, render once

    Returns a list of PromptSkeleton objects ready for semantic expansion.
    """
    personas = personas or list(Persona)
    intents = intents or list(Intent)
    templates = _TEMPLATES_RU if profile.market == "ru" else _TEMPLATES_EN

    skeletons: list[PromptSkeleton] = []

    for persona, intent, category in product(personas, intents, profile.categories):
        key = (persona, intent)
        template_list = templates.get(key, [])
        if not template_list:
            logger.warning("No templates for persona=%s intent=%s", persona, intent)
            continue

        for idx, tmpl in enumerate(template_list):
            # Check if template needs per-competitor expansion
            if "{competitor}" in tmpl:
                for competitor in profile.competitors:
                    rendered = tmpl.format(
                        brand=profile.brand,
                        competitor=competitor,
                        competitors=", ".join(profile.competitors),
                        category=category,
                        geo=profile.geo or "",
                    )
                    skeletons.append(
                        PromptSkeleton(
                            template=rendered,
                            persona=persona,
                            intent=intent,
                            category=category,
                            brand=profile.brand,
                            competitors=profile.competitors,
                            market=profile.market,
                        )
                    )
            else:
                rendered = tmpl.format(
                    brand=profile.brand,
                    competitor=profile.competitors[0] if profile.competitors else "",
                    competitors=", ".join(profile.competitors),
                    category=category,
                    geo=profile.geo or "",
                )
                skeletons.append(
                    PromptSkeleton(
                        template=rendered,
                        persona=persona,
                        intent=intent,
                        category=category,
                        brand=profile.brand,
                        competitors=profile.competitors,
                        market=profile.market,
                    )
                )

    logger.info(
        "Generated %d skeletons from %d personas x %d intents x %d categories",
        len(skeletons),
        len(personas),
        len(intents),
        len(profile.categories),
    )
    return skeletons
