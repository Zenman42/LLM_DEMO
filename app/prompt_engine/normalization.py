"""Layer 1: Input & Normalization Gateway.

Accepts an audit profile and normalizes brand names, competitors,
and categories for consistent downstream processing.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from app.prompt_engine.types import AuditProfile, ScenarioConfig
from app.schemas.prompt_engine import AuditProfileCreate

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Basic text normalization: strip, collapse whitespace, NFC unicode."""
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_brand(brand: str) -> str:
    """Normalize a brand name — preserve casing but clean whitespace."""
    return normalize_text(brand)


def deduplicate_list(items: list[str]) -> list[str]:
    """Deduplicate a list while preserving order (case-insensitive)."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(normalize_text(item))
    return result


def generate_brand_aliases(brand: str) -> list[str]:
    """Auto-generate common aliases for a brand name.

    Examples:
        "Ahrefs" → ["ahrefs"]
        "SEMrush" → ["semrush", "sem rush"]
        "Яндекс.Метрика" → ["яндекс метрика", "yandex metrika"]
    """
    aliases: list[str] = []

    # Lowercase variant
    lower = brand.lower()
    if lower != brand:
        aliases.append(lower)

    # Split CamelCase → "SEMrush" → "SEM rush" → "sem rush"
    camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", brand)
    if camel_split != brand:
        aliases.append(camel_split.lower())

    # Replace dots/hyphens with spaces: "Яндекс.Метрика" → "Яндекс Метрика"
    dot_replaced = re.sub(r"[.\-_]", " ", brand)
    if dot_replaced != brand:
        aliases.append(dot_replaced)
        aliases.append(dot_replaced.lower())

    return deduplicate_list(aliases)


def normalize_profile(input_data: AuditProfileCreate) -> AuditProfile:
    """Layer 1 main entry: normalize an AuditProfileCreate into an AuditProfile.

    Steps:
        1. Normalize brand name and aliases
        2. Auto-generate additional aliases
        3. Deduplicate competitors (excluding the brand itself)
        4. Normalize categories
        5. Validate and set market/geo defaults
    """
    # 1. Brand normalization
    brand = normalize_brand(input_data.brand)

    # 2. Merge user-provided aliases with auto-generated ones
    user_aliases = [normalize_brand(a) for a in input_data.brand_aliases]
    auto_aliases = generate_brand_aliases(brand)
    all_aliases = deduplicate_list(user_aliases + auto_aliases)

    # Remove the brand itself from aliases
    brand_lower = brand.lower()
    all_aliases = [a for a in all_aliases if a.lower() != brand_lower]

    # 3. Competitors: normalize, deduplicate, exclude brand
    competitors = deduplicate_list([normalize_brand(c) for c in input_data.competitors])
    competitors = [
        c for c in competitors if c.lower() != brand_lower and c.lower() not in {a.lower() for a in all_aliases}
    ]

    if not competitors:
        logger.warning("No valid competitors after normalization for brand=%s", brand)

    # 4. Categories / Scenarios: normalize, convert to ScenarioConfig
    scenarios: list[ScenarioConfig] = []
    seen_names: set[str] = set()
    for raw_cat in input_data.categories:
        sc = ScenarioConfig.from_raw(raw_cat)
        sc.name = normalize_text(sc.name)
        if sc.description:
            sc.description = normalize_text(sc.description)
        if sc.context_hints:
            sc.context_hints = normalize_text(sc.context_hints)
        name_key = sc.name.lower()
        if name_key and name_key not in seen_names:
            seen_names.add(name_key)
            scenarios.append(sc)

    if not scenarios:
        raise ValueError("At least one product/service category is required")

    # 5. Market and geo
    market = input_data.market.lower().strip()
    if market not in ("ru", "en"):
        market = "ru"

    geo = normalize_text(input_data.geo)

    # 6. Golden facts: normalize, deduplicate
    golden_facts = []
    if hasattr(input_data, "golden_facts") and input_data.golden_facts:
        golden_facts = deduplicate_list([normalize_text(f) for f in input_data.golden_facts])

    # 7. Brand description: normalize whitespace
    brand_description = ""
    if hasattr(input_data, "brand_description") and input_data.brand_description:
        brand_description = normalize_text(input_data.brand_description)

    logger.info(
        "Normalized profile: brand=%s, aliases=%d, competitors=%d, scenarios=%d, golden_facts=%d, brand_desc=%d chars, market=%s",
        brand,
        len(all_aliases),
        len(competitors),
        len(scenarios),
        len(golden_facts),
        len(brand_description),
        market,
    )

    return AuditProfile(
        brand=brand,
        brand_aliases=all_aliases,
        competitors=competitors,
        scenarios=scenarios,
        market=market,
        geo=geo,
        golden_facts=golden_facts,
        brand_description=brand_description,
    )
