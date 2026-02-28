"""Structural & Ranking Parser — Pipeline Step 3.

Detects the structural format of an LLM response and assigns
position weights to brands based on their rank:

  - Numbered list:  Rank 1 = 1.0, Rank 2 = 0.9, ..., Rank 10+ = 0.1
  - Bulleted list:  All items get uniform weight 0.8
  - Narrative text:  First-screen proximity rule (earlier = higher weight)
  - Table:          Parsed like numbered list by row order
  - Mixed:          Best effort combination
"""

from __future__ import annotations

import re
import logging

from app.analysis.types import (
    BrandMention,
    StructureAnalysis,
    StructureType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structure detection patterns
# ---------------------------------------------------------------------------

# Numbered list: "1. ", "1) ", "1: "
_NUMBERED_PATTERN = re.compile(
    r"^\s*(\d+)\s*[.):\-]\s+(.+)$",
    re.MULTILINE,
)

# Bulleted list: "- ", "* ", "• "
_BULLET_PATTERN = re.compile(
    r"^\s*[-*•]\s+(.+)$",
    re.MULTILINE,
)

# Markdown table: "| col1 | col2 |"
_TABLE_ROW_PATTERN = re.compile(
    r"^\s*\|(.+)\|\s*$",
    re.MULTILINE,
)

# Table separator: "|---|---|" or "| --- | --- |"
_TABLE_SEP_PATTERN = re.compile(
    r"^\s*\|[\s\-:|]+\|\s*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Position weight calculation
# ---------------------------------------------------------------------------


def position_weight_numbered(rank: int) -> float:
    """Calculate position weight for numbered lists.

    Rank 1 → 1.0, Rank 2 → 0.9, ..., Rank 10+ → 0.1
    """
    if rank <= 0:
        return 0.0
    if rank >= 10:
        return 0.1
    return round(1.0 - (rank - 1) * 0.1, 1)


def position_weight_bullet() -> float:
    """Uniform weight for bulleted lists."""
    return 0.8


def position_weight_narrative(char_offset: int, total_length: int) -> float:
    """Calculate weight by proximity to the start of the text.

    First 20% of text → 1.0
    20-50% → 0.7
    50-80% → 0.4
    80-100% → 0.2
    """
    if total_length == 0:
        return 0.5
    ratio = char_offset / total_length
    if ratio < 0.2:
        return 1.0
    if ratio < 0.5:
        return 0.7
    if ratio < 0.8:
        return 0.4
    return 0.2


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------


def detect_structure(text: str) -> StructureType:
    """Detect the dominant structural format of the response text."""
    numbered_matches = _NUMBERED_PATTERN.findall(text)
    bullet_matches = _BULLET_PATTERN.findall(text)
    table_rows = _TABLE_ROW_PATTERN.findall(text)
    table_seps = _TABLE_SEP_PATTERN.findall(text)

    has_table = len(table_rows) >= 2 and len(table_seps) >= 1
    has_numbered = len(numbered_matches) >= 2
    has_bullets = len(bullet_matches) >= 2

    # Priority: table > numbered > bulleted > mixed > narrative
    if has_table and not has_numbered and not has_bullets:
        return StructureType.TABLE
    if has_numbered and not has_bullets:
        return StructureType.NUMBERED_LIST
    if has_bullets and not has_numbered:
        return StructureType.BULLETED_LIST
    if (has_numbered and has_bullets) or (has_table and (has_numbered or has_bullets)):
        return StructureType.MIXED

    return StructureType.NARRATIVE


def _extract_list_items(text: str, structure: StructureType) -> list[str]:
    """Extract ordered list items from text based on detected structure."""
    if structure == StructureType.NUMBERED_LIST:
        matches = _NUMBERED_PATTERN.findall(text)
        # Sort by number and return text part
        sorted_matches = sorted(matches, key=lambda m: int(m[0]))
        return [m[1].strip() for m in sorted_matches]

    if structure == StructureType.BULLETED_LIST:
        matches = _BULLET_PATTERN.findall(text)
        return [m.strip() for m in matches]

    if structure == StructureType.TABLE:
        rows = _TABLE_ROW_PATTERN.findall(text)
        # Filter out separator rows and header-like rows
        items = []
        for row in rows:
            cells = [c.strip() for c in row.split("|") if c.strip()]
            if cells and not all(c.replace("-", "").replace(":", "").strip() == "" for c in cells):
                items.append(cells[0])  # First column typically has brand name
        return items

    if structure == StructureType.MIXED:
        # Try numbered first, then bulleted
        items = _extract_list_items(text, StructureType.NUMBERED_LIST)
        if not items:
            items = _extract_list_items(text, StructureType.BULLETED_LIST)
        return items

    return []


def _find_brand_in_items(
    items: list[str],
    brand_name: str,
    aliases: list[str] | None = None,
) -> int:
    """Find the rank (1-based) of a brand in list items. Returns 0 if not found."""
    all_names = [brand_name] + (aliases or [])
    for idx, item in enumerate(items):
        for name in all_names:
            if name and re.search(
                rf"(?<![а-яА-ЯёЁa-zA-Z0-9]){re.escape(name)}(?![а-яА-ЯёЁa-zA-Z0-9])",
                item,
                re.IGNORECASE,
            ):
                return idx + 1
    return 0


def analyze_structure(
    text: str,
    brands: list[BrandMention],
) -> StructureAnalysis:
    """Analyze text structure and assign position weights to brand mentions.

    Args:
        text: Cleaned response text.
        brands: List of BrandMention objects (target + competitors) to update.

    Returns:
        StructureAnalysis with detected structure and brands in order.
    """
    structure_type = detect_structure(text)
    items = _extract_list_items(text, structure_type)
    total_length = len(text)

    brands_in_list: list[str] = []

    for brand in brands:
        if not brand.is_mentioned:
            brand.position_rank = 0
            brand.position_weight = 0.0
            continue

        if structure_type in (StructureType.NUMBERED_LIST, StructureType.TABLE, StructureType.MIXED):
            rank = _find_brand_in_items(items, brand.name, brand.aliases_matched)
            brand.position_rank = rank
            brand.position_weight = position_weight_numbered(rank) if rank > 0 else 0.3
            if rank > 0:
                brands_in_list.append(brand.name)

        elif structure_type == StructureType.BULLETED_LIST:
            rank = _find_brand_in_items(items, brand.name, brand.aliases_matched)
            brand.position_rank = rank
            brand.position_weight = position_weight_bullet() if rank > 0 else 0.3
            if rank > 0:
                brands_in_list.append(brand.name)

        else:
            # Narrative: use char_offset proximity
            brand.position_rank = 0
            brand.position_weight = position_weight_narrative(
                brand.char_offset if brand.char_offset >= 0 else total_length,
                total_length,
            )

    return StructureAnalysis(
        structure_type=structure_type,
        total_items=len(items),
        brands_in_list=brands_in_list,
    )
