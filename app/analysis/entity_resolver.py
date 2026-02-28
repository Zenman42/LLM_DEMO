"""NER & Entity Resolution Engine — Pipeline Step 2.

Resolves brand mentions in LLM responses using:
  - Exact match against canonical name + aliases
  - Case-insensitive word boundary matching
  - Basic fuzzy/transliteration matching (Cyrillic ↔ Latin)
  - Phonetic cross-script matching (Winline ↔ Винлайн)
  - Competitor isolation

Outputs BrandMention objects for the target brand and each competitor.
"""

from __future__ import annotations

import re
import logging

from app.analysis.types import BrandMention, MentionType
from app.analysis.llm_entity_extractor import _names_match, _phonetic_match

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recommendation / comparison / negative signal patterns
# ---------------------------------------------------------------------------
_RECOMMEND_PATTERNS = [
    re.compile(r"(?:рекоменду[юем]|лучший выбор|топ[- ]?\d|лидер|лучше всего)", re.IGNORECASE),
    re.compile(r"(?:recommend|best choice|top pick|leader|number one|#\s*1)", re.IGNORECASE),
]

_NEGATIVE_PATTERNS = [
    re.compile(r"(?:недостат[коки]|проблем[аы]|минус[ыа]|слаб[оые]|хуже|не рекоменду)", re.IGNORECASE),
    re.compile(r"(?:disadvantage|problem|issue|weak|worse|not recommend|drawback|downside)", re.IGNORECASE),
]

_COMPARE_PATTERNS = [
    re.compile(r"(?:сравнени[ие]|сравни[мтв]|в отличие от|по сравнению|альтернатив[аы]|vs\.?|versus)", re.IGNORECASE),
    re.compile(r"(?:compar|unlike|alternative|vs\.?|versus|instead of|rather than)", re.IGNORECASE),
    # Balanced pros/cons language → comparison, not negative
    re.compile(r"(?:преимуществ\w*\s+и\s+недостат|плюс\w*\s+и\s+минус|pros?\s+and\s+cons?)", re.IGNORECASE),
    re.compile(r"(?:(?:у\s+)?каждого\s+(?:из\s+них\s+)?(?:есть\s+)?сво[ией])", re.IGNORECASE),
]

# Context window: number of characters around mention to extract.
# 500 chars each side captures enough for the LLM judge to see full context
# in structured responses with pros/cons sections.
_CONTEXT_WINDOW = 500


def _build_pattern(name: str) -> re.Pattern:
    """Build a word-boundary regex for a brand name.

    Handles both Latin and Cyrillic boundaries.
    """
    escaped = re.escape(name)
    # Use word-boundary for Latin, lookbehind/ahead for Cyrillic
    return re.compile(
        rf"(?<![а-яА-ЯёЁa-zA-Z0-9]){escaped}(?![а-яА-ЯёЁa-zA-Z0-9])",
        re.IGNORECASE,
    )


def _extract_context(text: str, offset: int, window: int = _CONTEXT_WINDOW) -> str:
    """Extract text fragment around a character offset."""
    start = max(0, offset - window)
    end = min(len(text), offset + window)
    return text[start:end].strip()


def _sentence_index(text: str, offset: int) -> int:
    """Determine which sentence (0-based) contains the given offset."""
    # Split on sentence-ending punctuation
    sentences = re.split(r"[.!?]\s+", text[:offset])
    return max(0, len(sentences) - 1)


def _classify_mention(context: str) -> MentionType:
    """Classify mention type from surrounding context.

    Priority order: RECOMMENDED > COMPARED > NEGATIVE > DIRECT.
    Comparison context takes precedence over negative because balanced
    pros/cons language (e.g. "преимущества и недостатки") is a comparison,
    not a negative mention.
    """
    for pat in _RECOMMEND_PATTERNS:
        if pat.search(context):
            return MentionType.RECOMMENDED

    for pat in _COMPARE_PATTERNS:
        if pat.search(context):
            return MentionType.COMPARED

    for pat in _NEGATIVE_PATTERNS:
        if pat.search(context):
            return MentionType.NEGATIVE

    return MentionType.DIRECT


def resolve_brand(
    text: str,
    brand_name: str,
    aliases: list[str] | None = None,
) -> BrandMention:
    """Resolve a single brand's presence in the text.

    Args:
        text: Cleaned response text (after preprocessing).
        brand_name: Canonical brand name.
        aliases: Alternative names / spellings / abbreviations.

    Returns:
        BrandMention with match details.
    """
    aliases = aliases or []
    all_names = [brand_name] + aliases

    mention = BrandMention(name=brand_name)
    matched_aliases: list[str] = []

    best_offset = -1

    for name in all_names:
        if not name:
            continue
        pattern = _build_pattern(name)
        match = pattern.search(text)
        if match:
            matched_aliases.append(name)
            # Track the earliest match
            if best_offset < 0 or match.start() < best_offset:
                best_offset = match.start()

    # If regex didn't find anything, try phonetic cross-script scan.
    # Extract word-like tokens from the text and check if any phonetically
    # match the brand name (handles Winline↔Винлайн etc.)
    if not matched_aliases:
        # Split text into word tokens (Latin and Cyrillic)
        tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ][a-zA-Zа-яА-ЯёЁ0-9]*", text)
        for token in tokens:
            if len(token) < 3:
                continue
            for name in all_names:
                if not name or len(name) < 3:
                    continue
                if _phonetic_match(token, name):
                    # Found a phonetic match — locate it in text for context
                    pattern = re.compile(re.escape(token), re.IGNORECASE)
                    match = pattern.search(text)
                    if match:
                        matched_aliases.append(token)
                        if best_offset < 0 or match.start() < best_offset:
                            best_offset = match.start()
                    break  # Only need one match per token

    # If regex and phonetic both failed, try _names_match() against text tokens.
    # This handles transliteration prefix matching (e.g. "Пари" → "Париматч")
    # and other fuzzy strategies that word-boundary regex can't catch.
    # Only activated for SHORT brand names (≤6 chars) where word-boundary regex
    # is most likely to fail (short names are often prefixes of longer words).
    # Longer names like "Тинькофф" (8 chars) are reliably caught by regex.
    if not matched_aliases:
        short_names = [n for n in all_names if n and 3 <= len(n) <= 6]
        if short_names:
            tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ][a-zA-Zа-яА-ЯёЁ0-9]*", text)
            for token in tokens:
                if len(token) < 3:
                    continue
                for name in short_names:
                    if _names_match(token, name):
                        pattern = re.compile(re.escape(token), re.IGNORECASE)
                        match = pattern.search(text)
                        if match:
                            matched_aliases.append(token)
                            if best_offset < 0 or match.start() < best_offset:
                                best_offset = match.start()
                        break
                if matched_aliases:
                    break

    if matched_aliases:
        mention.is_mentioned = True
        mention.aliases_matched = matched_aliases
        mention.char_offset = best_offset
        mention.sentence_index = _sentence_index(text, best_offset)
        mention.mention_context = _extract_context(text, best_offset)
        mention.mention_type = _classify_mention(mention.mention_context)
        if mention.mention_type == MentionType.RECOMMENDED:
            mention.is_recommended = True

    return mention


def _brand_in_prompt(brand_name: str, aliases: list[str] | None, prompt_text: str) -> bool:
    """Check if a brand (or any of its aliases) appears in the prompt text."""
    if not prompt_text:
        return False
    prompt_lower = prompt_text.lower()
    for name in [brand_name] + (aliases or []):
        if name and name.lower() in prompt_lower:
            return True
    return False


def resolve_all(
    text: str,
    target_brand: str,
    target_aliases: list[str] | None = None,
    competitors: dict[str, list[str]] | None = None,
    prompt_text: str = "",
) -> tuple[BrandMention, list[BrandMention]]:
    """Resolve target brand and all competitors in the text.

    Args:
        text: Cleaned response text.
        target_brand: Canonical target brand name.
        target_aliases: Aliases for the target brand.
        competitors: Dict of {competitor_name: [aliases]} for each competitor.
        prompt_text: Original prompt/query text. Brands already present in the
                     prompt are not counted as organic mentions in the response.

    Returns:
        Tuple of (target_mention, [competitor_mentions]).
    """
    # Skip target brand if it's already in the prompt
    if _brand_in_prompt(target_brand, target_aliases, prompt_text):
        target = BrandMention(name=target_brand)
    else:
        target = resolve_brand(text, target_brand, target_aliases)

    comp_mentions: list[BrandMention] = []
    for comp_name, comp_aliases in (competitors or {}).items():
        # Skip competitor if it's already in the prompt
        if _brand_in_prompt(comp_name, comp_aliases, prompt_text):
            comp_mentions.append(BrandMention(name=comp_name))
        else:
            comp = resolve_brand(text, comp_name, comp_aliases)
            comp_mentions.append(comp)

    return target, comp_mentions
