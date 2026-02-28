"""Context Evaluator / LLM-as-a-Judge — Pipeline Step 4.

Uses a cheap LLM (GPT-4o-mini) to evaluate:
  - mention_type (direct / recommended / compared / negative)
  - sentiment_score (-1.0 to +1.0)
  - sentiment_label (positive / neutral / negative / mixed)
  - is_recommended (bool)
  - context_tags (e.g. ["price", "quality", "reliability"])
  - is_hallucination (bool)

Falls back to rule-based heuristics when LLM call is unavailable
or for cost optimization in batch mode.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

from app.analysis.types import BrandMention, MentionType, SentimentLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM-as-a-Judge prompt template
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are a brand mention evaluator for AI visibility monitoring.

You receive:
- The brand name being tracked
- A fragment of an AI-generated response containing the brand mention
- The list of competitor names (if any)
- The name of the AI vendor that generated the response

Your task: classify the mention and evaluate sentiment.

Output ONLY valid JSON (no markdown, no explanation) with these fields:

- mention_type: one of "recommended", "compared", "negative", "direct"
  - "recommended" → the brand is explicitly recommended, called "best", "top pick", "#1 choice", etc.
  - "compared" → the brand is compared with competitors, pros/cons are listed, "vs" language is used,\
 or the response contrasts the brand against alternatives — even if one side has more advantages.\
 A balanced discussion of advantages AND disadvantages counts as "compared", NOT "negative".
  - "negative" → the brand is mentioned ONLY in a clearly negative way (warned against, called bad, etc.)\
 with NO positive counterbalance or comparison context
  - "direct" → the brand is simply mentioned without strong sentiment or comparison

- sentiment_score: float from -1.0 (very negative) to 1.0 (very positive), 0.0 is neutral
- sentiment_label: one of "positive", "neutral", "negative", "mixed"
- is_recommended: boolean, true ONLY if the brand is explicitly recommended as a top choice
- context_tags: list of strings describing topics discussed (e.g. ["price", "quality", "support", "booking", "hotels"])
- is_hallucination: boolean, true if the claims about the brand appear factually incorrect or fabricated

Important rules:
- If the text discusses both advantages AND disadvantages (pros/cons, плюсы/минусы) → mention_type is "compared", NOT "negative"
- If competitors are mentioned alongside the brand → mention_type is likely "compared"
- A structured review with "Плюсы" and "Минусы" sections is a balanced comparison → "compared"
- "negative" should ONLY be used when the text is overwhelmingly negative with NO positive aspects mentioned at all
- sentiment_label "mixed" means BOTH positive and negative aspects are discussed
- Be conservative with is_hallucination — only flag clearly false statements"""

_JUDGE_USER_TEMPLATE = """\
Brand: {brand}
Competitors: {competitors}
AI vendor: {vendor}

Context fragment (around the brand mention):
\"\"\"{context}\"\"\"\

Evaluate this brand mention. Return ONLY JSON."""

# Judge model configuration
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_API_URL = "https://api.openai.com/v1/chat/completions"
JUDGE_MAX_TOKENS = 512
JUDGE_TEMPERATURE = 0.0
JUDGE_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Rule-based fallback heuristics
# ---------------------------------------------------------------------------

_POSITIVE_WORDS_RU = {
    "лучший",
    "отличный",
    "надёжный",
    "надежный",
    "удобный",
    "быстрый",
    "рекомендую",
    "популярный",
    "качественный",
    "безопасный",
    "выгодный",
    "топ",
    "лидер",
    "превосходный",
    "идеальный",
}

_NEGATIVE_WORDS_RU = {
    "худший",
    "плохой",
    "медленный",
    "дорогой",
    "ненадёжный",
    "ненадежный",
    "проблема",
    "проблемы",
    "минус",
    "минусы",
    "недостаток",
    "недостатки",
    "опасный",
    "рискованный",
    "устаревший",
    "неудобный",
    "сложный",
}

_POSITIVE_WORDS_EN = {
    "best",
    "excellent",
    "reliable",
    "convenient",
    "fast",
    "recommend",
    "recommended",
    "popular",
    "quality",
    "safe",
    "leader",
    "top",
    "superior",
    "ideal",
    "great",
    "outstanding",
    "trusted",
    "leading",
}

_NEGATIVE_WORDS_EN = {
    "worst",
    "bad",
    "slow",
    "expensive",
    "unreliable",
    "problem",
    "problems",
    "disadvantage",
    "disadvantages",
    "drawback",
    "drawbacks",
    "dangerous",
    "risky",
    "outdated",
    "complex",
    "difficult",
    "poor",
    "issue",
    "issues",
    "worse",
}

_CONTEXT_TAG_PATTERNS = {
    "price": re.compile(r"(?:цен[аы]|стоимость|тариф|price|cost|pricing|cheap|expensive|fee)", re.IGNORECASE),
    "quality": re.compile(r"(?:качеств|quality|reliable|надёжн|надежн)", re.IGNORECASE),
    "support": re.compile(r"(?:поддержк|support|customer service|помощь)", re.IGNORECASE),
    "speed": re.compile(r"(?:скорость|быстр|speed|fast|performance|производительн)", re.IGNORECASE),
    "features": re.compile(r"(?:функци|feature|возможност|capability|инструмент|tool)", re.IGNORECASE),
    "security": re.compile(r"(?:безопасн|security|защит|encryption|шифрован)", re.IGNORECASE),
    "ease_of_use": re.compile(r"(?:удобн|простот|ease|user.friendly|интуитивн|intuitive|usability)", re.IGNORECASE),
    "integration": re.compile(r"(?:интеграци|integration|api|подключен|connect)", re.IGNORECASE),
}

# Valid mention_type values from the judge
_VALID_MENTION_TYPES = {"recommended", "compared", "negative", "direct"}


def _score_from_words(context: str) -> float:
    """Calculate a simple sentiment score from keyword matching."""
    words = set(re.findall(r"[а-яА-ЯёЁa-zA-Z]+", context.lower()))

    positive_hits = len(words & (_POSITIVE_WORDS_RU | _POSITIVE_WORDS_EN))
    negative_hits = len(words & (_NEGATIVE_WORDS_RU | _NEGATIVE_WORDS_EN))

    total = positive_hits + negative_hits
    if total == 0:
        return 0.0

    # Scale from -1 to 1
    raw_score = (positive_hits - negative_hits) / total
    return round(max(-1.0, min(1.0, raw_score)), 2)


def _label_from_score(score: float) -> SentimentLabel:
    """Convert numeric score to sentiment label."""
    if score > 0.25:
        return SentimentLabel.POSITIVE
    if score < -0.25:
        return SentimentLabel.NEGATIVE
    if abs(score) <= 0.1:
        return SentimentLabel.NEUTRAL
    return SentimentLabel.MIXED


def _extract_context_tags(context: str) -> list[str]:
    """Extract context tags from the mention fragment."""
    tags = []
    for tag, pattern in _CONTEXT_TAG_PATTERNS.items():
        if pattern.search(context):
            tags.append(tag)
    return tags


def _apply_sentiment_multiplier(brand: BrandMention) -> None:
    """Set sentiment_multiplier based on sentiment_label."""
    if brand.sentiment_label == SentimentLabel.POSITIVE:
        brand.sentiment_multiplier = 1.2
    elif brand.sentiment_label == SentimentLabel.NEGATIVE:
        brand.sentiment_multiplier = 0.5
    elif brand.sentiment_label == SentimentLabel.MIXED:
        brand.sentiment_multiplier = 0.8
    else:
        brand.sentiment_multiplier = 1.0


def _apply_mention_type_from_judge(brand: BrandMention, mention_type_str: str) -> None:
    """Apply mention_type from judge output, overriding the regex-based classification."""
    mention_type_str = mention_type_str.lower().strip()
    if mention_type_str not in _VALID_MENTION_TYPES:
        # Don't override if judge returned garbage
        logger.warning("Judge returned invalid mention_type: %s — keeping original", mention_type_str)
        return

    try:
        brand.mention_type = MentionType(mention_type_str)
    except ValueError:
        logger.warning("Cannot map judge mention_type '%s' to MentionType enum", mention_type_str)


def evaluate_brand_heuristic(brand: BrandMention) -> BrandMention:
    """Evaluate a brand mention using rule-based heuristics (no LLM call).

    Modifies the BrandMention in-place and returns it.
    """
    if not brand.is_mentioned or not brand.mention_context:
        return brand

    context = brand.mention_context

    # Sentiment scoring
    brand.sentiment_score = _score_from_words(context)
    brand.sentiment_label = _label_from_score(brand.sentiment_score)

    # Context tags
    brand.context_tags = _extract_context_tags(context)

    # Sentiment multiplier for RVS
    _apply_sentiment_multiplier(brand)

    # Mark evaluation method
    brand.judge_method = "heuristic"

    return brand


def parse_judge_response(raw_json: str, brand: BrandMention) -> BrandMention:
    """Parse the LLM judge response and update the BrandMention.

    Args:
        raw_json: Raw JSON string from the judge LLM.
        brand: BrandMention to update.

    Returns:
        Updated BrandMention.
    """
    try:
        # Try to extract JSON from markdown code blocks if present
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_json, re.DOTALL)
        if json_match:
            raw_json = json_match.group(1)

        data = json.loads(raw_json)

        # mention_type (NEW — judge now classifies this)
        if "mention_type" in data:
            _apply_mention_type_from_judge(brand, data["mention_type"])

        # Sentiment
        brand.sentiment_score = float(data.get("sentiment_score", 0.0))
        brand.sentiment_score = max(-1.0, min(1.0, brand.sentiment_score))

        label_str = data.get("sentiment_label", "neutral")
        try:
            brand.sentiment_label = SentimentLabel(label_str)
        except ValueError:
            brand.sentiment_label = _label_from_score(brand.sentiment_score)

        brand.is_recommended = bool(data.get("is_recommended", False))
        brand.is_hallucination = bool(data.get("is_hallucination", False))

        # If judge says recommended, align mention_type
        if brand.is_recommended and brand.mention_type != MentionType.RECOMMENDED:
            brand.mention_type = MentionType.RECOMMENDED

        tags = data.get("context_tags", [])
        if isinstance(tags, list):
            brand.context_tags = [str(t) for t in tags]

        _apply_sentiment_multiplier(brand)

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to parse judge response: %s — falling back to heuristic", e)
        evaluate_brand_heuristic(brand)

    return brand


def build_judge_prompt(
    brand: BrandMention,
    vendor: str = "",
    competitors: list[str] | None = None,
) -> tuple[str, str]:
    """Build the system + user prompts for the LLM judge.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    comp_str = ", ".join(competitors) if competitors else "none specified"
    user_prompt = _JUDGE_USER_TEMPLATE.format(
        brand=brand.name,
        context=brand.mention_context[:1000],  # Allow longer context for better judgement
        vendor=vendor,
        competitors=comp_str,
    )
    return _JUDGE_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Async LLM Judge call
# ---------------------------------------------------------------------------


async def _call_judge_llm(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    rpm_limiter=None,
    tpm_limiter=None,
) -> str | None:
    """Call GPT-4o-mini as a judge. Returns raw response text or None on failure."""
    if rpm_limiter is not None:
        await rpm_limiter.acquire()
    if tpm_limiter is not None:
        # Estimate: ~800 input tokens (system+user prompt) + ~300 output = ~1100
        await tpm_limiter.acquire(tokens=1200)

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
    }

    try:
        async with httpx.AsyncClient(timeout=JUDGE_TIMEOUT) as client:
            resp = await client.post(
                JUDGE_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"] or None

    except httpx.TimeoutException:
        logger.warning("Judge LLM call timed out after %ds", JUDGE_TIMEOUT)
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("Judge LLM HTTP error %s: %s", e.response.status_code, e)
        return None
    except Exception as e:
        logger.warning("Judge LLM unexpected error: %s", e)
        return None


async def evaluate_brand_with_judge(
    brand: BrandMention,
    vendor: str = "",
    competitors: list[str] | None = None,
    api_key: str | None = None,
    rpm_limiter=None,
    tpm_limiter=None,
) -> BrandMention:
    """Evaluate a single brand mention using LLM-as-a-Judge.

    Falls back to heuristic if:
      - brand not mentioned or no context
      - no API key available
      - LLM call fails

    Modifies the BrandMention in-place and returns it.
    """
    if not brand.is_mentioned or not brand.mention_context:
        return brand

    # Resolve API key: explicit > env var
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        logger.debug("No OpenAI API key for judge — falling back to heuristic")
        return evaluate_brand_heuristic(brand)

    system_prompt, user_prompt = build_judge_prompt(brand, vendor, competitors)

    # Store prompts for debug transparency (before the call — they're useful even if call fails)
    brand.judge_prompt_system = system_prompt
    brand.judge_prompt_user = user_prompt

    raw_response = await _call_judge_llm(
        system_prompt,
        user_prompt,
        key,
        rpm_limiter=rpm_limiter,
        tpm_limiter=tpm_limiter,
    )

    if raw_response is None:
        logger.info("Judge call failed for brand=%s — falling back to heuristic", brand.name)
        return evaluate_brand_heuristic(brand)

    # Store judge trace info
    brand.judge_method = "llm_judge"
    brand.judge_raw_response = raw_response

    logger.debug("Judge response for %s: %s", brand.name, raw_response[:200])
    return parse_judge_response(raw_response, brand)


async def evaluate_all_with_judge(
    target: BrandMention,
    competitors: list[BrandMention],
    vendor: str = "",
    api_key: str | None = None,
    rpm_limiter=None,
    tpm_limiter=None,
) -> tuple[BrandMention, list[BrandMention]]:
    """Evaluate target brand and all competitors using LLM-as-a-Judge.

    Only evaluates mentioned brands (skips not-mentioned ones).
    Falls back to heuristic per-brand if judge call fails.

    Returns:
        (target, competitors) with sentiment fields populated.
    """
    competitor_names = [c.name for c in competitors]

    # Evaluate target brand
    await evaluate_brand_with_judge(
        target,
        vendor,
        competitor_names,
        api_key,
        rpm_limiter=rpm_limiter,
        tpm_limiter=tpm_limiter,
    )

    # Evaluate mentioned competitors (no need to call judge for not-mentioned)
    for comp in competitors:
        if comp.is_mentioned:
            await evaluate_brand_with_judge(
                comp,
                vendor,
                competitor_names,
                api_key,
                rpm_limiter=rpm_limiter,
                tpm_limiter=tpm_limiter,
            )

    return target, competitors


def evaluate_all_heuristic(
    target: BrandMention,
    competitors: list[BrandMention],
) -> tuple[BrandMention, list[BrandMention]]:
    """Evaluate all brands using heuristic fallback.

    Returns:
        (target, competitors) with sentiment fields populated.
    """
    evaluate_brand_heuristic(target)
    for comp in competitors:
        evaluate_brand_heuristic(comp)
    return target, competitors
