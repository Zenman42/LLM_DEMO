"""Text Preprocessor & Sanitizer — Pipeline Step 1.

Cleans raw LLM responses before analysis:
  - Strips DeepSeek <think>...</think> reasoning blocks
  - Detects vendor guardrail / safety refusals
  - Cleans ChatGPT-style Markdown artifacts
  - Handles empty / malformed responses
"""

from __future__ import annotations

import re
import logging

from app.analysis.types import SanitizationFlag, SanitizedText

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek <think> tag pattern
# ---------------------------------------------------------------------------
_THINK_PATTERN = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Vendor refusal / guardrail patterns
# ---------------------------------------------------------------------------
_REFUSAL_PATTERNS_RU = [
    re.compile(r"я\s+(?:ии|искусственный интеллект|языковая модель)", re.IGNORECASE),
    re.compile(r"не\s+(?:могу|даю|предоставляю)\s+(?:финансовых?|медицинских?|юридических?)", re.IGNORECASE),
    re.compile(r"обратитесь\s+к\s+(?:специалисту|врачу|юристу)", re.IGNORECASE),
    re.compile(r"я\s+не\s+(?:могу|имею права)\s+(?:рекомендовать|советовать)", re.IGNORECASE),
    re.compile(
        r"не\s+является\s+(?:финансовой|медицинской|юридической)\s+(?:консультацией|рекомендацией)", re.IGNORECASE
    ),
]

_REFUSAL_PATTERNS_EN = [
    re.compile(r"i(?:'m|\s+am)\s+(?:an?\s+)?(?:ai|artificial intelligence|language model)", re.IGNORECASE),
    re.compile(r"(?:can(?:'t|not)|unable to)\s+(?:provide|give|offer)\s+(?:financial|medical|legal)", re.IGNORECASE),
    re.compile(r"consult\s+(?:a|your)\s+(?:professional|doctor|lawyer|advisor)", re.IGNORECASE),
    re.compile(r"not\s+(?:financial|medical|legal)\s+advice", re.IGNORECASE),
    re.compile(r"i\s+(?:can(?:'t|not)|am unable to)\s+(?:recommend|advise)", re.IGNORECASE),
]

# Combined
_REFUSAL_PATTERNS = _REFUSAL_PATTERNS_RU + _REFUSAL_PATTERNS_EN

# Vendor-specific censorship markers (e.g. from Gemini SAFETY filter)
_CENSORED_MARKERS = [
    "[CENSORED_BY_VENDOR]",
    "[CENSORED]",
    "[BLOCKED]",
]

# ---------------------------------------------------------------------------
# Markdown cleanup patterns
# ---------------------------------------------------------------------------
# Bold/italic markers that remain as artifacts
_MD_BOLD_ITALIC = re.compile(r"\*{1,3}(.*?)\*{1,3}")
# Excessive blank lines
_MD_BLANK_LINES = re.compile(r"\n{3,}")
# Leading/trailing whitespace per line
_MD_LINE_WHITESPACE = re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE)


def preprocess(
    text: str,
    vendor: str = "",
) -> SanitizedText:
    """Run text through the sanitization pipeline.

    Args:
        text: Raw response text from a GatewayResponse.
        vendor: Vendor name (e.g. "deepseek", "gemini") for vendor-specific rules.

    Returns:
        SanitizedText with cleaned text and metadata about what was stripped.
    """
    original_text = text

    # Empty check
    if not text or not text.strip():
        return SanitizedText(
            text="",
            original_text=original_text,
            flag=SanitizationFlag.EMPTY_RESPONSE,
        )

    # Check for censorship markers
    for marker in _CENSORED_MARKERS:
        if marker in text:
            return SanitizedText(
                text="",
                original_text=original_text,
                flag=SanitizationFlag.CENSORED,
            )

    # Strip DeepSeek <think> blocks
    think_content = ""
    if vendor.lower() in ("deepseek", "") or "<think>" in text.lower():
        think_matches = _THINK_PATTERN.findall(text)
        if think_matches:
            think_content = "\n".join(think_matches)
            text = _THINK_PATTERN.sub("", text).strip()
            if not text:
                return SanitizedText(
                    text="",
                    original_text=original_text,
                    flag=SanitizationFlag.EMPTY_RESPONSE,
                    think_content=think_content,
                    stripped_chars=len(original_text),
                )

    # Detect refusal patterns
    refusal_count = 0
    for pattern in _REFUSAL_PATTERNS:
        if pattern.search(text):
            refusal_count += 1

    # If multiple refusal signals or short disclaimer-only response, mark as refusal.
    # A genuine refusal is typically short, lacks numbered/bulleted lists, and
    # consists mostly of disclaimers without real informational content.
    has_list_content = bool(
        re.search(r"^\s*\d+[.)]\s+", text, re.MULTILINE) or re.search(r"^\s*[-*•]\s+", text, re.MULTILINE)
    )
    is_refusal = refusal_count >= 2 or (refusal_count >= 1 and len(text) < 200 and not has_list_content)
    if is_refusal:
        return SanitizedText(
            text=text,
            original_text=original_text,
            flag=SanitizationFlag.VENDOR_REFUSAL,
            think_content=think_content,
            stripped_chars=len(original_text) - len(text),
        )

    # Markdown cleanup
    cleaned = text
    # Remove bold/italic markers but keep inner text
    cleaned = _MD_BOLD_ITALIC.sub(r"\1", cleaned)
    # Collapse excessive blank lines
    cleaned = _MD_BLANK_LINES.sub("\n\n", cleaned)
    # Strip per-line whitespace
    cleaned = _MD_LINE_WHITESPACE.sub("", cleaned)
    # Final trim
    cleaned = cleaned.strip()

    stripped_chars = len(original_text) - len(cleaned)

    # Determine flag
    flag = SanitizationFlag.CLEAN
    if think_content:
        flag = SanitizationFlag.DEEPSEEK_THINK_STRIPPED

    return SanitizedText(
        text=cleaned,
        original_text=original_text,
        flag=flag,
        think_content=think_content,
        stripped_chars=stripped_chars,
    )
