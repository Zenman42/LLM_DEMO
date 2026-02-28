"""Prompt origin tracing — reconstruct how a query_text was generated.

Given a query_text and provider, reconstructs:
  - Which template generated it
  - What persona/intent/category it belongs to
  - What system_prompt was used (provider-specific)
  - What the final API payload looked like

This is the *inverse* of the prompt generation pipeline:
  matrix.py → expansion.py → adapters.py → dispatcher.py

Used by the Debug Console to show "how was this query formed".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.prompt_engine.matrix import _TEMPLATES_RU, _TEMPLATES_EN
from app.prompt_engine.adapters import (
    _CHATGPT_SYSTEM_PROMPTS,
)
from app.prompt_engine.types import Intent, Persona


@dataclass
class PromptOrigin:
    """Reconstructed origin of a query_text."""

    # Template matching
    matched_template: str = ""  # e.g. "Где найти честные отзывы о {brand} для {category}"
    persona: str = ""  # e.g. "skeptic"
    intent: str = ""  # e.g. "navigational"
    category: str = ""  # inferred from template match

    # Provider-specific adaptation
    system_prompt: str = ""  # actual system prompt sent to provider
    user_prompt: str = ""  # final user prompt (may differ from query_text after adaptation)

    # API payload reconstruction
    api_payload: dict = field(default_factory=dict)

    # Confidence
    match_confidence: str = "exact"  # "exact" | "fuzzy" | "none"


def _build_template_regex(template: str) -> re.Pattern:
    """Convert a template string with {placeholders} to a regex pattern."""
    # Escape regex special chars, then replace placeholders with .*
    escaped = re.escape(template)
    # Replace escaped placeholders \{brand\}, \{competitor\}, etc.
    pattern = re.sub(r"\\{[^}]+\\}", r"(.+?)", escaped)
    return re.compile(f"^{pattern}$", re.IGNORECASE)


def _match_templates(
    query_text: str,
    brand: str,
    competitors: list[str],
    market: str,
) -> tuple[str, Persona, Intent, str]:
    """Try to match query_text against known templates.

    Returns (matched_template, persona, intent, match_confidence).
    """
    templates = _TEMPLATES_RU if market == "ru" else _TEMPLATES_EN

    # First pass: exact template matching (substitute known values back)
    for (persona, intent), template_list in templates.items():
        for tmpl in template_list:
            # Try to match with brand and each competitor
            test_texts = []
            comps = competitors or [""]
            for comp in comps:
                rendered = tmpl.format(
                    brand=brand,
                    competitor=comp,
                    competitors=", ".join(competitors),
                    category="(.+)",  # will be regex
                    geo="",
                )
                test_texts.append((rendered, tmpl))

            # Also try without competitor
            rendered_no_comp = tmpl.format(
                brand=brand,
                competitor=competitors[0] if competitors else "",
                competitors=", ".join(competitors),
                category="(.+)",
                geo="",
            )
            test_texts.append((rendered_no_comp, tmpl))

            for rendered, original_tmpl in test_texts:
                try:
                    pat = re.compile(f"^{re.escape(rendered).replace(re.escape('(.+)'), '(.+?)')}$", re.IGNORECASE)
                    m = pat.match(query_text)
                    if m:
                        return original_tmpl, persona, intent, "exact"
                except re.error:
                    continue

    # Second pass: fuzzy — check if query_text contains key template fragments
    best_match = ("", None, None, 0)  # (template, persona, intent, score)
    for (persona, intent), template_list in templates.items():
        for tmpl in template_list:
            # Extract static words from template (remove placeholders)
            static_parts = re.sub(r"\{[^}]+\}", "", tmpl).split()
            static_words = [w.lower().strip(",.!?:") for w in static_parts if len(w) > 3]
            if not static_words:
                continue

            q_lower = query_text.lower()
            hits = sum(1 for w in static_words if w in q_lower)
            score = hits / len(static_words) if static_words else 0

            if score > best_match[3] and score > 0.4:
                best_match = (tmpl, persona, intent, score)

    if best_match[1] is not None:
        return best_match[0], best_match[1], best_match[2], "fuzzy"

    return "", None, None, "none"


# Provider → system prompt lookup (for collectors that use a generic system prompt)
_COLLECTOR_SYSTEM_PROMPTS = {
    "chatgpt": "You are a helpful assistant. Answer the following question thoroughly.",
    "deepseek": "You are a helpful assistant. Answer the following question thoroughly.",
    "perplexity": "You are a helpful assistant. Answer the following question thoroughly.",
    "yandexgpt": "You are a helpful assistant. Answer the following question thoroughly.",
    "gemini": "You are a helpful assistant. Answer the following question thoroughly.",
}


def _get_adapted_system_prompt(provider: str, persona: Persona | None, market: str) -> str:
    """Get the actual system prompt used by the prompt_engine adapter for this provider."""
    if provider == "chatgpt" and persona:
        prompts = _CHATGPT_SYSTEM_PROMPTS.get(persona, {})
        return prompts.get(market, _COLLECTOR_SYSTEM_PROMPTS.get(provider, ""))

    if provider == "gemini":
        if market == "ru":
            return (
                "Ты — аналитическая система. Отвечай строго на основе фактов из открытых источников. "
                "Приводи данные за текущий год. Структурируй ответ в виде объективной матрицы характеристик. "
                "Не давай субъективных оценок — только факты и данные."
            )
        return (
            "You are an analytical system. Respond strictly based on facts from public sources. "
            "Provide data from the current year. Structure your answer as an objective feature matrix. "
            "Do not provide subjective opinions — only facts and data."
        )

    if provider == "deepseek":
        if market == "ru":
            return (
                "Ты — аналитик-исследователь. Следуй чёткому алгоритму ответа. "
                "Будь лаконичен и структурирован. Используй нумерованные списки."
            )
        return (
            "You are a research analyst. Follow a clear response algorithm. "
            "Be concise and structured. Use numbered lists."
        )

    if provider == "yandexgpt":
        if market == "ru":
            return "Отвечай подробно и по существу."
        return "Answer thoroughly and to the point."

    if provider == "perplexity":
        if market == "ru":
            return (
                "Ты — поисковый ассистент с доступом к актуальной информации из интернета. "
                "Отвечай на основе свежих данных. Приводи ссылки на источники. "
                "Структурируй ответ чётко."
            )
        return (
            "You are a search assistant with access to current internet information. "
            "Respond based on fresh data. Provide source links. "
            "Structure your answer clearly."
        )

    # Fallback: collector-level generic prompt
    return _COLLECTOR_SYSTEM_PROMPTS.get(provider, "")


def _reconstruct_api_payload(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dict:
    """Reconstruct the actual JSON payload sent to the vendor API."""
    if provider == "yandexgpt":
        return {
            "modelUri": f"gpt://<folder_id>/{model}",
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt},
            ],
        }

    if provider == "gemini":
        payload: dict = {
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    # OpenAI-compatible: chatgpt, deepseek, perplexity
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def trace_prompt_origin(
    query_text: str,
    provider: str,
    model: str,
    target_brand: str = "",
    competitors: list[str] | None = None,
    market: str = "ru",
) -> PromptOrigin:
    """Reconstruct the full origin of a query_text.

    Attempts to:
      1. Match against template matrix to find persona/intent
      2. Determine the provider-specific system prompt
      3. Reconstruct the API payload as it was sent
    """
    competitors = competitors or []

    # Step 1: Match template
    matched_template, persona, intent, confidence = _match_templates(query_text, target_brand, competitors, market)

    # Step 2: Get system prompt (prefer adapted version if persona known)
    system_prompt = _get_adapted_system_prompt(provider, persona, market)

    # Step 3: Determine final user prompt (may differ from query_text for some adapters)
    user_prompt = query_text

    # Step 4: Reconstruct API payload
    api_payload = _reconstruct_api_payload(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return PromptOrigin(
        matched_template=matched_template,
        persona=persona.value if persona else "",
        intent=intent.value if intent else "",
        category="",  # Hard to extract reliably from rendered text
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_payload=api_payload,
        match_confidence=confidence,
    )
