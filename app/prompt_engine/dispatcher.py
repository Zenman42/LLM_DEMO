"""Layer 5: Metadata Registry & Dispatcher.

Enriches each adapted prompt with execution parameters:
  - Temperature (stochasticity control)
  - Resilience Runs (number of independent sessions)
  - Max tokens
  - Unique prompt ID

Produces FinalPrompt objects ready for API dispatch.
"""

from __future__ import annotations

import hashlib
import logging

from app.prompt_engine.types import AdaptedPrompt, FinalPrompt, Intent, TargetLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temperature mapping: Intent → Temperature
# ---------------------------------------------------------------------------

_TEMPERATURE_BY_INTENT: dict[Intent, float] = {
    Intent.INFORMATIONAL: 0.0,  # Factual, deterministic
    Intent.NAVIGATIONAL: 0.0,  # Factual, deterministic
    Intent.COMPARATIVE: 0.7,  # Creative, varied lists
    Intent.TRANSACTIONAL: 0.3,  # Semi-deterministic
}


# ---------------------------------------------------------------------------
# Resilience Runs mapping: Intent → number of independent runs
# ---------------------------------------------------------------------------

_RESILIENCE_BY_INTENT: dict[Intent, int] = {
    Intent.INFORMATIONAL: 1,  # Factual — should be stable
    Intent.NAVIGATIONAL: 1,  # Factual — should be stable
    Intent.COMPARATIVE: 3,  # Key metric — measure stability
    Intent.TRANSACTIONAL: 1,  # Usually stable
}


# ---------------------------------------------------------------------------
# Max tokens by provider
# ---------------------------------------------------------------------------

_MAX_TOKENS_BY_PROVIDER: dict[TargetLLM, int] = {
    TargetLLM.CHATGPT: 2048,
    TargetLLM.GEMINI: 2048,
    TargetLLM.DEEPSEEK: 2048,
    TargetLLM.YANDEXGPT: 2048,
}


def _generate_prompt_id(adapted: AdaptedPrompt, index: int) -> str:
    """Generate a deterministic, unique prompt ID."""
    raw = f"{adapted.target_llm.value}:{adapted.persona.value}:{adapted.intent.value}:{adapted.category}:{adapted.user_prompt[:100]}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def parameterize_prompts(
    adapted_prompts: list[AdaptedPrompt],
) -> list[FinalPrompt]:
    """Convert adapted prompts into fully parameterized FinalPrompt objects.

    Each FinalPrompt is ready for API dispatch with:
      - Unique prompt_id
      - Temperature based on intent
      - Resilience runs based on intent importance
      - Max tokens based on provider
    """
    results: list[FinalPrompt] = []

    for idx, adapted in enumerate(adapted_prompts):
        temperature = _TEMPERATURE_BY_INTENT.get(adapted.intent, 0.3)
        resilience = _RESILIENCE_BY_INTENT.get(adapted.intent, 1)
        max_tokens = _MAX_TOKENS_BY_PROVIDER.get(adapted.target_llm, 2048)
        prompt_id = _generate_prompt_id(adapted, idx)

        final = FinalPrompt(
            prompt_id=prompt_id,
            target_llm=adapted.target_llm,
            system_prompt=adapted.system_prompt,
            user_prompt=adapted.user_prompt,
            temperature=temperature,
            resilience_runs=resilience,
            max_tokens=max_tokens,
            persona=adapted.persona,
            intent=adapted.intent,
            category=adapted.category,
            brand=adapted.brand,
            competitors=adapted.competitors,
            market=adapted.market,
            skeleton_id=adapted.skeleton_id,
            pivot_axis=adapted.pivot_axis,
            measurement_type=adapted.measurement_type,
            query_class=adapted.query_class,
            query_subtype=adapted.query_subtype,
        )
        results.append(final)

    logger.info("Parameterized %d final prompts", len(results))
    return results
