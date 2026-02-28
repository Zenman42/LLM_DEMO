"""Micro-Scoring Calculator — Pipeline Step 6.

Computes:
  - Response Visibility Score (RVS):
      RVS = Presence(0/1) × Position_Weight(0.1–1.0) × Sentiment_Multiplier

  - Share of Model Local:
      share_of_model_local = 1 / total_mentioned_brands (for target brand)
      Represents the brand's share within the response's competitive landscape.
"""

from __future__ import annotations

import logging

from app.analysis.types import BrandMention

logger = logging.getLogger(__name__)


def calculate_rvs(brand: BrandMention) -> float:
    """Calculate Response Visibility Score for a brand mention.

    Formula: RVS = Presence × Position_Weight × Sentiment_Multiplier

    Where:
      - Presence = 1.0 if mentioned, 0.0 if not
      - Position_Weight = 0.1–1.0 (from ranking parser)
      - Sentiment_Multiplier = 0.5–1.2 (from context evaluator)

    Returns:
        RVS value between 0.0 and 1.2
    """
    if not brand.is_mentioned:
        return 0.0

    presence = 1.0
    position_weight = brand.position_weight
    sentiment_multiplier = brand.sentiment_multiplier

    rvs = presence * position_weight * sentiment_multiplier
    return round(rvs, 4)


def calculate_share_of_model_local(
    target: BrandMention,
    competitors: list[BrandMention],
) -> float:
    """Calculate the target brand's share among all mentioned brands.

    share_of_model_local = 1 / total_mentioned_brands

    If the target brand is not mentioned, returns 0.0.
    If only the target is mentioned, returns 1.0 (100%).

    Returns:
        Float between 0.0 and 1.0
    """
    if not target.is_mentioned:
        return 0.0

    mentioned_count = 1  # Target itself
    for comp in competitors:
        if comp.is_mentioned:
            mentioned_count += 1

    return round(1.0 / mentioned_count, 4)


def apply_scores(
    target: BrandMention,
    competitors: list[BrandMention],
) -> tuple[float, float]:
    """Calculate both RVS and share_of_model_local.

    Returns:
        Tuple of (rvs, share_of_model_local).
    """
    rvs = calculate_rvs(target)
    share = calculate_share_of_model_local(target, competitors)

    logger.debug(
        "Scoring: brand=%s, mentioned=%s, rank=%d, weight=%.2f, sentiment=%.2f, multiplier=%.2f → RVS=%.4f, share=%.4f",
        target.name,
        target.is_mentioned,
        target.position_rank,
        target.position_weight,
        target.sentiment_score,
        target.sentiment_multiplier,
        rvs,
        share,
    )

    return rvs, share
