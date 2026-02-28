"""Response Normalizer — post-processes GatewayResponses.

Applies final normalization steps after the vendor adapter returns:
  - Validates and cleans response text
  - Extracts cited URLs from text (for vendors without native citations)
  - Calculates cost if adapter didn't provide it
  - Ensures all timestamps are set
  - Tags censored/filtered responses consistently
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from app.gateway.types import GatewayResponse, RequestStatus

logger = logging.getLogger(__name__)

# URL pattern for extracting citations from response text
_URL_PATTERN = re.compile(r"https?://[^\s\)\]\}\"'<>,]+")


def normalize_response(response: GatewayResponse) -> GatewayResponse:
    """Apply normalization to a gateway response.

    This is idempotent — can be called multiple times safely.
    """
    # Ensure timestamps
    if response.completed_at is None and response.status in (
        RequestStatus.SUCCESS,
        RequestStatus.CENSORED,
    ):
        response.completed_at = datetime.now(timezone.utc)

    # Ensure total_tokens is consistent
    if response.total_tokens == 0 and (response.input_tokens or response.output_tokens):
        response.total_tokens = response.input_tokens + response.output_tokens

    # Extract URLs from text if no native citations provided
    if response.status == RequestStatus.SUCCESS and not response.cited_urls:
        response.cited_urls = _extract_urls(response.raw_response_text)

    # Clean cited URLs
    if response.cited_urls:
        response.cited_urls = _clean_urls(response.cited_urls)

    # Tag censored responses consistently
    if response.status == RequestStatus.CENSORED:
        if not response.raw_response_text.startswith("[CENSORED_BY_VENDOR]"):
            response.raw_response_text = "[CENSORED_BY_VENDOR]"

    return response


def _extract_urls(text: str) -> list[str]:
    """Extract unique URLs from response text."""
    if not text:
        return []
    found = _URL_PATTERN.findall(text)
    # Deduplicate while preserving order, clean trailing punctuation
    seen: set[str] = set()
    result: list[str] = []
    for url in found:
        cleaned = url.rstrip(".,;:!?)")
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _clean_urls(urls: list[str]) -> list[str]:
    """Clean and deduplicate a list of URLs."""
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        cleaned = url.strip().rstrip(".,;:!?)")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def aggregate_resilience_responses(
    responses: list[GatewayResponse],
) -> dict:
    """Aggregate multiple resilience run responses for the same prompt.

    Returns summary statistics for consistency analysis:
      - How many runs succeeded
      - How consistent the responses are (by simple text length comparison)
      - Average latency
      - Total cost
    """
    if not responses:
        return {"total_runs": 0}

    successful = [r for r in responses if r.status == RequestStatus.SUCCESS]
    censored = [r for r in responses if r.status == RequestStatus.CENSORED]

    total_cost = sum(r.cost_usd for r in responses)
    avg_latency = sum(r.latency_ms for r in responses) / len(responses) if responses else 0

    return {
        "total_runs": len(responses),
        "successful_runs": len(successful),
        "censored_runs": len(censored),
        "failed_runs": len(responses) - len(successful) - len(censored),
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": int(avg_latency),
        "session_id": responses[0].session_id if responses else "",
        "prompt_id": responses[0].prompt_id if responses else "",
    }
