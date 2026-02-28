"""Analysis Pipeline — orchestrator for the 6-step cascading NLP pipeline.

Chains the analysis steps in order:
  1. Text Preprocessor & Sanitizer
  2. NER & Entity Resolution Engine
  3. Structural & Ranking Parser
  4. Context Evaluator (LLM-as-a-Judge, heuristic fallback)
  5. Citation & Grounding Extractor
  6. Micro-Scoring Calculator

Input:  GatewayResponse (from Module 2)
Output: AnalyzedResponse (structured metrics for BI dashboard)
"""

from __future__ import annotations

import logging

from app.analysis.types import (
    AnalyzedResponse,
    SanitizationFlag,
)
from app.analysis.preprocessor import preprocess
from app.analysis.entity_resolver import resolve_all
from app.analysis.llm_entity_extractor import (
    extract_entities_with_gemini,
    extract_entities_with_llm,
    get_entity_names,
    match_entities,
)
from app.analysis.ranking_parser import analyze_structure
from app.analysis.context_evaluator import (
    evaluate_all_with_judge,
)
from app.analysis.citation_extractor import extract_citations
from app.analysis.scoring import apply_scores
from app.gateway.types import GatewayResponse

logger = logging.getLogger(__name__)


async def analyze_response(
    response: GatewayResponse,
    target_brand: str,
    target_aliases: list[str] | None = None,
    competitors: dict[str, list[str]] | None = None,
    intent: str = "",
    persona: str = "",
    judge_api_key: str | None = None,
    prompt_text: str = "",
    rpm_limiter=None,
    tpm_limiter=None,
    gemini_api_key: str | None = None,
) -> AnalyzedResponse:
    """Run the full analysis pipeline on a single GatewayResponse.

    Args:
        response: Normalized response from Module 2's Gateway.
        target_brand: Canonical brand name to track.
        target_aliases: Alternative names for the target brand.
        competitors: Dict of {competitor_name: [aliases]}.
        intent: Intent from Module 1 metadata (informational, comparative, etc.)
        persona: Persona from Module 1 metadata.
        judge_api_key: OpenAI API key for LLM-as-a-Judge evaluation.
                       If None, falls back to OPENAI_API_KEY env var, then to heuristics.
        prompt_text: Original prompt/query text. Brands already present in the
                     prompt are not counted as organic mentions in the response.

    Returns:
        AnalyzedResponse with all analysis results.
    """
    result = AnalyzedResponse(
        request_id=response.request_id,
        session_id=response.session_id,
        vendor=response.vendor.value,
        model_version=response.model_version,
        prompt_id=response.prompt_id,
        tenant_id=response.tenant_id,
        project_id=response.project_id,
        llm_query_id=response.llm_query_id,
        intent=intent,
        persona=persona,
        latency_ms=response.latency_ms,
        cost_usd=response.cost_usd,
        total_tokens=response.total_tokens,
    )

    # Step 1: Text Preprocessor & Sanitizer
    sanitized = preprocess(
        text=response.raw_response_text,
        vendor=response.vendor.value,
    )
    result.sanitization = sanitized

    # Early exit for non-analyzable responses
    if sanitized.flag in (
        SanitizationFlag.EMPTY_RESPONSE,
        SanitizationFlag.CENSORED,
        SanitizationFlag.VENDOR_REFUSAL,
    ):
        logger.info(
            "Response %s skipped analysis: %s",
            response.request_id,
            sanitized.flag.value,
        )
        return result

    text = sanitized.text

    # Step 2a: LLM Entity Extraction (OpenAI → Gemini fallback → regex)
    # Returns enriched format: [{"name": ..., "context": ..., "type": ...}]
    llm_entities_enriched: list = []
    if judge_api_key:
        llm_entities_enriched = await extract_entities_with_llm(
            text,
            judge_api_key,
            rpm_limiter=rpm_limiter,
            tpm_limiter=tpm_limiter,
        )

    # Fallback to Gemini if OpenAI returned nothing and Gemini key is available
    # Gemini returns old format (list[str]), wrap into enriched
    if not llm_entities_enriched and gemini_api_key:
        logger.info(
            "OpenAI entity extraction returned empty, falling back to Gemini: request=%s",
            response.request_id,
        )
        gemini_names = await extract_entities_with_gemini(text, gemini_api_key)
        llm_entities_enriched = [{"name": n, "context": "", "type": "other"} for n in gemini_names]

    # Store enriched entities for Phase 3 consolidation (with context for profiling)
    result.raw_entities = llm_entities_enriched

    # Extract plain names for Phase 2 matching (backward-compatible)
    llm_entity_names = get_entity_names(llm_entities_enriched)

    # Step 2: NER & Entity Resolution
    if llm_entity_names:
        # Use LLM-extracted entities with fuzzy matching
        target_mention, comp_mentions, discovered = match_entities(
            extracted=llm_entity_names,
            target_brand=target_brand,
            target_aliases=target_aliases or [],
            competitors=competitors or {},
            full_text=text,
            prompt_text=prompt_text,
        )
        result.discovered_entities = discovered
    else:
        # Fallback to regex-based resolution
        target_mention, comp_mentions = resolve_all(
            text=text,
            target_brand=target_brand,
            target_aliases=target_aliases,
            competitors=competitors,
            prompt_text=prompt_text,
        )
    result.target_brand = target_mention
    result.competitors = comp_mentions

    # Step 3: Structural & Ranking Parser
    all_brands = [target_mention] + comp_mentions
    structure = analyze_structure(text, all_brands)
    result.structure = structure

    # Step 4: Context Evaluator — LLM-as-a-Judge (with heuristic fallback)
    await evaluate_all_with_judge(
        target=target_mention,
        competitors=comp_mentions,
        vendor=response.vendor.value,
        api_key=judge_api_key,
        rpm_limiter=rpm_limiter,
        tpm_limiter=tpm_limiter,
    )

    # Step 5: Citation & Grounding Extractor
    citations = extract_citations(
        text=text,
        native_urls=response.cited_urls,
    )
    result.citations = citations

    # Step 6: Micro-Scoring Calculator
    rvs, share = apply_scores(target_mention, comp_mentions)
    result.response_visibility_score = rvs
    result.share_of_model_local = share

    logger.info(
        "Analysis complete: request=%s, brand=%s, mentioned=%s, "
        "mention_type=%s, RVS=%.4f, share=%.2f, structure=%s, citations=%d",
        response.request_id,
        target_brand,
        target_mention.is_mentioned,
        target_mention.mention_type.value,
        rvs,
        share,
        structure.structure_type.value,
        len(citations),
    )

    return result


async def analyze_batch(
    responses: list[GatewayResponse],
    target_brand: str,
    target_aliases: list[str] | None = None,
    competitors: dict[str, list[str]] | None = None,
    intent: str = "",
    persona: str = "",
    judge_api_key: str | None = None,
) -> list[AnalyzedResponse]:
    """Run the analysis pipeline on a batch of responses.

    Useful for processing resilience runs (same prompt, multiple executions).

    Returns:
        List of AnalyzedResponse objects.
    """
    results = []
    for resp in responses:
        analyzed = await analyze_response(
            response=resp,
            target_brand=target_brand,
            target_aliases=target_aliases,
            competitors=competitors,
            intent=intent,
            persona=persona,
            judge_api_key=judge_api_key,
        )
        results.append(analyzed)

    logger.info(
        "Batch analysis complete: %d responses, brand=%s",
        len(results),
        target_brand,
    )
    return results
