"""Base LLM collector with response analysis.

Uses Module 3's analysis pipeline for deep NLP analysis when available,
with a fallback to inline regex-based heuristics.

Collection uses a two-phase approach to avoid OpenAI rate-limit contention:
  Phase 1: Query all LLM providers (only provider API calls)
  Phase 2: Analyze responses via OpenAI judge (only OpenAI API calls)
"""

import asyncio
import logging
import re
import time
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.base import BaseCollector, CollectionResult
from app.models.discovered_entity import DiscoveredEntity
from app.models.llm_query import LlmQuery
from app.models.llm_snapshot import LlmSnapshot

logger = logging.getLogger(__name__)

# Try to import Module 3 pipeline — graceful fallback if unavailable
try:
    from app.analysis.pipeline import analyze_response as pipeline_analyze
    from app.gateway.types import GatewayResponse, GatewayVendor

    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    logger.info("Module 3 analysis pipeline not available; using inline heuristics")


# Map provider name → GatewayVendor enum (used when pipeline is available)
_VENDOR_MAP: dict[str, str] = {
    "chatgpt": "chatgpt",
    "deepseek": "deepseek",
    "perplexity": "perplexity",
    "yandexgpt": "yandexgpt",
    "gemini": "gemini",
    "gigachat": "gigachat",
}


@dataclass
class AnalysisResult:
    """Result of analyzing a single LLM response."""

    brand_mentioned: bool = False
    mention_type: str = "none"  # direct | recommended | compared | negative | none
    mention_context: str = ""
    competitor_mentions: dict[str, bool] = field(default_factory=dict)
    cited_urls: list[str] = field(default_factory=list)
    # GEO metrics (populated by Module 3 pipeline, None for fallback heuristics)
    sentiment_label: str | None = None  # positive | neutral | negative | mixed
    position_rank: int | None = None  # 0=not in list, 1+=position
    structure_type: str | None = None  # numbered_list | bulleted_list | narrative | table | mixed
    is_hallucination: bool | None = None
    discovered_entities: list[str] = field(default_factory=list)  # brands found but not matched
    raw_extracted_entities: list[str] = field(
        default_factory=list
    )  # all entities from LLM extraction (before matching)


@dataclass
class LlmResponse:
    """Raw response from an LLM API."""

    text: str
    model: str
    tokens: int = 0
    cost_usd: float = 0.0
    cited_urls: list[str] = field(default_factory=list)  # some APIs return citations natively


# ---------------------------------------------------------------------------
# RPM-aware rate limiter (token bucket)
# ---------------------------------------------------------------------------


class RpmLimiter:
    """Token-bucket rate limiter that enforces requests-per-minute.

    Allows bursts up to *burst* tokens, refills at *rpm* tokens per minute.
    Each ``acquire()`` consumes one token, sleeping when the bucket is empty.
    """

    def __init__(self, rpm: int, burst: int | None = None):
        self.rpm = rpm
        self.burst = burst or max(rpm // 4, 1)
        self._tokens = float(self.burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            # Refill tokens based on elapsed time
            self._tokens = min(
                self.burst,
                self._tokens + elapsed * (self.rpm / 60.0),
            )
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # Not enough tokens — calculate wait time
            deficit = 1.0 - self._tokens
            wait = deficit / (self.rpm / 60.0)

        # Sleep outside the lock so other coroutines can refill
        logger.debug("RpmLimiter: waiting %.2fs (rpm=%d)", wait, self.rpm)
        await asyncio.sleep(wait)

        # Consume after waiting
        async with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0)
            self._last_refill = time.monotonic()


class TpmLimiter:
    """Token-bucket rate limiter that enforces tokens-per-minute.

    Unlike RpmLimiter (1 token per request), this consumes a variable number
    of tokens per call, matching the estimated token usage of each OpenAI request.
    """

    def __init__(self, tpm: int, burst: int | None = None):
        self.tpm = tpm
        self.burst = burst or max(tpm // 4, 1000)
        self._tokens = float(self.burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1000) -> None:
        """Wait until *tokens* are available, then consume them."""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self.burst,
                    self._tokens + elapsed * (self.tpm / 60.0),
                )
                self._last_refill = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Not enough tokens — calculate wait time
                deficit = tokens - self._tokens
                wait = deficit / (self.tpm / 60.0)

            logger.debug("TpmLimiter: waiting %.2fs for %d tokens (tpm=%d)", wait, tokens, self.tpm)
            await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Concurrency & RPM settings per provider
# ---------------------------------------------------------------------------

# Max parallel connections per provider (semaphore size)
_COLLECTOR_CONCURRENCY: dict[str, int] = {
    "chatgpt": 5,  # Bottleneck is TPM not RPM — semaphore just prevents runaway
    "gemini": 2,  # Free tier: ~15 RPM
    "perplexity": 2,  # Tight rate limits on free/low tiers
    "deepseek": 5,  # Reasonable limits
    "yandexgpt": 3,  # ~10 RPM
    "gigachat": 1,  # Very tight rate limits — sequential to avoid 429s
}

# Requests per minute per provider (for RPM limiter)
_COLLECTOR_RPM: dict[str, int] = {
    "chatgpt": 60,  # Actual limit 500 RPM — safe to go higher, TPM is the real limiter
    "gemini": 14,  # Free tier ~15 RPM
    "perplexity": 15,  # Conservative estimate
    "deepseek": 25,  # Moderate limits
    "yandexgpt": 8,  # ~10 RPM
    "gigachat": 3,  # Very tight rate limits on personal tier
}

# RPM for OpenAI judge/entity-extraction calls (shared across all providers)
# 500 RPM total, minus ChatGPT provider queries → plenty of headroom
_JUDGE_RPM = 100
_JUDGE_CONCURRENCY = 10

# TPM (tokens per minute) for ALL OpenAI calls combined:
# ChatGPT queries + entity extraction + judge — all share the same quota.
# gpt-4o-mini limit: 200,000 TPM.
# Leave ~25% headroom → 150K effective.
_OPENAI_TPM = 150_000


# Patterns for mention type classification (fallback heuristics)
_RECOMMEND_PATTERNS = re.compile(
    r"\b(recommend|suggest|best choice|top pick|ideal|excellent|great option|highly rated|go.to)\b",
    re.IGNORECASE,
)
_NEGATIVE_PATTERNS = re.compile(
    r"\b(avoid|issues? with|problems? with|drawback|downside|not recommend|worse|inferior|lacking)\b",
    re.IGNORECASE,
)
_COMPARE_PATTERNS = re.compile(
    r"\b(compar(?:e[ds]?|ing|ison)|vs\.?|versus|alternatively|instead of|better than|worse than|similar to|competitor)\b",
    re.IGNORECASE,
)


class BaseLlmCollector(BaseCollector):
    """Base class for all LLM collectors."""

    provider: str = "unknown"

    def __init__(self, api_key: str, tenant_id: uuid.UUID):
        self.api_key = api_key
        self.tenant_id = tenant_id

    @abstractmethod
    async def query_llm(self, prompt: str) -> LlmResponse:
        """Send a prompt to the LLM and return the raw response. Implemented by subclasses."""
        ...

    async def collect(
        self,
        *,
        db: AsyncSession,
        project_id: int,
        queries: list[LlmQuery],
        brand_name: str,
        competitors: list[str] | None = None,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
        concurrency_ctx: Any | None = None,
    ) -> CollectionResult:
        """Run all queries against this LLM provider and save results.

        When *concurrency_ctx* is provided, queries are executed in parallel
        using a two-phase approach:
          Phase 1: Collect all LLM responses (provider-rate-limited)
          Phase 2: Analyze via OpenAI judge (OpenAI-rate-limited)

        Otherwise, falls back to sequential processing (backward-compatible).
        """
        result = CollectionResult()

        if not queries:
            result.skipped = True
            result.skip_reason = "No LLM queries"
            return result

        today = date.today()
        competitors = competitors or []

        if concurrency_ctx is not None:
            # --- Two-phase parallel path ---
            return await self._collect_parallel(
                db=db,
                project_id=project_id,
                queries=queries,
                brand_name=brand_name,
                competitors=competitors,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                concurrency_ctx=concurrency_ctx,
                today=today,
            )

        # --- Sequential path (backward-compatible) ---
        for query in queries:
            try:
                llm_resp = await self.query_llm(query.query_text)

                # Merge query-level competitors with project-level
                all_competitors = list(set(competitors + (query.competitors or [])))

                # Merge native citations from API with extracted ones
                all_cited = llm_resp.cited_urls.copy()

                target_brand = query.target_brand or brand_name

                # Use Module 3 pipeline if available, otherwise fallback to inline heuristics
                analysis = await self._analyze_with_pipeline(
                    llm_resp=llm_resp,
                    query=query,
                    target_brand=target_brand,
                    competitors=all_competitors,
                    project_id=project_id,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                )

                # Combine cited URLs: native from API + extracted from text
                combined_urls = list(dict.fromkeys(all_cited + analysis.cited_urls))

                await self._save_snapshot(
                    db=db,
                    llm_query=query,
                    today=today,
                    llm_model=llm_resp.model,
                    analysis=analysis,
                    raw_response=llm_resp.text,
                    response_tokens=llm_resp.tokens,
                    cost_usd=llm_resp.cost_usd,
                    cited_urls=combined_urls,
                )
                await self._upsert_discovered_entities(
                    db=db,
                    project_id=project_id,
                    discovered_entities=analysis.discovered_entities,
                    today=today,
                    competitors=all_competitors,
                    target_brand=target_brand,
                )
                result.collected += 1

            except Exception as e:
                logger.error(
                    "%s error for query %d (%s): %s",
                    self.provider,
                    query.id,
                    query.query_text[:50],
                    e,
                )
                result.errors.append(f"{self.provider}/query_{query.id}: {e}")

        return result

    async def _collect_parallel(
        self,
        *,
        db: AsyncSession,
        project_id: int,
        queries: list[LlmQuery],
        brand_name: str,
        competitors: list[str],
        openai_api_key: str | None,
        gemini_api_key: str | None,
        concurrency_ctx: Any,
        today: date,
    ) -> CollectionResult:
        """Two-phase parallel collection.

        Phase 1 — Query: Call the LLM provider for all queries (rate-limited
        by provider semaphore + RPM limiter).  No OpenAI calls here.

        Phase 2 — Analyze: Run OpenAI judge/entity-extraction on collected
        responses (rate-limited by a separate judge semaphore + RPM limiter).

        This separation prevents ChatGPT query calls and judge calls from
        competing for the same OpenAI RPM quota.
        """
        import random

        result = CollectionResult()

        # --- Provider-level concurrency & RPM ---
        max_concurrent = _COLLECTOR_CONCURRENCY.get(self.provider, 5)
        provider_rpm = _COLLECTOR_RPM.get(self.provider, 20)
        provider_semaphore = asyncio.Semaphore(max_concurrent)
        provider_rpm_limiter = RpmLimiter(rpm=provider_rpm, burst=max(max_concurrent, 3))

        # --- Judge-level concurrency & RPM (shared via concurrency_ctx) ---
        judge_rpm_limiter = getattr(concurrency_ctx, "judge_rpm_limiter", None)
        if judge_rpm_limiter is None:
            judge_rpm_limiter = RpmLimiter(rpm=_JUDGE_RPM, burst=_JUDGE_CONCURRENCY)
        judge_semaphore = getattr(concurrency_ctx, "openai_semaphore", None)
        if judge_semaphore is None:
            judge_semaphore = asyncio.Semaphore(_JUDGE_CONCURRENCY)

        # --- TPM limiter (shared across ALL OpenAI calls) ---
        openai_tpm_limiter = getattr(concurrency_ctx, "openai_tpm_limiter", None)

        db_lock = asyncio.Lock()

        logger.info(
            "%s: two-phase collection starting — %d queries, max_concurrent=%d, provider_rpm=%d",
            self.provider,
            len(queries),
            max_concurrent,
            provider_rpm,
        )

        # ==================================================================
        # Phase 1: Query LLM provider (no OpenAI judge calls)
        # ==================================================================

        @dataclass
        class _QueryResult:
            query: LlmQuery
            llm_resp: LlmResponse | None = None
            error: str | None = None

        # ChatGPT queries go to the same OpenAI API as judge/entity calls,
        # so they must share the TPM budget.
        _is_openai_provider = self.provider == "chatgpt"

        async def _fetch_one(query: LlmQuery) -> _QueryResult:
            """Fetch a single LLM response with RPM + concurrency limiting."""
            for attempt in range(6):  # 1 initial + 5 retries
                try:
                    await provider_rpm_limiter.acquire()
                    # ChatGPT provider shares TPM with judge/entity extraction
                    if _is_openai_provider and openai_tpm_limiter is not None:
                        await openai_tpm_limiter.acquire(tokens=2000)
                    async with provider_semaphore:
                        resp = await self.query_llm(query.query_text)
                    return _QueryResult(query=query, llm_resp=resp)
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str
                    is_retryable = is_rate_limit or any(code in err_str for code in ("502", "503", "529"))
                    if is_retryable and attempt < 5:
                        if is_rate_limit:
                            # 429: longer backoff — API needs time to recover
                            base_delay = min(30 * (attempt + 1), 120)
                        else:
                            base_delay = min(2 ** (attempt + 1), 60)
                        delay = random.uniform(base_delay * 0.7, base_delay * 1.3)
                        logger.warning(
                            "%s: retryable error for query %d (attempt %d/5), retrying in %.1fs: %s",
                            self.provider,
                            query.id,
                            attempt + 1,
                            delay,
                            e,
                        )
                        await asyncio.sleep(delay)
                    else:
                        return _QueryResult(query=query, error=str(e))
            # Should not reach here, but just in case
            return _QueryResult(query=query, error="max retries exhausted")

        fetch_tasks = [_fetch_one(q) for q in queries]
        fetch_results: list[_QueryResult] = await asyncio.gather(
            *fetch_tasks,
            return_exceptions=True,
        )

        # Separate successes from failures
        successful: list[_QueryResult] = []
        for i, fr in enumerate(fetch_results):
            if isinstance(fr, Exception):
                query = queries[i]
                logger.error(
                    "%s: unexpected error fetching query %d: %s",
                    self.provider,
                    query.id,
                    fr,
                )
                result.errors.append(f"{self.provider}/query_{query.id}: {fr}")
            elif fr.error:
                logger.error(
                    "%s: failed to fetch query %d (%s): %s",
                    self.provider,
                    fr.query.id,
                    fr.query.query_text[:50],
                    fr.error,
                )
                result.errors.append(f"{self.provider}/query_{fr.query.id}: {fr.error}")
            else:
                successful.append(fr)

        logger.info(
            "%s: phase 1 done — %d/%d responses collected, %d errors",
            self.provider,
            len(successful),
            len(queries),
            len(result.errors),
        )

        # ==================================================================
        # Phase 2: Analyze + Save (OpenAI judge, rate-limited separately)
        # ==================================================================

        async def _analyze_and_save(qr: _QueryResult) -> None:
            """Analyze one response via judge and save to DB."""
            query = qr.query
            llm_resp = qr.llm_resp

            all_competitors = list(set(competitors + (query.competitors or [])))
            all_cited = llm_resp.cited_urls.copy()
            target_brand = query.target_brand or brand_name

            # Rate-limit OpenAI calls (entity extraction + judge) via rpm_limiter
            # and tpm_limiter passed into pipeline — each OpenAI call acquires
            # its own RPM token and TPM tokens before firing.
            # Semaphore limits total concurrent analysis tasks.
            async with judge_semaphore:
                analysis = await self._analyze_with_pipeline(
                    llm_resp=llm_resp,
                    query=query,
                    target_brand=target_brand,
                    competitors=all_competitors,
                    project_id=project_id,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    rpm_limiter=judge_rpm_limiter,
                    tpm_limiter=openai_tpm_limiter,
                )

            combined_urls = list(dict.fromkeys(all_cited + analysis.cited_urls))

            async with db_lock:
                await self._save_snapshot(
                    db=db,
                    llm_query=query,
                    today=today,
                    llm_model=llm_resp.model,
                    analysis=analysis,
                    raw_response=llm_resp.text,
                    response_tokens=llm_resp.tokens,
                    cost_usd=llm_resp.cost_usd,
                    cited_urls=combined_urls,
                )
                await self._upsert_discovered_entities(
                    db=db,
                    project_id=project_id,
                    discovered_entities=analysis.discovered_entities,
                    today=today,
                    competitors=all_competitors,
                    target_brand=target_brand,
                )

        analyze_tasks = [_analyze_and_save(qr) for qr in successful]
        analyze_outcomes = await asyncio.gather(*analyze_tasks, return_exceptions=True)

        for i, outcome in enumerate(analyze_outcomes):
            if outcome is None:
                result.collected += 1
            elif isinstance(outcome, Exception):
                qr = successful[i]
                logger.error(
                    "%s: analysis/save error for query %d (%s): %s",
                    self.provider,
                    qr.query.id,
                    qr.query.query_text[:50],
                    outcome,
                )
                result.errors.append(f"{self.provider}/query_{qr.query.id}: {outcome}")

        logger.info(
            "%s: two-phase collection done — %d collected, %d errors (of %d queries)",
            self.provider,
            result.collected,
            len(result.errors),
            len(queries),
        )
        return result

    async def _analyze_with_pipeline(
        self,
        llm_resp: LlmResponse,
        query: LlmQuery,
        target_brand: str,
        competitors: list[str],
        project_id: int,
        openai_api_key: str | None = None,
        gemini_api_key: str | None = None,
        rpm_limiter=None,
        tpm_limiter=None,
    ) -> AnalysisResult:
        """Analyze using Module 3 pipeline if available, otherwise fallback."""
        if not _PIPELINE_AVAILABLE:
            return self.analyze_response(
                response_text=llm_resp.text,
                target_brand=target_brand,
                competitors=competitors,
                prompt_text=query.query_text,
            )

        try:
            # Bridge LlmResponse → GatewayResponse for Module 3
            vendor_str = _VENDOR_MAP.get(self.provider, "chatgpt")
            gateway_resp = GatewayResponse(
                request_id=uuid.uuid4().hex[:16],
                vendor=GatewayVendor(vendor_str),
                model_version=llm_resp.model,
                raw_response_text=llm_resp.text,
                cited_urls=llm_resp.cited_urls,
                total_tokens=llm_resp.tokens,
                cost_usd=llm_resp.cost_usd,
                tenant_id=str(self.tenant_id),
                project_id=project_id,
                llm_query_id=query.id,
            )

            # Build competitors dict with auto-generated transliteration aliases
            comp_dict = None
            if competitors:
                from app.analysis.llm_entity_extractor import generate_transliteration_aliases

                comp_dict = {c: generate_transliteration_aliases(c) for c in competitors}

            # Use OpenAI key for judge + entity extraction:
            # 1) Explicit openai_api_key from tenant (works for all providers)
            # 2) Self api_key if this is the ChatGPT provider
            # 3) None → falls back to OPENAI_API_KEY env var in pipeline
            judge_key = openai_api_key or (self.api_key if self.provider == "chatgpt" else None)

            # Also auto-generate transliteration aliases for target brand
            target_aliases = generate_transliteration_aliases(target_brand)

            analyzed = await pipeline_analyze(
                response=gateway_resp,
                target_brand=target_brand,
                target_aliases=target_aliases,
                competitors=comp_dict,
                intent=query.query_type or "",
                judge_api_key=judge_key,
                prompt_text=query.query_text,
                rpm_limiter=rpm_limiter,
                tpm_limiter=tpm_limiter,
                gemini_api_key=gemini_api_key,
            )

            # Extract structure type safely
            _struct_type = None
            if hasattr(analyzed, "structure") and analyzed.structure:
                st = analyzed.structure.structure_type
                _struct_type = st.value if hasattr(st, "value") else str(st)

            # Extract sentiment label safely
            _sent_label = None
            if hasattr(analyzed.target_brand, "sentiment_label") and analyzed.target_brand.sentiment_label:
                sl = analyzed.target_brand.sentiment_label
                _sent_label = sl.value if hasattr(sl, "value") else str(sl)

            # Convert AnalyzedResponse → AnalysisResult (our internal format)
            result = AnalysisResult(
                brand_mentioned=analyzed.target_brand.is_mentioned,
                mention_type=analyzed.target_brand.mention_type.value,
                mention_context=analyzed.target_brand.mention_context,
                competitor_mentions={c.name: c.is_mentioned for c in analyzed.competitors},
                cited_urls=[c.url for c in analyzed.citations],
                sentiment_label=_sent_label,
                position_rank=analyzed.target_brand.position_rank,
                structure_type=_struct_type,
                is_hallucination=getattr(analyzed.target_brand, "is_hallucination", None),
                discovered_entities=getattr(analyzed, "discovered_entities", []),
                raw_extracted_entities=getattr(analyzed, "raw_entities", []),
            )

            logger.debug(
                "Pipeline analysis: query=%d brand_mentioned=%s mention_type=%s RVS=%.4f",
                query.id,
                result.brand_mentioned,
                result.mention_type,
                analyzed.response_visibility_score,
            )
            return result

        except Exception as e:
            logger.warning(
                "Pipeline analysis failed for query %d, falling back to heuristics: %s",
                query.id,
                e,
            )
            return self.analyze_response(
                response_text=llm_resp.text,
                target_brand=target_brand,
                competitors=competitors,
                prompt_text=query.query_text,
            )

    def analyze_response(
        self,
        response_text: str,
        target_brand: str,
        competitors: list[str],
        prompt_text: str = "",
    ) -> AnalysisResult:
        """Analyze an LLM response using regex heuristics (fallback method).

        Brands already mentioned in the prompt_text are not counted as
        organic mentions in the response.
        """
        result = AnalysisResult()

        if not target_brand:
            return result

        # Determine which brands are in the prompt (should not count as mentions)
        prompt_lower = prompt_text.lower() if prompt_text else ""

        # --- Brand mention detection ---
        brand_in_prompt = bool(prompt_lower and target_brand.lower() in prompt_lower)
        if not brand_in_prompt:
            brand_pattern = re.compile(
                r"\b" + re.escape(target_brand) + r"\b",
                re.IGNORECASE,
            )
            brand_match = brand_pattern.search(response_text)
            result.brand_mentioned = brand_match is not None

            if brand_match:
                # Extract context: ~150 chars around the first mention
                start = max(0, brand_match.start() - 75)
                end = min(len(response_text), brand_match.end() + 75)
                result.mention_context = response_text[start:end].strip()

                # Classify mention type using surrounding context (~300 chars)
                ctx_start = max(0, brand_match.start() - 150)
                ctx_end = min(len(response_text), brand_match.end() + 150)
                context_window = response_text[ctx_start:ctx_end]

                result.mention_type = self._classify_mention(context_window, competitors)

        # --- Competitor detection ---
        for comp in competitors:
            # Skip if competitor is in the prompt
            if prompt_lower and comp.lower() in prompt_lower:
                result.competitor_mentions[comp] = False
                continue
            comp_pattern = re.compile(r"\b" + re.escape(comp) + r"\b", re.IGNORECASE)
            result.competitor_mentions[comp] = bool(comp_pattern.search(response_text))

        # --- URL extraction ---
        url_pattern = re.compile(r"https?://[^\s\)\]\}\"'<>,]+")
        found_urls = url_pattern.findall(response_text)
        # Clean trailing punctuation
        result.cited_urls = [url.rstrip(".,;:!?)") for url in found_urls]

        return result

    @staticmethod
    def _classify_mention(context: str, competitors: list[str]) -> str:
        """Classify the type of brand mention based on surrounding text."""
        has_competitor = any(re.search(r"\b" + re.escape(c) + r"\b", context, re.IGNORECASE) for c in competitors)

        if _NEGATIVE_PATTERNS.search(context):
            return "negative"
        if has_competitor and _COMPARE_PATTERNS.search(context):
            return "compared"
        if _RECOMMEND_PATTERNS.search(context):
            return "recommended"
        return "direct"

    async def _save_snapshot(
        self,
        db: AsyncSession,
        llm_query: LlmQuery,
        today: date,
        llm_model: str,
        analysis: AnalysisResult,
        raw_response: str,
        response_tokens: int,
        cost_usd: float,
        cited_urls: list[str],
    ) -> None:
        """Upsert a snapshot into llm_snapshots."""
        stmt = (
            pg_insert(LlmSnapshot)
            .values(
                tenant_id=self.tenant_id,
                llm_query_id=llm_query.id,
                date=today,
                llm_provider=self.provider,
                llm_model=llm_model,
                brand_mentioned=analysis.brand_mentioned,
                mention_type=analysis.mention_type,
                mention_context=analysis.mention_context[:1000] if analysis.mention_context else None,
                competitor_mentions=analysis.competitor_mentions or None,
                cited_urls=cited_urls or None,
                raw_entities=analysis.raw_extracted_entities or None,
                sentiment_label=analysis.sentiment_label,
                position_rank=analysis.position_rank,
                structure_type=analysis.structure_type,
                is_hallucination=analysis.is_hallucination,
                raw_response=raw_response,
                response_tokens=response_tokens,
                cost_usd=cost_usd,
                collected_at=datetime.now(timezone.utc),
            )
            .on_conflict_do_update(
                constraint="uq_llm_snapshot",
                set_={
                    "llm_model": llm_model,
                    "brand_mentioned": analysis.brand_mentioned,
                    "mention_type": analysis.mention_type,
                    "mention_context": analysis.mention_context[:1000] if analysis.mention_context else None,
                    "competitor_mentions": analysis.competitor_mentions or None,
                    "cited_urls": cited_urls or None,
                    "raw_entities": analysis.raw_extracted_entities or None,
                    "sentiment_label": analysis.sentiment_label,
                    "position_rank": analysis.position_rank,
                    "structure_type": analysis.structure_type,
                    "is_hallucination": analysis.is_hallucination,
                    "raw_response": raw_response,
                    "response_tokens": response_tokens,
                    "cost_usd": cost_usd,
                    "collected_at": datetime.now(timezone.utc),
                },
            )
        )
        await db.execute(stmt)
        await db.commit()

    async def _upsert_discovered_entities(
        self,
        db: AsyncSession,
        project_id: int,
        discovered_entities: list[str],
        today: date,
        competitors: list[str],
        target_brand: str,
    ) -> None:
        """Upsert discovered entities from LLM response analysis.

        Skips entities that match the target brand or existing competitors.
        Non-fatal: errors are logged but do not break the save flow.
        """
        if not discovered_entities:
            return

        from app.analysis.llm_entity_extractor import _names_match

        # Build list of known names for fuzzy dedup
        known_names = [target_brand] + list(competitors)

        try:
            for entity_name in discovered_entities:
                # Use fuzzy matching (same as pipeline) to skip known brands
                if any(_names_match(entity_name, kn) for kn in known_names):
                    continue

                stmt = (
                    pg_insert(DiscoveredEntity)
                    .values(
                        tenant_id=self.tenant_id,
                        project_id=project_id,
                        entity_name=entity_name,
                        mention_count=1,
                        first_seen=today,
                        last_seen=today,
                        status="pending",
                    )
                    .on_conflict_do_update(
                        constraint="uq_project_entity",
                        set_={
                            "mention_count": DiscoveredEntity.mention_count + 1,
                            "last_seen": today,
                        },
                    )
                )
                await db.execute(stmt)
            await db.commit()
        except Exception as e:
            logger.warning(
                "Failed to upsert discovered entities (non-fatal): %s",
                e,
            )
            await db.rollback()
