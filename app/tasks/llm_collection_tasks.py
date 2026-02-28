"""Celery tasks for LLM visibility data collection.

Supports two-level parallelism:
  Level 1: All LLM providers run concurrently (asyncio.gather)
  Level 2: Queries within each provider run concurrently (semaphore in BaseLlmCollector)

Collection uses a two-phase approach per provider:
  Phase 1: Query the LLM provider (provider-rate-limited)
  Phase 2: Analyze via OpenAI judge (judge-rate-limited, shared across providers)
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from app.collectors.llm_base import RpmLimiter, TpmLimiter
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@dataclass
class _ConcurrencyContext:
    """Shared concurrency primitives passed to each provider collector."""

    openai_semaphore: asyncio.Semaphore  # Global limit for OpenAI judge/entity calls
    judge_rpm_limiter: RpmLimiter  # Shared RPM limiter for judge/entity calls
    openai_tpm_limiter: TpmLimiter  # Shared TPM limiter for ALL OpenAI calls
    session_factory: Any  # async_sessionmaker — each provider gets its own session


def _run_async(coro):
    """Run an async coroutine from sync Celery task context.

    Creates a fresh event loop each time to avoid conflicts with
    the module-level SQLAlchemy engine (which may be bound to a
    different loop created by uvicorn).
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_session_factory():
    """Create a fresh async engine + session factory for Celery worker context.

    The module-level engine from app.db.postgres is bound to uvicorn's event loop
    and cannot be reused in a new event loop created by _run_async().
    Pool size increased for parallel provider execution.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.core.config import settings

    engine = create_async_engine(
        settings.postgres_url,
        echo=False,
        pool_size=10,
        max_overflow=10,
        pool_pre_ping=True,
    )
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False), engine


# Map of provider name → (module_path, class_name, api_key_field)
_PROVIDER_MAP = {
    "chatgpt": ("app.collectors.llm_openai", "OpenAiCollector", "openai_api_key"),
    "deepseek": ("app.collectors.llm_deepseek", "DeepSeekCollector", "deepseek_api_key"),
    "perplexity": ("app.collectors.llm_perplexity", "PerplexityCollector", "perplexity_api_key"),
    "yandexgpt": ("app.collectors.llm_yandexgpt", "YandexGptCollector", "yandexgpt_api_key"),
    "gemini": ("app.collectors.llm_gemini", "GeminiCollector", "gemini_api_key"),
    "gigachat": ("app.collectors.llm_gigachat", "GigaChatCollector", "gigachat_api_key"),
}


async def _collect_single_provider(
    *,
    provider_name: str,
    collector,
    queries: list,
    brand_name: str,
    openai_api_key: str | None,
    gemini_api_key: str | None,
    project_id: int,
    concurrency_ctx: _ConcurrencyContext,
) -> dict:
    """Collect data for a single provider using its own DB session.

    Each provider gets its own AsyncSession from the shared session_factory
    to avoid concurrent write conflicts.
    """
    async with concurrency_ctx.session_factory() as db:
        try:
            collection_result = await collector.collect(
                db=db,
                project_id=project_id,
                queries=queries,
                brand_name=brand_name,
                competitors=None,  # competitors are per-query in LlmQuery model
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
                concurrency_ctx=concurrency_ctx,
            )
            return {
                "collected": collection_result.collected,
                "errors": collection_result.errors,
            }
        except Exception as e:
            logger.error(
                "LLM collection failed for provider=%s project=%d: %s",
                provider_name,
                project_id,
                e,
            )
            return {"error": str(e)}


async def _collect_llm_project_async(tenant_id: str, project_id: int) -> dict:
    """Run LLM collection for a single project across all configured providers.

    All providers are launched in parallel via asyncio.gather.
    Each provider processes its queries concurrently (semaphore-limited).
    """
    import importlib

    from sqlalchemy import select

    from app.core.encryption import decrypt_value
    from app.models.llm_query import LlmQuery
    from app.models.project import Project
    from app.models.tenant import Tenant

    tid = UUID(tenant_id)

    session_factory, engine = _make_session_factory()
    try:
        # --- Phase 1: Load project/tenant/queries (single DB session) ---
        async with session_factory() as db:
            # Load project
            project_result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tid))
            project = project_result.scalar_one_or_none()
            if not project:
                return {"error": f"Project {project_id} not found"}

            if not project.track_llm:
                return {"error": "LLM tracking not enabled", "project_id": project_id}

            # Load tenant for API keys
            tenant = await db.get(Tenant, tid)
            if not tenant:
                return {"error": "Tenant not found"}

            # Load active queries
            queries_result = await db.execute(
                select(LlmQuery).where(
                    LlmQuery.project_id == project_id,
                    LlmQuery.tenant_id == tid,
                    LlmQuery.is_active == True,  # noqa: E712
                )
            )
            queries = queries_result.scalars().all()
            if not queries:
                return {"project_id": project_id, "collected": 0, "message": "No active queries"}

            providers = project.llm_providers or ["chatgpt"]

            # Decrypt API keys for entity extraction + judge + consolidation
            _openai_key_encrypted = getattr(tenant, "openai_api_key", None)
            openai_api_key = decrypt_value(_openai_key_encrypted) if _openai_key_encrypted else None

            _gemini_key_encrypted = getattr(tenant, "gemini_api_key", None)
            gemini_api_key = decrypt_value(_gemini_key_encrypted) if _gemini_key_encrypted else None

            _perplexity_key_encrypted = getattr(tenant, "perplexity_api_key", None)
            perplexity_api_key = decrypt_value(_perplexity_key_encrypted) if _perplexity_key_encrypted else None

            # Cache decrypted keys and tenant data (session will close after this block)
            brand_name = project.brand_name or project.name
            brand_aliases = project.brand_aliases or []
            project_competitors = project.competitors or []
            project_name = project.name
            brand_sub_brands = project.brand_sub_brands or {}
            competitor_sub_brands = project.competitor_sub_brands or {}

            # Prepare provider collectors (validate keys before launching parallel tasks)
            provider_collectors: dict[str, Any] = {}  # name → collector instance
            results: dict[str, dict] = {}

            for provider_name in providers:
                if provider_name not in _PROVIDER_MAP:
                    results[provider_name] = {"error": f"Unknown provider: {provider_name}"}
                    continue

                module_path, class_name, key_field = _PROVIDER_MAP[provider_name]

                # Get encrypted API key from tenant
                encrypted_key = getattr(tenant, key_field, None)
                if not encrypted_key:
                    results[provider_name] = {"error": f"API key not configured for {provider_name}"}
                    continue

                api_key = decrypt_value(encrypted_key)
                if not api_key:
                    results[provider_name] = {"error": f"Failed to decrypt API key for {provider_name}"}
                    continue

                # Dynamically import collector class
                module = importlib.import_module(module_path)
                collector_cls = getattr(module, class_name)

                # Build extra kwargs for providers that need more than api_key
                extra_kwargs = {}
                if provider_name == "yandexgpt":
                    encrypted_folder = getattr(tenant, "yandexgpt_folder_id", None)
                    if not encrypted_folder:
                        results[provider_name] = {"error": "YandexGPT folder_id not configured"}
                        continue
                    folder_id = decrypt_value(encrypted_folder)
                    if not folder_id:
                        results[provider_name] = {"error": "Failed to decrypt YandexGPT folder_id"}
                        continue
                    extra_kwargs["folder_id"] = folder_id

                collector = collector_cls(api_key=api_key, tenant_id=tid, **extra_kwargs)
                provider_collectors[provider_name] = collector

        # --- Phase 2: Launch all providers in parallel ---
        if provider_collectors:
            # Shared concurrency context: OpenAI semaphore + RPM/TPM limiters + session factory.
            # All limiters are shared across ALL providers so they don't
            # exceed the OpenAI quota collectively (queries + entity extraction + judge).
            from app.collectors.llm_base import _JUDGE_CONCURRENCY, _JUDGE_RPM, _OPENAI_TPM

            concurrency_ctx = _ConcurrencyContext(
                openai_semaphore=asyncio.Semaphore(_JUDGE_CONCURRENCY),
                judge_rpm_limiter=RpmLimiter(rpm=_JUDGE_RPM, burst=_JUDGE_CONCURRENCY),
                openai_tpm_limiter=TpmLimiter(tpm=_OPENAI_TPM),
                session_factory=session_factory,
            )

            logger.info(
                "Launching parallel LLM collection: project=%d providers=%s queries=%d",
                project_id,
                list(provider_collectors.keys()),
                len(queries),
            )

            provider_coros = [
                _collect_single_provider(
                    provider_name=pname,
                    collector=collector,
                    queries=queries,
                    brand_name=brand_name,
                    openai_api_key=openai_api_key,
                    gemini_api_key=gemini_api_key,
                    project_id=project_id,
                    concurrency_ctx=concurrency_ctx,
                )
                for pname, collector in provider_collectors.items()
            ]

            provider_outcomes = await asyncio.gather(*provider_coros, return_exceptions=True)

            for pname, outcome in zip(provider_collectors.keys(), provider_outcomes):
                if isinstance(outcome, Exception):
                    logger.error(
                        "Provider %s crashed: %s",
                        pname,
                        outcome,
                    )
                    results[pname] = {"error": str(outcome)}
                else:
                    results[pname] = outcome

        # --- Phase 3: Post-collection Entity Consolidation ---
        # Uses LLM to cluster entity name variants (e.g. "Группа ПИК" + "ПИК" → "ПИК")
        # and rewrite competitor_mentions / discovered_entities with canonical names.
        consolidation_result = None
        if (openai_api_key or gemini_api_key) and provider_collectors:
            try:
                from datetime import date as date_type

                from app.analysis.entity_consolidator import run_post_collection_consolidation

                today = date_type.today()
                async with session_factory() as db:
                    consolidation_result = await run_post_collection_consolidation(
                        db=db,
                        project_id=project_id,
                        tenant_id=tid,
                        today=today,
                        brand_name=brand_name,
                        brand_aliases=brand_aliases,
                        competitors=project_competitors,
                        openai_api_key=openai_api_key,
                        rpm_limiter=concurrency_ctx.judge_rpm_limiter,
                        tpm_limiter=concurrency_ctx.openai_tpm_limiter,
                        gemini_api_key=gemini_api_key,
                        brand_sub_brands=brand_sub_brands,
                        competitor_sub_brands=competitor_sub_brands,
                        perplexity_api_key=perplexity_api_key,
                    )
                logger.info(
                    "Phase 3 consolidation done: project=%d, clusters=%d, merged=%d, snapshots_updated=%d",
                    project_id,
                    consolidation_result.clusters_found,
                    consolidation_result.entities_merged,
                    consolidation_result.snapshots_updated,
                )
            except Exception as e:
                logger.warning(
                    "Phase 3 consolidation failed (non-fatal): project=%d, error=%s",
                    project_id,
                    e,
                )

        response = {
            "project_id": project_id,
            "project_name": project_name,
            "providers": results,
        }
        if consolidation_result:
            response["consolidation"] = {
                "clusters_found": consolidation_result.clusters_found,
                "entities_merged": consolidation_result.entities_merged,
                "brand_matches": consolidation_result.brand_matches,
                "competitor_matches": consolidation_result.competitor_matches,
                "new_discovered": consolidation_result.new_discovered,
                "snapshots_updated": consolidation_result.snapshots_updated,
            }
        return response
    finally:
        await engine.dispose()


@celery_app.task(
    bind=True,
    name="collect_llm_project",
    max_retries=0,
)
def collect_llm_project_task(self, tenant_id: str, project_id: int):
    """Celery task: collect LLM visibility data for a single project.

    Retries are handled at the provider/query level (exponential backoff + jitter),
    so Celery-level retries are disabled to avoid duplicate task execution.
    """
    logger.info("Starting LLM collection for tenant=%s project=%d", tenant_id, project_id)
    try:
        result = _run_async(_collect_llm_project_async(tenant_id, project_id))
        logger.info("LLM collection done for project %d: %s", project_id, result)
        return result
    except Exception as exc:
        logger.error("LLM collection failed for project %d: %s", project_id, exc)
        return {"error": str(exc), "project_id": project_id}


# ---------------------------------------------------------------------------
# Re-consolidation task (Phase 3 only, no data collection)
# ---------------------------------------------------------------------------


async def _reconsolidate_async(tenant_id: str, project_id: int, date_str: str | None = None):
    """Run entity consolidation (Phase 3) without re-collecting data."""
    from datetime import date as date_type, datetime

    from sqlalchemy import select

    from app.core.encryption import decrypt_value

    tid = UUID(tenant_id)
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else date_type.today()

    session_factory, engine = _make_session_factory()

    try:
        async with session_factory() as db:
            from app.models.project import Project
            from app.models.tenant import Tenant

            project = (await db.execute(select(Project).where(Project.id == project_id))).scalar_one_or_none()
            if not project:
                return {"error": "Project not found"}

            tenant = (await db.execute(select(Tenant).where(Tenant.id == tid))).scalar_one_or_none()
            if not tenant:
                return {"error": "Tenant not found"}

            encrypted_key = getattr(tenant, "openai_api_key", None)
            openai_api_key = decrypt_value(encrypted_key) if encrypted_key else None

            encrypted_gemini = getattr(tenant, "gemini_api_key", None)
            gemini_api_key = decrypt_value(encrypted_gemini) if encrypted_gemini else None

            encrypted_pplx = getattr(tenant, "perplexity_api_key", None)
            perplexity_api_key = decrypt_value(encrypted_pplx) if encrypted_pplx else None

            if not openai_api_key and not gemini_api_key:
                return {"error": "No API key configured (need OpenAI or Gemini)"}

            brand_name = project.brand_name or project.name
            brand_aliases = project.brand_aliases or []
            competitors = project.competitors or []
            brand_sub_brands = project.brand_sub_brands or {}
            competitor_sub_brands = project.competitor_sub_brands or {}

        # Run consolidation with fresh DB session
        from app.analysis.entity_consolidator import run_post_collection_consolidation

        rpm_limiter = RpmLimiter(rpm=500)
        tpm_limiter = TpmLimiter(tpm=2_000_000)

        async with session_factory() as db:
            result = await run_post_collection_consolidation(
                db=db,
                project_id=project_id,
                tenant_id=tid,
                today=target_date,
                brand_name=brand_name,
                brand_aliases=brand_aliases,
                competitors=competitors,
                openai_api_key=openai_api_key,
                rpm_limiter=rpm_limiter,
                tpm_limiter=tpm_limiter,
                gemini_api_key=gemini_api_key,
                brand_sub_brands=brand_sub_brands,
                competitor_sub_brands=competitor_sub_brands,
                perplexity_api_key=perplexity_api_key,
            )

        return {
            "project_id": project_id,
            "date": str(target_date),
            "clusters_found": result.clusters_found,
            "entities_merged": result.entities_merged,
            "brand_matches": result.brand_matches,
            "competitor_matches": result.competitor_matches,
            "new_discovered": result.new_discovered,
            "snapshots_updated": result.snapshots_updated,
        }
    finally:
        await engine.dispose()


@celery_app.task(bind=True, name="reconsolidate_entities", max_retries=0)
def reconsolidate_entities_task(self, tenant_id: str, project_id: int, date_str: str | None = None):
    """Celery task: re-run entity consolidation (Phase 3) for a project."""
    logger.info("Starting re-consolidation for tenant=%s project=%d date=%s", tenant_id, project_id, date_str)
    try:
        result = _run_async(_reconsolidate_async(tenant_id, project_id, date_str))
        logger.info("Re-consolidation done for project %d: %s", project_id, result)
        return result
    except Exception as exc:
        logger.error("Re-consolidation failed for project %d: %s", project_id, exc)
        return {"error": str(exc), "project_id": project_id}
