"""Celery tasks for data collection."""

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _make_session_factory():
    """Create a fresh async engine + session factory for Celery worker context.

    The module-level engine from app.db.postgres is bound to uvicorn's event loop
    and cannot be reused in a new event loop created by _run_async().
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.core.config import settings

    engine = create_async_engine(
        settings.postgres_url,
        echo=settings.app_debug,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
    )
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ---------------------------------------------------------------------------
#  Dispatcher: runs every minute via Celery Beat.
#  Queries all active tenants and checks if current UTC hour:minute matches
#  their collection_hour:collection_minute. If so, dispatches collection.
# ---------------------------------------------------------------------------


async def _find_tenants_to_collect() -> list[str]:
    """Return tenant_ids whose scheduled time matches current UTC time."""
    from sqlalchemy import select

    from app.models.tenant import Tenant

    now = datetime.now(timezone.utc)
    current_hour = now.hour
    current_minute = now.minute

    session_factory = _make_session_factory()
    async with session_factory() as db:
        result = await db.execute(
            select(Tenant.id).where(
                Tenant.is_active == True,  # noqa: E712
                Tenant.collection_hour == current_hour,
                Tenant.collection_minute == current_minute,
            )
        )
        return [str(row[0]) for row in result.all()]


@celery_app.task(name="dispatch_collections")
def dispatch_collections_task():
    """Beat dispatcher: check which tenants need collection now and fire tasks."""
    tenant_ids = _run_async(_find_tenants_to_collect())
    if not tenant_ids:
        return {"dispatched": 0}

    for tid in tenant_ids:
        collect_all_projects_task.delay(tid)
        logger.info("Dispatched collection for tenant %s", tid)

    return {"dispatched": len(tenant_ids), "tenant_ids": tenant_ids}


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


async def _collect_project_async(tenant_id: str, project_id: int) -> dict:
    """Run collection for a single project using async DB session."""
    from app.services.collection_service import collect_project

    session_factory = _make_session_factory()
    async with session_factory() as db:
        result = await collect_project(db, UUID(tenant_id), project_id)
        return {
            "project_id": result.project_id,
            "project_name": result.project_name,
            "domain": result.domain,
            "justmagic_collected": result.justmagic.collected if result.justmagic else 0,
            "ywm_collected": result.ywm.collected if result.ywm else 0,
            "gsc_collected": result.gsc.collected if result.gsc else 0,
            "errors": result.errors,
        }


async def _find_llm_projects(tenant_id: str) -> list[int]:
    """Return project IDs with track_llm=True for the given tenant."""
    from sqlalchemy import select

    from app.models.project import Project

    tid = UUID(tenant_id)
    session_factory = _make_session_factory()
    async with session_factory() as db:
        result = await db.execute(
            select(Project.id).where(
                Project.tenant_id == tid,
                Project.is_active == True,  # noqa: E712
                Project.track_llm == True,  # noqa: E712
            )
        )
        return [row[0] for row in result.all()]


async def _collect_all_async(tenant_id: str) -> list[dict]:
    """Run collection for all active projects of a tenant."""
    from app.core.encryption import decrypt_value
    from app.models.tenant import Tenant
    from app.notifications.telegram import send_daily_report
    from app.services.collection_service import collect_all_projects

    session_factory = _make_session_factory()
    async with session_factory() as db:
        results = await collect_all_projects(db, UUID(tenant_id))

        # Send Telegram report if configured
        tenant = await db.get(Tenant, UUID(tenant_id))
        if tenant and tenant.telegram_bot_token and tenant.telegram_chat_id:
            bot_token = decrypt_value(tenant.telegram_bot_token)
            chat_id = decrypt_value(tenant.telegram_chat_id)
            if bot_token and chat_id:
                await send_daily_report(results, bot_token, chat_id)

    return [
        {
            "project_id": r.project_id,
            "project_name": r.project_name,
            "justmagic_collected": r.justmagic.collected if r.justmagic else 0,
            "ywm_collected": r.ywm.collected if r.ywm else 0,
            "gsc_collected": r.gsc.collected if r.gsc else 0,
            "errors": r.errors,
        }
        for r in results
    ]


@celery_app.task(
    bind=True,
    name="collect_project",
    max_retries=2,
    default_retry_delay=60,
)
def collect_project_task(self, tenant_id: str, project_id: int):
    """Celery task: collect data for a single project."""
    logger.info("Starting collection for tenant=%s project=%d", tenant_id, project_id)
    try:
        result = _run_async(_collect_project_async(tenant_id, project_id))
        logger.info("Collection done for project %d: %s", project_id, result)
        return result
    except Exception as exc:
        logger.error("Collection failed for project %d: %s", project_id, exc)
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    name="collect_all_projects",
    max_retries=1,
    default_retry_delay=120,
)
def collect_all_projects_task(self, tenant_id: str):
    """Celery task: collect data for all active projects of a tenant."""
    logger.info("Starting collection for all projects, tenant=%s", tenant_id)
    try:
        results = _run_async(_collect_all_async(tenant_id))
        logger.info("Collection done for tenant %s: %d projects", tenant_id, len(results))

        # Dispatch LLM collection for projects that have track_llm enabled
        _dispatch_llm_collection(tenant_id)

        return results
    except Exception as exc:
        logger.error("Collection failed for tenant %s: %s", tenant_id, exc)
        raise self.retry(exc=exc)


def _dispatch_llm_collection(tenant_id: str) -> int:
    """Dispatch LLM collection tasks for all projects with track_llm=True."""
    from app.tasks.llm_collection_tasks import collect_llm_project_task

    project_ids = _run_async(_find_llm_projects(tenant_id))
    for pid in project_ids:
        collect_llm_project_task.delay(tenant_id, pid)
        logger.info("Dispatched LLM collection for tenant=%s project=%d", tenant_id, pid)
    return len(project_ids)
