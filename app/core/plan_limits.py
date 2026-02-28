"""Plan limit enforcement — check tenant quotas before creating resources."""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import PlanLimitError
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.tenant import Tenant


async def check_project_limit(db: AsyncSession, tenant_id: UUID) -> None:
    """Raise PlanLimitError if tenant has reached max_projects."""
    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise PlanLimitError("Tenant not found")

    result = await db.execute(
        select(func.count())
        .select_from(Project)
        .where(
            Project.tenant_id == tenant_id,
            Project.is_active == True,  # noqa: E712
        )
    )
    current_count = result.scalar() or 0

    if current_count >= tenant.max_projects:
        raise PlanLimitError(
            f"Plan '{tenant.plan}' allows max {tenant.max_projects} projects. "
            f"Current: {current_count}. Upgrade your plan to add more."
        )


async def check_keyword_limit(db: AsyncSession, tenant_id: UUID, adding: int = 1) -> None:
    """Raise PlanLimitError if total keywords across tenant would exceed max_keywords."""
    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise PlanLimitError("Tenant not found")

    result = await db.execute(
        select(func.count())
        .select_from(Keyword)
        .where(
            Keyword.tenant_id == tenant_id,
            Keyword.is_active == True,  # noqa: E712
        )
    )
    current_count = result.scalar() or 0

    if current_count + adding > tenant.max_keywords:
        raise PlanLimitError(
            f"Plan '{tenant.plan}' allows max {tenant.max_keywords} keywords. "
            f"Current: {current_count}, adding: {adding}. Upgrade your plan to add more."
        )


async def check_llm_query_limit(db: AsyncSession, tenant_id: UUID, adding: int = 1) -> None:
    """No-op — LLM query limits disabled."""
    return
