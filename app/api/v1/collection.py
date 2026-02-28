"""Collection API endpoints — trigger and check collection tasks."""

import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id, require_role
from app.core.exceptions import BadRequestError, NotFoundError
from app.core.rate_limit import limiter
from app.db.postgres import get_db
from app.models.keyword import Keyword
from app.models.llm_query import LlmQuery
from app.models.project import Project
from app.schemas.common import MessageResponse

router = APIRouter(prefix="/collection", tags=["collection"])


@router.post(
    "/projects/{project_id}",
    response_model=MessageResponse,
    status_code=202,
    dependencies=[Depends(require_role("member"))],
)
@limiter.limit("5/minute")
async def trigger_project_collection(
    request: Request,
    project_id: int,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Start async collection for a single project via Celery. Requires member role."""
    # Pre-flight: verify project exists, belongs to tenant, and has keywords
    project = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = project.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    if not project.is_active:
        raise BadRequestError("Project is inactive")

    keyword_count = await db.execute(
        select(func.count())
        .select_from(Keyword)
        .where(
            Keyword.project_id == project_id,
            Keyword.is_active == True,  # noqa: E712
        )
    )
    if keyword_count.scalar() == 0:
        raise BadRequestError("Project has no active keywords")

    from app.tasks.collection_tasks import collect_project_task

    task = collect_project_task.delay(str(tenant_id), project_id)
    return MessageResponse(message=f"Collection started (task_id={task.id})")


@router.post(
    "/all",
    response_model=MessageResponse,
    status_code=202,
    dependencies=[Depends(require_role("member"))],
)
@limiter.limit("2/minute")
async def trigger_all_collection(
    request: Request,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
):
    """Start async collection for all projects via Celery. Requires member role."""
    from app.tasks.collection_tasks import collect_all_projects_task

    task = collect_all_projects_task.delay(str(tenant_id))
    return MessageResponse(message=f"Collection started for all projects (task_id={task.id})")


@router.post(
    "/projects/{project_id}/llm",
    response_model=MessageResponse,
    status_code=202,
    dependencies=[Depends(require_role("member"))],
)
@limiter.limit("5/minute")
async def trigger_llm_collection(
    request: Request,
    project_id: int,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Start async LLM collection for a single project via Celery. Requires member role."""
    project = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = project.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    if not project.is_active:
        raise BadRequestError("Project is inactive")
    if not project.track_llm:
        raise BadRequestError("LLM tracking is not enabled for this project")

    query_count = await db.execute(
        select(func.count())
        .select_from(LlmQuery)
        .where(
            LlmQuery.project_id == project_id,
            LlmQuery.is_active == True,  # noqa: E712
        )
    )
    if query_count.scalar() == 0:
        raise BadRequestError("Project has no active LLM queries")

    from app.tasks.llm_collection_tasks import collect_llm_project_task

    task = collect_llm_project_task.delay(str(tenant_id), project_id)
    return MessageResponse(message=f"LLM collection started (task_id={task.id})")


@router.post(
    "/projects/{project_id}/reconsolidate",
    response_model=MessageResponse,
    status_code=202,
    dependencies=[Depends(require_role("member"))],
)
@limiter.limit("5/minute")
async def trigger_reconsolidation(
    request: Request,
    project_id: int,
    date: str | None = None,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Re-run entity consolidation (Phase 3) for a project. Optional date param (YYYY-MM-DD)."""
    project = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = project.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")

    from app.tasks.llm_collection_tasks import reconsolidate_entities_task

    task = reconsolidate_entities_task.delay(str(tenant_id), project_id, date)
    return MessageResponse(message=f"Re-consolidation started (task_id={task.id})")


@router.get("/status/{task_id}")
async def get_task_status(
    task_id: str,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
):
    """Check Celery task status. Requires authentication."""
    from app.tasks.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": result.status,
    }
    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)
    return response
