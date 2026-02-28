"""LLM Queries CRUD — manage prompts for LLM visibility monitoring."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id
from app.core.exceptions import NotFoundError
from app.core.plan_limits import check_llm_query_limit
from app.db.postgres import get_db
from app.models.llm_query import LlmQuery
from app.models.project import Project
from app.schemas.llm_query import (
    LlmQueryBulkAction,
    LlmQueryBulkCreate,
    LlmQueryBulkResponse,
    LlmQueryBulkToggle,
    LlmQueryCreate,
    LlmQueryResponse,
    LlmQueryUpdate,
)

router = APIRouter(prefix="/projects/{project_id}/llm-queries", tags=["llm-queries"])

_SORT_COLUMNS = {
    "created_at": LlmQuery.created_at,
    "query_text": LlmQuery.query_text,
    "query_class": LlmQuery.query_class,
    "is_active": LlmQuery.is_active,
}


async def _get_project(project_id: int, tenant_id: UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    return project


def _query_to_response(q: LlmQuery) -> LlmQueryResponse:
    """Convert LlmQuery ORM instance to response."""
    resp = LlmQueryResponse.model_validate(q)
    resp.tags = []
    return resp


# ── List with filters & sorting ──────────────────────────────────


@router.get("/")
async def list_llm_queries(
    project_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    query_class: str | None = Query(None),
    query_subtype: str | None = Query(None),
    is_active: bool | None = Query(None),
    search: str | None = Query(None),
    scenario: str | None = Query(None, description="Filter by scenario/category name"),
    sort_by: str = Query("created_at"),
    sort_dir: str = Query("desc", pattern=r"^(asc|desc)$"),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    base_where = (LlmQuery.project_id == project_id) & (LlmQuery.tenant_id == tenant_id)

    if query_class:
        if query_class == "custom":
            base_where = base_where & (LlmQuery.query_class.notin_(["thematic", "branded"]))
        else:
            base_where = base_where & (LlmQuery.query_class == query_class)
    if query_subtype:
        base_where = base_where & (LlmQuery.query_subtype == query_subtype)
    if is_active is not None:
        base_where = base_where & (LlmQuery.is_active == is_active)
    if search:
        base_where = base_where & (LlmQuery.query_text.ilike(f"%{search}%"))
    if scenario:
        base_where = base_where & (LlmQuery.category == scenario)

    count_result = await db.execute(select(func.count(LlmQuery.id)).where(base_where))
    total = count_result.scalar() or 0

    col = _SORT_COLUMNS.get(sort_by, LlmQuery.created_at)
    order = col.desc() if sort_dir == "desc" else col.asc()

    result = await db.execute(select(LlmQuery).where(base_where).order_by(order).offset(offset).limit(limit))
    items = result.scalars().unique().all()

    return {
        "items": [_query_to_response(q) for q in items],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


# ── Stats ────────────────────────────────────────────────────────


@router.get("/stats")
async def llm_query_stats(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    base_where = (LlmQuery.project_id == project_id) & (LlmQuery.tenant_id == tenant_id)

    total_q = await db.execute(select(func.count(LlmQuery.id)).where(base_where))
    total = total_q.scalar() or 0

    active_q = await db.execute(
        select(func.count(LlmQuery.id)).where(base_where & (LlmQuery.is_active == True))  # noqa: E712
    )
    active = active_q.scalar() or 0

    class_q = await db.execute(
        select(LlmQuery.query_class, func.count(LlmQuery.id)).where(base_where).group_by(LlmQuery.query_class)
    )
    by_class = {row[0] or "unclassified": row[1] for row in class_q.all()}

    subtype_q = await db.execute(
        select(LlmQuery.query_subtype, func.count(LlmQuery.id)).where(base_where).group_by(LlmQuery.query_subtype)
    )
    by_subtype = {row[0] or "unclassified": row[1] for row in subtype_q.all()}

    # Scenario counts from LlmQuery.category directly
    scenario_q = await db.execute(
        select(LlmQuery.category, func.count(LlmQuery.id))
        .where(base_where & LlmQuery.category.isnot(None) & (LlmQuery.category != ""))
        .group_by(LlmQuery.category)
    )
    by_scenario = {row[0]: row[1] for row in scenario_q.all()}

    return {
        "total": total,
        "active": active,
        "inactive": total - active,
        "by_class": by_class,
        "by_subtype": by_subtype,
        "by_scenario": by_scenario,
    }


# ── Bulk create ──────────────────────────────────────────────────


@router.post("/", response_model=LlmQueryBulkResponse, status_code=201)
async def add_llm_queries(
    project_id: int,
    body: LlmQueryBulkCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)
    await check_llm_query_limit(db, tenant_id, adding=len(body.queries))

    created = 0
    skipped = 0
    for q in body.queries:
        stmt = (
            pg_insert(LlmQuery)
            .values(
                tenant_id=tenant_id,
                project_id=project_id,
                query_text=q.query_text,
                query_type=q.query_type,
                target_brand=q.target_brand or project.brand_name,
                competitors=q.competitors,
                query_class=q.query_class,
                query_subtype=q.query_subtype,
                category=q.category,
            )
            .on_conflict_do_nothing(constraint="uq_project_llm_query")
        )
        result = await db.execute(stmt)
        if result.rowcount > 0:
            created += 1
        else:
            skipped += 1

    return LlmQueryBulkResponse(created=created, skipped=skipped)


# ── Single create ────────────────────────────────────────────────


@router.post("/single", response_model=LlmQueryResponse, status_code=201)
async def add_llm_query(
    project_id: int,
    body: LlmQueryCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)
    await check_llm_query_limit(db, tenant_id, adding=1)
    query = LlmQuery(
        tenant_id=tenant_id,
        project_id=project_id,
        query_text=body.query_text,
        query_type=body.query_type,
        target_brand=body.target_brand or project.brand_name,
        competitors=body.competitors,
        query_class=body.query_class,
        query_subtype=body.query_subtype,
        category=body.category,
    )
    db.add(query)
    await db.flush()

    result = await db.execute(select(LlmQuery).where(LlmQuery.id == query.id))
    query = result.scalar_one()
    return _query_to_response(query)


# ── Update ───────────────────────────────────────────────────────


@router.put("/{query_id}", response_model=LlmQueryResponse)
async def update_llm_query(
    project_id: int,
    query_id: int,
    body: LlmQueryUpdate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)
    result = await db.execute(
        select(LlmQuery).where(
            LlmQuery.id == query_id,
            LlmQuery.project_id == project_id,
            LlmQuery.tenant_id == tenant_id,
        )
    )
    query = result.scalar_one_or_none()
    if not query:
        raise NotFoundError("LLM query not found")

    update_data = body.model_dump(exclude_unset=True)

    # Remove tag_ids if present — tags system not active yet
    update_data.pop("tag_ids", None)

    for field, value in update_data.items():
        setattr(query, field, value)

    await db.flush()

    result = await db.execute(select(LlmQuery).where(LlmQuery.id == query_id))
    query = result.scalar_one()
    return _query_to_response(query)


# ── Delete ───────────────────────────────────────────────────────


@router.delete("/{query_id}", status_code=204)
async def delete_llm_query(
    project_id: int,
    query_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)
    result = await db.execute(
        select(LlmQuery).where(
            LlmQuery.id == query_id,
            LlmQuery.project_id == project_id,
            LlmQuery.tenant_id == tenant_id,
        )
    )
    query = result.scalar_one_or_none()
    if not query:
        raise NotFoundError("LLM query not found")

    await db.delete(query)


# ── Bulk delete ──────────────────────────────────────────────────


@router.post("/bulk-delete")
async def bulk_delete_llm_queries(
    project_id: int,
    body: LlmQueryBulkAction,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    stmt = delete(LlmQuery).where(
        LlmQuery.id.in_(body.ids),
        LlmQuery.project_id == project_id,
        LlmQuery.tenant_id == tenant_id,
    )
    result = await db.execute(stmt)
    return {"deleted": result.rowcount}


# ── Bulk toggle active ──────────────────────────────────────────


@router.post("/bulk-toggle")
async def bulk_toggle_llm_queries(
    project_id: int,
    body: LlmQueryBulkToggle,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    stmt = (
        update(LlmQuery)
        .where(
            LlmQuery.id.in_(body.ids),
            LlmQuery.project_id == project_id,
            LlmQuery.tenant_id == tenant_id,
        )
        .values(is_active=body.is_active)
    )
    result = await db.execute(stmt)
    return {"updated": result.rowcount, "is_active": body.is_active}
