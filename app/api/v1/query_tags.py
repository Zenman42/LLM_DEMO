"""Query Tags API — manage scenario and class tags for LLM queries."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id
from app.core.exceptions import NotFoundError
from app.db.postgres import get_db
from app.models.project import Project
from app.models.query_tag import QueryTag, QueryTagLink
from app.schemas.query_tag import (
    QueryTagCreate,
    QueryTagResponse,
    QueryTagUpdate,
    TagAssignment,
)

router = APIRouter(prefix="/projects/{project_id}/query-tags", tags=["query-tags"])


async def _get_project(project_id: int, tenant_id: UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    return project


# ── List tags ──────────────────────────────────────────────────────


@router.get("/")
async def list_query_tags(
    project_id: int,
    tag_type: str | None = Query(None, pattern=r"^(class|scenario)$"),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    # Build query with query_count subquery
    count_subq = (
        select(func.count(QueryTagLink.id))
        .where(QueryTagLink.tag_id == QueryTag.id)
        .correlate(QueryTag)
        .scalar_subquery()
    )

    stmt = (
        select(QueryTag, count_subq.label("query_count"))
        .where(QueryTag.project_id == project_id, QueryTag.tenant_id == tenant_id)
        .order_by(QueryTag.tag_type, QueryTag.tag_name)
    )

    if tag_type:
        stmt = stmt.where(QueryTag.tag_type == tag_type)

    result = await db.execute(stmt)
    rows = result.all()

    items = []
    for tag, count in rows:
        resp = QueryTagResponse.model_validate(tag)
        resp.query_count = count or 0
        items.append(resp)

    return {"items": items}


# ── Create tag ─────────────────────────────────────────────────────


@router.post("/", status_code=201)
async def create_query_tag(
    project_id: int,
    body: QueryTagCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    # Check for existing tag
    existing = await db.execute(
        select(QueryTag).where(
            QueryTag.project_id == project_id,
            QueryTag.tenant_id == tenant_id,
            QueryTag.tag_name == body.tag_name,
            QueryTag.tag_type == body.tag_type,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Tag already exists")

    tag = QueryTag(
        project_id=project_id,
        tenant_id=tenant_id,
        tag_name=body.tag_name,
        tag_type=body.tag_type,
        description=body.description,
    )
    db.add(tag)
    await db.flush()

    return QueryTagResponse.model_validate(tag)


# ── Update tag ─────────────────────────────────────────────────────


@router.put("/{tag_id}")
async def update_query_tag(
    project_id: int,
    tag_id: int,
    body: QueryTagUpdate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    result = await db.execute(
        select(QueryTag).where(
            QueryTag.id == tag_id,
            QueryTag.project_id == project_id,
            QueryTag.tenant_id == tenant_id,
        )
    )
    tag = result.scalar_one_or_none()
    if not tag:
        raise NotFoundError("Tag not found")

    update_data = body.model_dump(exclude_unset=True)

    # Check uniqueness if tag_name is being changed
    new_name = update_data.get("tag_name")
    if new_name and new_name != tag.tag_name:
        conflict = await db.execute(
            select(QueryTag.id).where(
                QueryTag.project_id == project_id,
                QueryTag.tag_name == new_name,
                QueryTag.tag_type == tag.tag_type,
                QueryTag.id != tag_id,
            )
        )
        if conflict.scalar_one_or_none():
            raise HTTPException(status_code=409, detail=f"Tag '{new_name}' of type '{tag.tag_type}' already exists")

    for field, value in update_data.items():
        setattr(tag, field, value)

    await db.flush()
    return QueryTagResponse.model_validate(tag)


# ── Delete tag ─────────────────────────────────────────────────────


@router.delete("/{tag_id}", status_code=204)
async def delete_query_tag(
    project_id: int,
    tag_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    result = await db.execute(
        select(QueryTag).where(
            QueryTag.id == tag_id,
            QueryTag.project_id == project_id,
            QueryTag.tenant_id == tenant_id,
        )
    )
    tag = result.scalar_one_or_none()
    if not tag:
        raise NotFoundError("Tag not found")

    await db.delete(tag)


# ── Bulk assign tags ───────────────────────────────────────────────


@router.post("/assign")
async def assign_tags(
    project_id: int,
    body: TagAssignment,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    # Validate that all tag_ids belong to this project
    valid_tags = await db.execute(
        select(QueryTag.id).where(
            QueryTag.id.in_(body.tag_ids),
            QueryTag.project_id == project_id,
        )
    )
    valid_tag_ids = set(valid_tags.scalars().all())
    invalid = set(body.tag_ids) - valid_tag_ids
    if invalid:
        raise HTTPException(status_code=400, detail=f"Tag IDs not found in this project: {sorted(invalid)}")

    created = 0
    for query_id in body.query_ids:
        for tag_id in body.tag_ids:
            stmt = (
                pg_insert(QueryTagLink)
                .values(query_id=query_id, tag_id=tag_id)
                .on_conflict_do_nothing(constraint="uq_query_tag_link")
            )
            result = await db.execute(stmt)
            created += result.rowcount

    return {"assigned": created}


# ── Bulk unassign tags ─────────────────────────────────────────────


@router.post("/unassign")
async def unassign_tags(
    project_id: int,
    body: TagAssignment,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    stmt = delete(QueryTagLink).where(
        QueryTagLink.query_id.in_(body.query_ids),
        QueryTagLink.tag_id.in_(body.tag_ids),
    )
    result = await db.execute(stmt)
    return {"removed": result.rowcount}
