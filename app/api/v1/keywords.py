from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id
from app.core.exceptions import NotFoundError
from app.core.plan_limits import check_keyword_limit
from app.db.postgres import get_db
from app.models.keyword import Keyword
from app.models.project import Project
from app.schemas.keyword import KeywordBulkCreate, KeywordBulkResponse, KeywordCreate, KeywordResponse, KeywordUpdate

router = APIRouter(prefix="/projects/{project_id}/keywords", tags=["keywords"])


async def _get_project(project_id: int, tenant_id: UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    return project


@router.get("/")
async def list_keywords(
    project_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    base_where = (Keyword.project_id == project_id) & (Keyword.tenant_id == tenant_id)

    # Total count
    count_result = await db.execute(select(func.count(Keyword.id)).where(base_where))
    total = count_result.scalar() or 0

    # Paginated results
    result = await db.execute(select(Keyword).where(base_where).order_by(Keyword.keyword).offset(offset).limit(limit))
    items = result.scalars().all()

    return {
        "items": [KeywordResponse.model_validate(k) for k in items],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.post("/", response_model=KeywordBulkResponse, status_code=201)
async def add_keywords(
    project_id: int,
    body: KeywordBulkCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)
    await check_keyword_limit(db, tenant_id, adding=len(body.keywords))

    created = 0
    skipped = 0
    for kw in body.keywords:
        stmt = (
            pg_insert(Keyword)
            .values(
                tenant_id=tenant_id,
                project_id=project_id,
                keyword=kw.keyword,
                target_url=kw.target_url,
            )
            .on_conflict_do_nothing(constraint="uq_project_keyword")
        )
        result = await db.execute(stmt)
        if result.rowcount > 0:
            created += 1
        else:
            skipped += 1

    return KeywordBulkResponse(created=created, skipped=skipped)


@router.post("/single", response_model=KeywordResponse, status_code=201)
async def add_keyword(
    project_id: int,
    body: KeywordCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)
    await check_keyword_limit(db, tenant_id, adding=1)
    keyword = Keyword(
        tenant_id=tenant_id,
        project_id=project_id,
        keyword=body.keyword,
        target_url=body.target_url,
    )
    db.add(keyword)
    await db.flush()
    return keyword


@router.put("/{keyword_id}", response_model=KeywordResponse)
async def update_keyword(
    project_id: int,
    keyword_id: int,
    body: KeywordUpdate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)
    result = await db.execute(
        select(Keyword).where(
            Keyword.id == keyword_id,
            Keyword.project_id == project_id,
            Keyword.tenant_id == tenant_id,
        )
    )
    keyword = result.scalar_one_or_none()
    if not keyword:
        raise NotFoundError("Keyword not found")

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(keyword, field, value)

    await db.flush()
    return keyword


@router.delete("/{keyword_id}", status_code=204)
async def delete_keyword(
    project_id: int,
    keyword_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)
    result = await db.execute(
        select(Keyword).where(
            Keyword.id == keyword_id,
            Keyword.project_id == project_id,
            Keyword.tenant_id == tenant_id,
        )
    )
    keyword = result.scalar_one_or_none()
    if not keyword:
        raise NotFoundError("Keyword not found")

    await db.delete(keyword)
