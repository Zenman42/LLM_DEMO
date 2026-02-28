from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import (
    ROLE_HIERARCHY,
    check_project_access,
    get_current_tenant_id,
    get_current_user,
    require_role,
)
from app.core.exceptions import NotFoundError
from app.core.plan_limits import check_project_limit
from app.db.postgres import get_db
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.query_tag import QueryTag
from app.models.serp import SerpSnapshot
from app.models.user import User
from app.models.user_project import UserProject
from app.schemas.project import ProjectCreate, ProjectResponse, ProjectUpdate, ProjectWithStats

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("/", response_model=list[ProjectWithStats])
async def list_projects(
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    # Single query: projects + keyword count + last collected
    stmt = (
        select(
            Project,
            func.count(Keyword.id).label("keyword_count"),
            func.max(SerpSnapshot.collected_at).label("last_collected"),
        )
        .outerjoin(Keyword, (Keyword.project_id == Project.id) & (Keyword.is_active == True))  # noqa: E712
        .outerjoin(SerpSnapshot, SerpSnapshot.keyword_id == Keyword.id)
        .where(Project.tenant_id == tenant_id)
    )

    # Non-admin users: filter to assigned projects only
    if ROLE_HIERARCHY.get(user.role, 0) < ROLE_HIERARCHY["admin"]:
        stmt = stmt.join(
            UserProject,
            and_(UserProject.project_id == Project.id, UserProject.user_id == user.id),
        )

    stmt = stmt.group_by(Project.id).order_by(Project.created_at.desc())
    result = await db.execute(stmt)
    rows = result.all()

    return [
        ProjectWithStats(
            **ProjectResponse.model_validate(row.Project).model_dump(),
            keyword_count=row.keyword_count,
            last_collected=row.last_collected,
        )
        for row in rows
    ]


@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(
    body: ProjectCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await check_project_limit(db, tenant_id)

    project = Project(
        tenant_id=tenant_id,
        name=body.name,
        domain=body.domain,
        search_engine=body.search_engine,
        region_google=body.region_google,
        region_yandex=body.region_yandex,
        ywm_host_id=body.ywm_host_id,
        gsc_site_url=body.gsc_site_url,
        track_llm=body.track_llm,
        track_ai_overview=body.track_ai_overview,
        llm_providers=body.llm_providers or [],
        brand_name=body.brand_name,
        brand_aliases=body.brand_aliases or [],
        competitors=body.competitors or [],
        categories=body.categories or [],
        golden_facts=body.golden_facts or [],
        personas=body.personas or [],
        market=body.market,
        geo=body.geo,
    )
    db.add(project)
    await db.flush()

    # Auto-create scenario tags from categories list
    for cat in body.categories or []:
        tag_name = cat if isinstance(cat, str) else (cat.get("name") if isinstance(cat, dict) else str(cat))
        if tag_name:
            db.add(
                QueryTag(
                    project_id=project.id,
                    tenant_id=tenant_id,
                    tag_name=tag_name,
                    tag_type="scenario",
                )
            )
    await db.flush()

    return project


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    await check_project_access(project_id, user, db)
    return project


@router.put("/{project_id}", response_model=ProjectResponse, dependencies=[Depends(require_role("member"))])
async def update_project(
    project_id: int,
    body: ProjectUpdate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    await check_project_access(project_id, user, db)

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)

    await db.flush()
    return project


@router.delete("/{project_id}", status_code=204, dependencies=[Depends(require_role("admin"))])
async def delete_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    await check_project_access(project_id, user, db)

    await db.delete(project)
