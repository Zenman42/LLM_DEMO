"""User management API — invite users, change roles, deactivate, manage project access."""

import secrets
import uuid
from collections import defaultdict

from fastapi import APIRouter, Depends
from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id, get_current_user, require_role
from app.core.exceptions import BadRequestError, ConflictError, NotFoundError
from app.core.security import hash_password
from app.db.postgres import get_db
from app.models.project import Project
from app.models.user import User
from app.models.user_project import UserProject
from app.schemas.common import MessageResponse
from app.schemas.user import (
    InviteUserRequest,
    InviteUserResponse,
    UpdateRoleRequest,
    UserListPaginated,
    UserListResponse,
)
from app.schemas.user_project import UserProjectAssign, UserProjectItem

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/", response_model=UserListPaginated, dependencies=[Depends(require_role("admin"))])
async def list_users(
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """List all users in the tenant. Requires admin role."""
    count_result = await db.execute(select(func.count()).select_from(User).where(User.tenant_id == tenant_id))
    total = count_result.scalar() or 0

    result = await db.execute(select(User).where(User.tenant_id == tenant_id).order_by(User.created_at.asc()))
    users = result.scalars().all()

    # Fetch all project assignments for the tenant
    up_result = await db.execute(select(UserProject).where(UserProject.tenant_id == tenant_id))
    assignments = up_result.scalars().all()
    user_projects_map: dict[uuid.UUID, list[int]] = defaultdict(list)
    for up in assignments:
        user_projects_map[up.user_id].append(up.project_id)

    return UserListPaginated(
        items=[
            UserListResponse(
                id=u.id,
                email=u.email,
                role=u.role,
                is_active=u.is_active,
                created_at=u.created_at,
                project_ids=user_projects_map.get(u.id, []),
            )
            for u in users
        ],
        total=total,
    )


@router.post(
    "/invite", response_model=InviteUserResponse, status_code=201, dependencies=[Depends(require_role("admin"))]
)
async def invite_user(
    body: InviteUserRequest,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Invite a new user to the tenant. Requires admin role."""
    # Check duplicate email
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise ConflictError("Email already registered")

    # Generate temporary password
    temp_password = secrets.token_urlsafe(16)

    user = User(
        tenant_id=tenant_id,
        email=body.email,
        password_hash=hash_password(temp_password),
        role=body.role,
    )
    db.add(user)
    await db.flush()

    # Assign project access if project_ids provided
    if body.project_ids:
        # Validate project_ids belong to tenant
        proj_result = await db.execute(
            select(Project.id).where(Project.tenant_id == tenant_id, Project.id.in_(body.project_ids))
        )
        valid_ids = {row[0] for row in proj_result.all()}
        for pid in body.project_ids:
            if pid not in valid_ids:
                raise BadRequestError(f"Project {pid} not found in this tenant")
            db.add(UserProject(user_id=user.id, project_id=pid, tenant_id=tenant_id))
        await db.flush()

    return InviteUserResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        temporary_password=temp_password,
    )


@router.patch(
    "/{user_id}/role",
    response_model=UserListResponse,
    dependencies=[Depends(require_role("owner"))],
)
async def update_user_role(
    user_id: uuid.UUID,
    body: UpdateRoleRequest,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change a user's role. Requires owner role."""
    if current_user.id == user_id:
        raise BadRequestError("Cannot change your own role")

    result = await db.execute(select(User).where(User.id == user_id, User.tenant_id == tenant_id))
    user = result.scalar_one_or_none()
    if not user:
        raise NotFoundError("User not found")

    if user.role == "owner":
        raise BadRequestError("Cannot change the role of the owner")

    user.role = body.role
    await db.flush()

    # Return with project_ids
    up_result = await db.execute(select(UserProject.project_id).where(UserProject.user_id == user.id))
    project_ids = [row[0] for row in up_result.all()]

    return UserListResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        project_ids=project_ids,
    )


@router.delete(
    "/{user_id}",
    response_model=MessageResponse,
    dependencies=[Depends(require_role("admin"))],
)
async def deactivate_user(
    user_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate a user (soft delete). Requires admin role."""
    if current_user.id == user_id:
        raise BadRequestError("Cannot deactivate yourself")

    result = await db.execute(select(User).where(User.id == user_id, User.tenant_id == tenant_id))
    user = result.scalar_one_or_none()
    if not user:
        raise NotFoundError("User not found")

    if user.role == "owner":
        raise BadRequestError("Cannot deactivate the owner")

    user.is_active = False
    await db.flush()
    return MessageResponse(message=f"User {user.email} deactivated")


# ---------- Project access management ----------


@router.get(
    "/{user_id}/projects",
    response_model=list[UserProjectItem],
    dependencies=[Depends(require_role("admin"))],
)
async def list_user_projects(
    user_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """List projects assigned to a user. Requires admin role."""
    # Verify user belongs to tenant
    user_result = await db.execute(select(User).where(User.id == user_id, User.tenant_id == tenant_id))
    if not user_result.scalar_one_or_none():
        raise NotFoundError("User not found")

    result = await db.execute(
        select(UserProject.project_id, Project.name)
        .join(Project, Project.id == UserProject.project_id)
        .where(UserProject.user_id == user_id, UserProject.tenant_id == tenant_id)
        .order_by(Project.name)
    )
    return [UserProjectItem(project_id=row[0], project_name=row[1]) for row in result.all()]


@router.put(
    "/{user_id}/projects",
    response_model=list[UserProjectItem],
    dependencies=[Depends(require_role("admin"))],
)
async def set_user_projects(
    user_id: uuid.UUID,
    body: UserProjectAssign,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Replace all project assignments for a user. Requires admin role."""
    # Verify user belongs to tenant
    user_result = await db.execute(select(User).where(User.id == user_id, User.tenant_id == tenant_id))
    if not user_result.scalar_one_or_none():
        raise NotFoundError("User not found")

    # Validate all project_ids belong to tenant
    if body.project_ids:
        proj_result = await db.execute(
            select(Project.id).where(Project.tenant_id == tenant_id, Project.id.in_(body.project_ids))
        )
        valid_ids = {row[0] for row in proj_result.all()}
        for pid in body.project_ids:
            if pid not in valid_ids:
                raise BadRequestError(f"Project {pid} not found in this tenant")

    # Delete all existing assignments
    await db.execute(
        delete(UserProject).where(
            and_(UserProject.user_id == user_id, UserProject.tenant_id == tenant_id)
        )
    )

    # Insert new assignments
    for pid in body.project_ids:
        db.add(UserProject(user_id=user_id, project_id=pid, tenant_id=tenant_id))

    await db.flush()

    # Return updated list
    result = await db.execute(
        select(UserProject.project_id, Project.name)
        .join(Project, Project.id == UserProject.project_id)
        .where(UserProject.user_id == user_id, UserProject.tenant_id == tenant_id)
        .order_by(Project.name)
    )
    return [UserProjectItem(project_id=row[0], project_name=row[1]) for row in result.all()]
