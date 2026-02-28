from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import ROLE_HIERARCHY, get_current_user
from app.db.postgres import get_db
from app.models.user import User
from app.models.user_project import UserProject
from app.schemas.auth import UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # Admin/owner have access to all projects — return empty list (frontend treats as "all")
    if ROLE_HIERARCHY.get(user.role, 0) >= ROLE_HIERARCHY["admin"]:
        project_ids: list[int] = []
    else:
        result = await db.execute(
            select(UserProject.project_id).where(UserProject.user_id == user.id)
        )
        project_ids = [row[0] for row in result.all()]

    return UserResponse(
        id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        theme=user.theme,
        project_ids=project_ids,
    )
