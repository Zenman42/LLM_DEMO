from collections.abc import Callable
from uuid import UUID

import jwt
from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.db.postgres import get_db
from app.models.user import User
from app.models.user_project import UserProject

# Role hierarchy: higher index = more privileges
ROLE_HIERARCHY: dict[str, int] = {
    "viewer": 0,
    "member": 1,
    "admin": 2,
    "owner": 3,
}


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    authorization: str = Header(..., description="Bearer <token>"),
) -> User:
    if not authorization.startswith("Bearer "):
        raise UnauthorizedError("Invalid authorization header")

    token = authorization[7:]
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        raise UnauthorizedError("Invalid or expired token")

    if payload.get("type") != "access":
        raise UnauthorizedError("Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedError("Invalid token payload")

    result = await db.execute(select(User).where(User.id == UUID(user_id), User.is_active == True))  # noqa: E712
    user = result.scalar_one_or_none()
    if not user:
        raise UnauthorizedError("User not found or inactive")

    return user


async def get_current_tenant_id(user: User = Depends(get_current_user)) -> UUID:
    return user.tenant_id


async def check_project_access(project_id: int, user: User, db: AsyncSession) -> None:
    """Verify user has access to the given project.

    Admin/owner roles bypass the check (they access all tenant projects).
    Viewer/member roles must have an explicit UserProject assignment.
    """
    if ROLE_HIERARCHY.get(user.role, 0) >= ROLE_HIERARCHY["admin"]:
        return
    result = await db.execute(
        select(UserProject.id).where(
            UserProject.user_id == user.id,
            UserProject.project_id == project_id,
            UserProject.tenant_id == user.tenant_id,
        )
    )
    if not result.scalar_one_or_none():
        raise ForbiddenError("You do not have access to this project")


async def get_demo_user(db: AsyncSession = Depends(get_db)) -> User:
    """Return demo user without checking JWT. Used as dependency override."""
    result = await db.execute(
        select(User).where(User.email == settings.demo_user_email, User.is_active == True)  # noqa: E712
    )
    user = result.scalar_one_or_none()
    if not user:
        raise UnauthorizedError(f"Demo user not found: {settings.demo_user_email}")
    return user


def require_role(min_role: str) -> Callable:
    """Dependency factory: require user to have at least `min_role` privileges.

    Usage:
        @router.put("/settings", dependencies=[Depends(require_role("admin"))])
    """
    min_level = ROLE_HIERARCHY.get(min_role, 0)

    async def _check(user: User = Depends(get_current_user)) -> User:
        user_level = ROLE_HIERARCHY.get(user.role, 0)
        if user_level < min_level:
            raise ForbiddenError(f"Requires at least '{min_role}' role")
        return user

    return _check
