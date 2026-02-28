"""Account endpoints — profile, password change."""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.core.exceptions import BadRequestError, ConflictError
from app.core.security import hash_password, verify_password
from app.db.postgres import get_db
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.account import AccountResponse, ChangePasswordRequest, UpdateProfileRequest, UpdateThemeRequest
from app.schemas.common import MessageResponse

router = APIRouter(prefix="/account", tags=["account"])


async def _build_account_response(user: User, db: AsyncSession) -> AccountResponse:
    result = await db.execute(select(Tenant).where(Tenant.id == user.tenant_id))
    tenant = result.scalar_one()
    return AccountResponse(
        id=str(user.id),
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        theme=user.theme,
        tenant_name=tenant.name,
        tenant_slug=tenant.slug,
    )


@router.get("/me", response_model=AccountResponse)
async def get_account(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current user profile with tenant info."""
    return await _build_account_response(user, db)


@router.patch("/me", response_model=AccountResponse)
async def update_profile(
    body: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update profile (email)."""
    if body.email and body.email != user.email:
        existing = await db.execute(select(User).where(User.email == body.email))
        if existing.scalar_one_or_none():
            raise ConflictError("Email already in use")
        user.email = body.email
        await db.flush()

    return await _build_account_response(user, db)


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    body: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change password. Requires current password."""
    if not verify_password(body.current_password, user.password_hash):
        raise BadRequestError("Current password is incorrect")

    if body.current_password == body.new_password:
        raise BadRequestError("New password must differ from current")

    user.password_hash = hash_password(body.new_password)
    await db.flush()
    return MessageResponse(message="Password changed successfully")


@router.patch("/theme", response_model=MessageResponse)
async def update_theme(
    body: UpdateThemeRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update user UI theme (light/dark)."""
    user.theme = body.theme
    await db.flush()
    return MessageResponse(message="Theme updated")
