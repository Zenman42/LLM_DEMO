from uuid import UUID

import jwt
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import ROLE_HIERARCHY, get_current_user
from app.core.exceptions import ConflictError, UnauthorizedError
from app.core.rate_limit import limiter
from app.core.security import create_access_token, create_refresh_token, decode_token, hash_password, verify_password
from app.db.postgres import get_db
from app.models.tenant import Tenant
from app.models.user import User
from app.models.user_project import UserProject
from app.schemas.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
@limiter.limit("5/minute")
async def register(request: Request, body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check unique email
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise ConflictError("Email already registered")

    # Check unique slug
    existing_tenant = await db.execute(select(Tenant).where(Tenant.slug == body.tenant_slug))
    if existing_tenant.scalar_one_or_none():
        raise ConflictError("Tenant slug already taken")

    # Create tenant
    tenant = Tenant(name=body.tenant_name, slug=body.tenant_slug)
    db.add(tenant)
    await db.flush()

    # Create owner user
    user = User(
        tenant_id=tenant.id,
        email=body.email,
        password_hash=hash_password(body.password),
        role="owner",
    )
    db.add(user)
    await db.flush()

    return TokenResponse(
        access_token=create_access_token(user.id, tenant.id),
        refresh_token=create_refresh_token(user.id, tenant.id),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(request: Request, body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email, User.is_active == True))  # noqa: E712
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        raise UnauthorizedError("Invalid email or password")

    return TokenResponse(
        access_token=create_access_token(user.id, user.tenant_id),
        refresh_token=create_refresh_token(user.id, user.tenant_id),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_token(body.refresh_token)
    except jwt.PyJWTError:
        raise UnauthorizedError("Invalid or expired refresh token")

    if payload.get("type") != "refresh":
        raise UnauthorizedError("Invalid token type")

    user_id = UUID(payload["sub"])
    tenant_id = UUID(payload["tid"])

    # Verify user still exists and active
    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))  # noqa: E712
    user = result.scalar_one_or_none()
    if not user:
        raise UnauthorizedError("User not found or inactive")

    return TokenResponse(
        access_token=create_access_token(user.id, tenant_id),
        refresh_token=create_refresh_token(user.id, tenant_id),
    )


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
