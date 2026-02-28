"""API Key management endpoints — create, list, delete API keys."""

import hashlib
import secrets
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id, get_current_user, require_role
from app.core.exceptions import NotFoundError
from app.db.postgres import get_db
from app.models.api_key import ApiKey
from app.models.user import User
from app.schemas.api_key import ApiKeyCreateRequest, ApiKeyCreateResponse, ApiKeyListItem, ApiKeyListResponse
from app.schemas.common import MessageResponse

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


def _generate_api_key() -> tuple[str, str]:
    """Generate a raw API key and its SHA-256 hash.

    Returns:
        (raw_key, key_hash) — raw_key is shown once, key_hash is stored.
    """
    raw_key = f"pt_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, key_hash


@router.get("/", response_model=ApiKeyListResponse)
async def list_api_keys(
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """List all API keys for the tenant (masked). Requires authentication."""
    count_result = await db.execute(select(func.count()).select_from(ApiKey).where(ApiKey.tenant_id == tenant_id))
    total = count_result.scalar() or 0

    result = await db.execute(select(ApiKey).where(ApiKey.tenant_id == tenant_id).order_by(ApiKey.created_at.desc()))
    keys = result.scalars().all()

    return ApiKeyListResponse(
        items=[
            ApiKeyListItem(
                id=k.id,
                name=k.name,
                key_prefix=k.key_hash[:8] + "...",
                last_used_at=k.last_used_at,
                created_at=k.created_at,
            )
            for k in keys
        ],
        total=total,
    )


@router.post(
    "/",
    response_model=ApiKeyCreateResponse,
    status_code=201,
    dependencies=[Depends(require_role("admin"))],
)
async def create_api_key(
    body: ApiKeyCreateRequest,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key. The full key is returned ONLY at creation time. Requires admin role."""
    raw_key, key_hash = _generate_api_key()

    api_key = ApiKey(
        tenant_id=tenant_id,
        user_id=current_user.id,
        key_hash=key_hash,
        name=body.name,
    )
    db.add(api_key)
    await db.flush()

    return ApiKeyCreateResponse(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,
        created_at=api_key.created_at,
    )


@router.delete(
    "/{key_id}",
    response_model=MessageResponse,
    dependencies=[Depends(require_role("admin"))],
)
async def delete_api_key(
    key_id: uuid.UUID,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Delete an API key. Requires admin role."""
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.tenant_id == tenant_id))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise NotFoundError("API key not found")

    await db.delete(api_key)
    return MessageResponse(message="API key deleted")
