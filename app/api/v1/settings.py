"""Settings API — manage tenant credentials and collection schedule."""

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id, require_role
from app.core.encryption import encrypt_value
from app.db.postgres import get_db
from app.models.tenant import Tenant

router = APIRouter(prefix="/settings", tags=["settings"])


# --- Schemas ---


class CredentialsResponse(BaseModel):
    justmagic_api_key: bool  # True if configured (never returns actual value)
    ywm_token: bool
    ywm_host_id_set: bool
    gsc_credentials: bool
    telegram_bot_token: bool
    telegram_chat_id: bool
    # LLM API keys
    openai_api_key: bool = False
    deepseek_api_key: bool = False
    perplexity_api_key: bool = False
    yandexgpt_api_key: bool = False
    yandexgpt_folder_id: bool = False
    gemini_api_key: bool = False
    gigachat_api_key: bool = False


class CredentialsUpdate(BaseModel):
    justmagic_api_key: str | None = None
    ywm_token: str | None = None
    ywm_host_id: str | None = None
    gsc_credentials_json: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    # LLM API keys
    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    perplexity_api_key: str | None = None
    yandexgpt_api_key: str | None = None
    yandexgpt_folder_id: str | None = None
    gemini_api_key: str | None = None
    gigachat_api_key: str | None = None


class ScheduleResponse(BaseModel):
    collection_hour: int
    collection_minute: int


class ScheduleUpdate(BaseModel):
    collection_hour: int | None = None
    collection_minute: int | None = None


# --- Endpoints ---


@router.get("/credentials", response_model=CredentialsResponse, dependencies=[Depends(require_role("admin"))])
async def get_credentials(
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Show which credentials are configured (masked — never returns actual values)."""
    tenant = await db.get(Tenant, tenant_id)
    return CredentialsResponse(
        justmagic_api_key=bool(tenant.justmagic_api_key),
        ywm_token=bool(tenant.ywm_token),
        ywm_host_id_set=bool(tenant.ywm_host_id),
        gsc_credentials=bool(tenant.gsc_credentials_json),
        telegram_bot_token=bool(tenant.telegram_bot_token),
        telegram_chat_id=bool(tenant.telegram_chat_id),
        openai_api_key=bool(tenant.openai_api_key),
        deepseek_api_key=bool(tenant.deepseek_api_key),
        perplexity_api_key=bool(tenant.perplexity_api_key),
        yandexgpt_api_key=bool(tenant.yandexgpt_api_key),
        yandexgpt_folder_id=bool(tenant.yandexgpt_folder_id),
        gemini_api_key=bool(tenant.gemini_api_key),
        gigachat_api_key=bool(tenant.gigachat_api_key),
    )


@router.put("/credentials", response_model=CredentialsResponse, dependencies=[Depends(require_role("admin"))])
async def update_credentials(
    payload: CredentialsUpdate,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Update tenant API credentials. Only provided fields are updated. Requires admin role."""
    tenant = await db.get(Tenant, tenant_id)

    if payload.justmagic_api_key is not None:
        tenant.justmagic_api_key = encrypt_value(payload.justmagic_api_key) if payload.justmagic_api_key else None
    if payload.ywm_token is not None:
        tenant.ywm_token = encrypt_value(payload.ywm_token) if payload.ywm_token else None
    if payload.ywm_host_id is not None:
        tenant.ywm_host_id = encrypt_value(payload.ywm_host_id) if payload.ywm_host_id else None
    if payload.gsc_credentials_json is not None:
        tenant.gsc_credentials_json = (
            encrypt_value(payload.gsc_credentials_json) if payload.gsc_credentials_json else None
        )
    if payload.telegram_bot_token is not None:
        tenant.telegram_bot_token = encrypt_value(payload.telegram_bot_token) if payload.telegram_bot_token else None
    if payload.telegram_chat_id is not None:
        tenant.telegram_chat_id = encrypt_value(payload.telegram_chat_id) if payload.telegram_chat_id else None
    # LLM API keys
    if payload.openai_api_key is not None:
        tenant.openai_api_key = encrypt_value(payload.openai_api_key) if payload.openai_api_key else None
    if payload.deepseek_api_key is not None:
        tenant.deepseek_api_key = encrypt_value(payload.deepseek_api_key) if payload.deepseek_api_key else None
    if payload.perplexity_api_key is not None:
        tenant.perplexity_api_key = encrypt_value(payload.perplexity_api_key) if payload.perplexity_api_key else None
    if payload.yandexgpt_api_key is not None:
        tenant.yandexgpt_api_key = encrypt_value(payload.yandexgpt_api_key) if payload.yandexgpt_api_key else None
    if payload.yandexgpt_folder_id is not None:
        tenant.yandexgpt_folder_id = encrypt_value(payload.yandexgpt_folder_id) if payload.yandexgpt_folder_id else None
    if payload.gemini_api_key is not None:
        tenant.gemini_api_key = encrypt_value(payload.gemini_api_key) if payload.gemini_api_key else None
    if payload.gigachat_api_key is not None:
        tenant.gigachat_api_key = encrypt_value(payload.gigachat_api_key) if payload.gigachat_api_key else None

    await db.commit()
    await db.refresh(tenant)

    return CredentialsResponse(
        justmagic_api_key=bool(tenant.justmagic_api_key),
        ywm_token=bool(tenant.ywm_token),
        ywm_host_id_set=bool(tenant.ywm_host_id),
        gsc_credentials=bool(tenant.gsc_credentials_json),
        telegram_bot_token=bool(tenant.telegram_bot_token),
        telegram_chat_id=bool(tenant.telegram_chat_id),
        openai_api_key=bool(tenant.openai_api_key),
        deepseek_api_key=bool(tenant.deepseek_api_key),
        perplexity_api_key=bool(tenant.perplexity_api_key),
        yandexgpt_api_key=bool(tenant.yandexgpt_api_key),
        yandexgpt_folder_id=bool(tenant.yandexgpt_folder_id),
        gemini_api_key=bool(tenant.gemini_api_key),
        gigachat_api_key=bool(tenant.gigachat_api_key),
    )


@router.get("/schedule", response_model=ScheduleResponse)
async def get_schedule(
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Get collection schedule."""
    tenant = await db.get(Tenant, tenant_id)
    return ScheduleResponse(
        collection_hour=tenant.collection_hour,
        collection_minute=tenant.collection_minute,
    )


@router.put("/schedule", response_model=ScheduleResponse, dependencies=[Depends(require_role("admin"))])
async def update_schedule(
    payload: ScheduleUpdate,
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Update collection schedule. Requires admin role."""
    tenant = await db.get(Tenant, tenant_id)

    if payload.collection_hour is not None:
        if not 0 <= payload.collection_hour <= 23:
            from app.core.exceptions import BadRequestError

            raise BadRequestError("collection_hour must be 0-23")
        tenant.collection_hour = payload.collection_hour

    if payload.collection_minute is not None:
        if not 0 <= payload.collection_minute <= 59:
            from app.core.exceptions import BadRequestError

            raise BadRequestError("collection_minute must be 0-59")
        tenant.collection_minute = payload.collection_minute

    await db.commit()
    await db.refresh(tenant)

    return ScheduleResponse(
        collection_hour=tenant.collection_hour,
        collection_minute=tenant.collection_minute,
    )
