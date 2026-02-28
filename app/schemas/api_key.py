"""API key management schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class ApiKeyCreateResponse(BaseModel):
    id: UUID
    name: str
    key: str  # full key — shown ONLY at creation time
    created_at: datetime

    model_config = {"from_attributes": True}


class ApiKeyListItem(BaseModel):
    id: UUID
    name: str | None
    key_prefix: str  # first 8 chars + "..."
    last_used_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ApiKeyListResponse(BaseModel):
    items: list[ApiKeyListItem]
    total: int
