"""User management schemas."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserListResponse(BaseModel):
    id: UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime
    project_ids: list[int] = []

    model_config = {"from_attributes": True}


class InviteUserRequest(BaseModel):
    email: EmailStr
    role: str = Field(default="member", pattern="^(viewer|member|admin)$")
    project_ids: list[int] = []


class InviteUserResponse(BaseModel):
    id: UUID
    email: str
    role: str
    temporary_password: str


class UpdateRoleRequest(BaseModel):
    role: str = Field(..., pattern="^(viewer|member|admin)$")


class UserListPaginated(BaseModel):
    items: list[UserListResponse]
    total: int
