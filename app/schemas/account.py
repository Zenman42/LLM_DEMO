"""Account management schemas."""

from pydantic import BaseModel, EmailStr, Field


class AccountResponse(BaseModel):
    """Extended user profile including tenant info."""

    id: str
    email: str
    role: str
    is_active: bool
    theme: str = "dark"
    tenant_name: str
    tenant_slug: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=128)


class UpdateProfileRequest(BaseModel):
    """Only email update for now."""

    email: EmailStr | None = None


class UpdateThemeRequest(BaseModel):
    theme: str = Field(..., pattern=r"^(light|dark)$")
