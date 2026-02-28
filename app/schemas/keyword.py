from datetime import datetime

from pydantic import BaseModel, Field


class KeywordCreate(BaseModel):
    keyword: str = Field(min_length=1, max_length=500)
    target_url: str | None = Field(None, max_length=2048)


class KeywordBulkCreate(BaseModel):
    keywords: list[KeywordCreate] = Field(min_length=1, max_length=1000)


class KeywordUpdate(BaseModel):
    target_url: str | None = Field(None, max_length=2048)
    is_active: bool | None = None


class KeywordResponse(BaseModel):
    id: int
    project_id: int
    keyword: str
    target_url: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class KeywordBulkResponse(BaseModel):
    created: int
    skipped: int
