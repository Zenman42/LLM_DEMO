from datetime import datetime

from pydantic import BaseModel, Field


class LlmQueryCreate(BaseModel):
    query_text: str = Field(min_length=1, max_length=2000)
    query_type: str = Field("custom", pattern=r"^(brand_check|comparison|recommendation|custom)$")
    target_brand: str | None = Field(None, max_length=255)
    competitors: list[str] | None = None
    query_class: str | None = None
    query_subtype: str | None = None
    category: str | None = None
    tag_ids: list[int] | None = None


class LlmQueryBulkCreate(BaseModel):
    queries: list[LlmQueryCreate] = Field(min_length=1, max_length=500)


class LlmQueryUpdate(BaseModel):
    query_text: str | None = Field(None, min_length=1, max_length=2000)
    query_type: str | None = Field(None, pattern=r"^(brand_check|comparison|recommendation|custom)$")
    target_brand: str | None = None
    competitors: list[str] | None = None
    is_active: bool | None = None
    query_class: str | None = None
    query_subtype: str | None = None
    category: str | None = None
    tag_ids: list[int] | None = None


class LlmQueryResponse(BaseModel):
    id: int
    project_id: int
    query_text: str
    query_type: str
    target_brand: str | None
    competitors: list[str] | None
    query_class: str | None
    query_subtype: str | None
    category: str | None
    measurement_type: str | None
    is_active: bool
    created_at: datetime
    tags: list = []

    model_config = {"from_attributes": True}


class LlmQueryBulkResponse(BaseModel):
    created: int
    skipped: int


class LlmQueryBulkAction(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=500)


class LlmQueryBulkToggle(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=500)
    is_active: bool
