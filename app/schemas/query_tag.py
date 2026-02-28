from datetime import datetime

from pydantic import BaseModel, Field


class QueryTagCreate(BaseModel):
    tag_name: str = Field(min_length=1, max_length=255)
    tag_type: str = Field(pattern=r"^(class|scenario)$")
    description: str | None = None


class QueryTagUpdate(BaseModel):
    tag_name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class QueryTagResponse(BaseModel):
    id: int
    project_id: int
    tag_name: str
    tag_type: str
    description: str | None
    created_at: datetime
    query_count: int = 0

    model_config = {"from_attributes": True}


class QueryTagBrief(BaseModel):
    """Minimal tag info for embedding in query responses."""

    id: int
    tag_name: str
    tag_type: str


class TagAssignment(BaseModel):
    """Assign/remove tags from queries."""

    query_ids: list[int] = Field(min_length=1, max_length=500)
    tag_ids: list[int] = Field(min_length=1, max_length=50)
