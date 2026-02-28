"""Schemas for user-project access assignment."""

from pydantic import BaseModel


class UserProjectAssign(BaseModel):
    """Request to set project assignments for a user."""

    project_ids: list[int]


class UserProjectItem(BaseModel):
    """A single user-project assignment."""

    project_id: int
    project_name: str
