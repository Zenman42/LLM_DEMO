"""Pydantic schemas for Scenario Management API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScenarioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = Field("", max_length=2000)
    parent: str | None = Field(None, max_length=255)


class ScenarioUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=2000)
    parent: str | None = None


class ScenarioResponse(BaseModel):
    name: str
    description: str = ""
    parent: str | None = None
    prompt_count: int = 0


class ScenarioTreeItem(ScenarioResponse):
    children: list[ScenarioResponse] = []


class ScenarioListResponse(BaseModel):
    scenarios: list[ScenarioTreeItem]
    uncategorized_count: int = 0


class MovePromptsRequest(BaseModel):
    query_ids: list[int] = Field(min_length=1, max_length=500)
    target_scenario: str = Field("", max_length=255)


class PromptToSaveItem(BaseModel):
    user_prompt: str = Field(min_length=1, max_length=2000)
    query_class: str = ""
    query_subtype: str = ""
    measurement_type: str = ""


class ScenarioSavePromptsRequest(BaseModel):
    prompts: list[PromptToSaveItem] = Field(min_length=1, max_length=500)
