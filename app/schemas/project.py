from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    domain: str | None = Field(None, max_length=255)
    search_engine: str = Field("both", pattern=r"^(google|yandex|both)$")
    region_google: str | None = Field(None, max_length=200)
    region_yandex: int = 213
    ywm_host_id: str | None = Field(None, max_length=200)
    gsc_site_url: str | None = Field(None, max_length=200)
    # LLM tracking
    track_llm: bool = False
    track_ai_overview: bool = False
    llm_providers: list[str] | None = None  # ["chatgpt", "deepseek", "perplexity", "yandexgpt"]
    brand_name: str | None = Field(None, max_length=255)
    brand_description: str | None = None
    # Onboarding audit profile fields
    brand_aliases: list[str] | None = None
    competitors: list[str] | None = None
    categories: list[Any] | None = None
    golden_facts: list[str] | None = None
    personas: list[str] | None = None
    brand_sub_brands: dict[str, list[str]] | None = None
    competitor_sub_brands: dict[str, dict[str, list[str]]] | None = None
    market: str | None = Field(None, pattern=r"^(ru|en)$")
    geo: str | None = Field(None, max_length=255)


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    domain: str | None = Field(None, max_length=255)
    search_engine: str | None = Field(None, pattern=r"^(google|yandex|both)$")
    region_google: str | None = None
    region_yandex: int | None = None
    ywm_host_id: str | None = None
    gsc_site_url: str | None = None
    is_active: bool | None = None
    # LLM tracking
    track_llm: bool | None = None
    track_ai_overview: bool | None = None
    llm_providers: list[str] | None = None
    brand_name: str | None = None
    brand_description: str | None = None
    # Onboarding audit profile fields
    brand_aliases: list[str] | None = None
    competitors: list[str] | None = None
    categories: list[Any] | None = None
    golden_facts: list[str] | None = None
    personas: list[str] | None = None
    brand_sub_brands: dict[str, list[str]] | None = None
    competitor_sub_brands: dict[str, dict[str, list[str]]] | None = None
    market: str | None = Field(None, pattern=r"^(ru|en)$")
    geo: str | None = Field(None, max_length=255)


class ProjectResponse(BaseModel):
    id: int
    tenant_id: UUID
    name: str
    domain: str | None
    search_engine: str
    region_google: str | None
    region_yandex: int
    ywm_host_id: str | None
    gsc_site_url: str | None
    track_llm: bool
    track_ai_overview: bool
    llm_providers: list[str] | None
    brand_name: str | None
    brand_description: str | None = None
    # Onboarding audit profile fields
    brand_aliases: list[str] | None = None
    competitors: list[str] | None = None
    categories: list[Any] | None = None
    golden_facts: list[str] | None = None
    personas: list[str] | None = None
    brand_sub_brands: dict[str, list[str]] | None = None
    competitor_sub_brands: dict[str, dict[str, list[str]]] | None = None
    market: str | None = None
    geo: str | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectWithStats(ProjectResponse):
    keyword_count: int = 0
    llm_query_count: int = 0
    last_collected: datetime | None = None
