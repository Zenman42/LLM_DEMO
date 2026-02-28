"""Pydantic schemas for the Prompt Engineering Module API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input: Audit profile from the user
# ---------------------------------------------------------------------------


class AuditProfileCreate(BaseModel):
    """Input profile for prompt generation."""

    brand: str = Field(min_length=1, max_length=255, description="Primary brand name")
    brand_aliases: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Synonyms, abbreviations, alternative spellings",
    )
    competitors: list[str] = Field(
        min_length=1,
        description="Competitor brand names (at least one required)",
    )
    categories: list[Any] = Field(
        min_length=1,
        max_length=20,
        description="Product/service categories — strings (legacy) or scenario objects {name, description, context_hints, thematic_subtypes, branded_subtypes}",
    )
    market: str = Field(
        "ru",
        pattern=r"^(ru|en)$",
        description="Target market language: ru or en",
    )
    geo: str = Field(
        "",
        max_length=255,
        description="Geographic context, e.g. 'Russia', 'Moscow'",
    )

    # Generation options
    personas: list[str] = Field(
        default_factory=lambda: ["beginner", "skeptic"],
        description="Which personas to include (default: beginner, skeptic)",
    )
    intents: list[str] = Field(
        default_factory=lambda: ["informational", "navigational", "comparative", "transactional"],
        description="Which intent types to include (legacy, used for fallback only)",
    )
    target_llms: list[str] = Field(
        default_factory=lambda: ["chatgpt", "deepseek", "perplexity", "yandexgpt", "gigachat"],
        description="Which LLM providers to generate prompts for",
    )
    golden_facts: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Anti-hallucination facts about the brand for fact-check prompts",
    )
    brand_description: str = Field(
        "",
        description="Comprehensive company profile with full product/service catalog",
    )
    enable_expansion: bool = Field(
        True,
        description="Enable LLM-based semantic expansion (generates 5-10x more prompts)",
    )
    enable_pivots: bool = Field(
        True,
        description="Enable semantic pivots (adjective substitution)",
    )


# ---------------------------------------------------------------------------
# Input: Save selected prompts to project
# ---------------------------------------------------------------------------


class PromptToSave(BaseModel):
    """A single prompt selected by the user for saving."""

    user_prompt: str = Field(min_length=1, max_length=2000, description="Prompt text")
    query_class: str = Field("", description="thematic | branded")
    query_subtype: str = Field(
        "", description="category | attribute | scenario | event | reputation | comparison | negative | fact_check"
    )
    persona: str = Field("", description="Persona used for generation")
    intent: str = Field("", description="Legacy intent")
    category: str = Field("", description="Business category")
    measurement_type: str = Field("", description="Legacy measurement type")


class SavePromptsRequest(BaseModel):
    """Request to save a curated list of prompts to a project."""

    prompts: list[PromptToSave] = Field(
        min_length=1,
        max_length=5000,
        description="List of prompts to save (edited/curated by user)",
    )
    brand: str = Field(min_length=1, max_length=255, description="Target brand name")
    competitors: list[str] = Field(
        default_factory=list,
        description="Competitor brand names",
    )


# ---------------------------------------------------------------------------
# Output: Generated prompt suite
# ---------------------------------------------------------------------------


class PromptMetadata(BaseModel):
    """Metadata attached to each generated prompt."""

    persona: str
    intent: str
    category: str
    brand: str
    competitors: list[str]
    market: str
    skeleton_id: str = ""
    pivot_axis: str = ""
    measurement_type: str = ""
    query_class: str = ""
    query_subtype: str = ""


class FinalPromptResponse(BaseModel):
    """Single generated prompt ready for dispatch."""

    prompt_id: str
    target_llm: str
    system_prompt: str
    user_prompt: str
    temperature: float
    resilience_runs: int
    max_tokens: int
    metadata: PromptMetadata


class PromptSuiteResponse(BaseModel):
    """Complete prompt test suite — output of Module 1."""

    total_prompts: int
    skeletons_generated: int
    expanded_prompts: int
    adapted_prompts: int
    by_provider: dict[str, int]
    by_intent: dict[str, int]
    by_persona: dict[str, int]
    by_measurement_type: dict[str, int] = {}
    by_query_class: dict[str, int] = {}
    by_query_subtype: dict[str, int] = {}
    prompts: list[FinalPromptResponse]


class PromptSuiteSummary(BaseModel):
    """Summary without full prompt list (for large suites)."""

    total_prompts: int
    skeletons_generated: int
    expanded_prompts: int
    adapted_prompts: int
    by_provider: dict[str, int]
    by_intent: dict[str, int]
    by_persona: dict[str, int]
    by_measurement_type: dict[str, int] = {}
    by_query_class: dict[str, int] = {}
    by_query_subtype: dict[str, int] = {}


# Rebuild models to resolve forward references
FinalPromptResponse.model_rebuild()
