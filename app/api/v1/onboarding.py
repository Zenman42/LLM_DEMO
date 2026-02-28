"""Onboarding endpoint — atomic project creation + prompt generation.

POST /api/v1/onboarding/complete
  1. Creates a Project with all audit-profile fields
  2. Runs PromptPipeline to generate prompts
  3. Saves prompts as LlmQuery rows
  4. Returns project_id + generation stats + sample prompts
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id
from app.core.encryption import decrypt_value
from app.core.plan_limits import check_project_limit  # noqa: F401
from app.db.postgres import get_db
from app.models.llm_query import LlmQuery
from app.models.project import Project
from app.models.query_tag import QueryTag
from app.models.tenant import Tenant
from app.services.tag_utils import link_tags_for_prompt
from app.prompt_engine.pipeline import PromptPipeline
from app.schemas.prompt_engine import AuditProfileCreate
from app.services.brand_research import (
    research_brand,
    suggest_categories,
    suggest_competitors,
    suggest_scenario_description,
    suggest_sub_brands,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


async def _create_scenario_tags(db: AsyncSession, project_id: int, tenant_id: UUID, categories: list) -> None:
    """Create scenario tags from project categories list (strings or ScenarioConfig objects)."""
    for cat in categories:
        # Categories can be plain strings or dicts with {name, description}
        if isinstance(cat, str):
            tag_name, description = cat, None
        elif isinstance(cat, dict):
            tag_name = cat.get("name") or cat.get("category") or str(cat)
            description = cat.get("description")
        else:
            tag_name, description = str(cat), None

        if not tag_name:
            continue

        existing = await db.execute(
            select(QueryTag.id).where(
                QueryTag.project_id == project_id,
                QueryTag.tag_name == tag_name,
                QueryTag.tag_type == "scenario",
            )
        )
        if not existing.scalar_one_or_none():
            db.add(
                QueryTag(
                    project_id=project_id,
                    tenant_id=tenant_id,
                    tag_name=tag_name,
                    tag_type="scenario",
                    description=description,
                )
            )
    await db.flush()


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class OnboardingRequest(BaseModel):
    """Full wizard payload — all 4 steps combined."""

    # Step 1: Brand DNA (includes market & geo)
    domain: str | None = Field(None, max_length=255)
    brand_name: str = Field(min_length=1, max_length=255)
    brand_aliases: list[str] = Field(default_factory=list, max_length=20)
    market: str = Field("ru", pattern=r"^(ru|en)$")
    geo: str = Field("", max_length=255)

    # Step 2: Competitive Radar
    competitors: list[str] = Field(min_length=1, max_length=10)

    # Step 3: Customer Scenarios & Prompts
    categories: list[Any] = Field(min_length=1, max_length=20)

    # Hard-coded personas (beginner + skeptic)
    personas: list[str] = Field(
        default_factory=lambda: ["beginner", "skeptic"],
    )

    # Step 4: Finalize & Launch
    golden_facts: list[str] = Field(default_factory=list, max_length=50)

    # Brand context (auto-generated or user-edited)
    brand_description: str = Field("")

    # Generation options (review screen)
    target_llms: list[str] = Field(
        default_factory=lambda: ["chatgpt", "deepseek", "yandexgpt", "gigachat"],
    )
    intents: list[str] = Field(
        default_factory=lambda: ["informational", "navigational", "comparative", "transactional"],
    )
    enable_expansion: bool = False
    enable_pivots: bool = True

    # Sub-brands (optional)
    brand_sub_brands: dict[str, list[str]] = Field(default_factory=dict)
    competitor_sub_brands: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Brand research endpoint
# ---------------------------------------------------------------------------


class BrandResearchRequest(BaseModel):
    """Request to auto-research a brand."""

    brand_name: str = Field(min_length=1, max_length=255)
    domain: str | None = Field(None, max_length=255)
    market: str = Field("ru", pattern=r"^(ru|en)$")


@router.post("/research-brand")
async def research_brand_endpoint(
    body: BrandResearchRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Auto-research a brand and return a description.

    Called from the wizard Step 1 when the user enters a brand name.
    Uses gpt-4o-mini to generate a 2-4 sentence company description.
    """
    tenant = await db.get(Tenant, tenant_id)
    openai_key = decrypt_value(tenant.openai_api_key) if tenant and tenant.openai_api_key else None
    perplexity_key = decrypt_value(tenant.perplexity_api_key) if tenant and tenant.perplexity_api_key else None
    gemini_key = decrypt_value(tenant.gemini_api_key) if tenant and tenant.gemini_api_key else None

    if not openai_key and not perplexity_key and not gemini_key:
        return {
            "brand_description": "",
            "error": "no_api_key",
            "message": "No API keys configured (Gemini, OpenAI or Perplexity). Enter brand description manually.",
        }

    try:
        description = await research_brand(
            brand_name=body.brand_name,
            api_key=openai_key or "",
            domain=body.domain,
            market=body.market,
            perplexity_api_key=perplexity_key,
            gemini_api_key=gemini_key,
        )
        return {"brand_description": description}
    except Exception as e:
        logger.warning("Brand research failed for '%s': %s", body.brand_name, e)
        return {
            "brand_description": "",
            "error": "research_failed",
            "message": f"Could not research brand: {e}",
        }


# ---------------------------------------------------------------------------
# Competitor suggestion endpoint
# ---------------------------------------------------------------------------


class SuggestRequest(BaseModel):
    """Request to suggest competitors or categories."""

    brand_name: str = Field(min_length=1, max_length=255)
    brand_description: str = Field("")
    market: str = Field("ru", pattern=r"^(ru|en)$")
    existing: list[str] = Field(default_factory=list)


@router.post("/suggest-competitors")
async def suggest_competitors_endpoint(
    body: SuggestRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Suggest direct competitors for the brand using AI.

    Called from the wizard Step 2 when the user clicks "Suggest".
    Uses Perplexity (web search) or OpenAI as fallback.
    """
    tenant = await db.get(Tenant, tenant_id)
    openai_key = decrypt_value(tenant.openai_api_key) if tenant and tenant.openai_api_key else None
    perplexity_key = decrypt_value(tenant.perplexity_api_key) if tenant and tenant.perplexity_api_key else None
    gemini_key = decrypt_value(tenant.gemini_api_key) if tenant and tenant.gemini_api_key else None

    if not openai_key and not perplexity_key and not gemini_key:
        return {"competitors": [], "error": "no_api_key"}

    try:
        competitors = await suggest_competitors(
            brand_name=body.brand_name,
            brand_description=body.brand_description,
            api_key=openai_key or "",
            market=body.market,
            perplexity_api_key=perplexity_key,
            gemini_api_key=gemini_key,
            existing=body.existing,
        )
        return {"competitors": competitors}
    except Exception as e:
        logger.warning("Competitor suggestion failed for '%s': %s", body.brand_name, e)
        return {"competitors": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Category suggestion endpoint
# ---------------------------------------------------------------------------


@router.post("/suggest-categories")
async def suggest_categories_endpoint(
    body: SuggestRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Suggest product categories / user scenarios using AI.

    Called from the wizard Step 3 when the user clicks "Suggest".
    Uses Perplexity (web search) or OpenAI as fallback.
    """
    tenant = await db.get(Tenant, tenant_id)
    openai_key = decrypt_value(tenant.openai_api_key) if tenant and tenant.openai_api_key else None
    perplexity_key = decrypt_value(tenant.perplexity_api_key) if tenant and tenant.perplexity_api_key else None
    gemini_key = decrypt_value(tenant.gemini_api_key) if tenant and tenant.gemini_api_key else None

    if not openai_key and not perplexity_key and not gemini_key:
        return {"categories": [], "error": "no_api_key"}

    try:
        categories = await suggest_categories(
            brand_name=body.brand_name,
            brand_description=body.brand_description,
            api_key=openai_key or "",
            market=body.market,
            perplexity_api_key=perplexity_key,
            gemini_api_key=gemini_key,
            existing=body.existing,
        )
        return {"categories": categories}
    except Exception as e:
        logger.warning("Category suggestion failed for '%s': %s", body.brand_name, e)
        return {"categories": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Sub-brand suggestion endpoint
# ---------------------------------------------------------------------------


@router.post("/suggest-sub-brands")
async def suggest_sub_brands_endpoint(
    body: SuggestRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Suggest sub-brands / product lines for a brand using AI.

    Called from wizard Step 1 (brand sub-brands) and Step 2 (competitor sub-brands).
    """
    tenant = await db.get(Tenant, tenant_id)
    openai_key = decrypt_value(tenant.openai_api_key) if tenant and tenant.openai_api_key else None
    perplexity_key = decrypt_value(tenant.perplexity_api_key) if tenant and tenant.perplexity_api_key else None
    gemini_key = decrypt_value(tenant.gemini_api_key) if tenant and tenant.gemini_api_key else None

    if not openai_key and not perplexity_key and not gemini_key:
        return {"sub_brands": [], "error": "no_api_key"}

    try:
        sub_brands = await suggest_sub_brands(
            brand_name=body.brand_name,
            brand_description=body.brand_description,
            api_key=openai_key or "",
            market=body.market,
            perplexity_api_key=perplexity_key,
            gemini_api_key=gemini_key,
            existing=body.existing,
        )
        return {"sub_brands": sub_brands}
    except Exception as e:
        logger.warning("Sub-brand suggestion failed for '%s': %s", body.brand_name, e)
        return {"sub_brands": [], "error": str(e)}


# ---------------------------------------------------------------------------
# Scenario description generation endpoint
# ---------------------------------------------------------------------------


class ScenarioDescRequest(BaseModel):
    """Request to generate a scenario description."""

    brand_name: str = Field(min_length=1, max_length=255)
    brand_description: str = Field("")
    scenario_name: str = Field(min_length=1, max_length=255)
    market: str = Field("ru", pattern=r"^(ru|en)$")


@router.post("/generate-scenario-description")
async def generate_scenario_description_endpoint(
    body: ScenarioDescRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Generate a 1-2 sentence description for a scenario.

    Called from the wizard Step 3 when the user clicks "Generate Description"
    for a specific scenario. Uses the brand context to produce a relevant
    description.
    """
    tenant = await db.get(Tenant, tenant_id)
    openai_key = decrypt_value(tenant.openai_api_key) if tenant and tenant.openai_api_key else None
    perplexity_key = decrypt_value(tenant.perplexity_api_key) if tenant and tenant.perplexity_api_key else None
    gemini_key = decrypt_value(tenant.gemini_api_key) if tenant and tenant.gemini_api_key else None

    if not openai_key and not perplexity_key and not gemini_key:
        return {"description": "", "error": "no_api_key"}

    try:
        description = await suggest_scenario_description(
            brand_name=body.brand_name,
            brand_description=body.brand_description,
            scenario_name=body.scenario_name,
            api_key=openai_key or "",
            market=body.market,
            perplexity_api_key=perplexity_key,
            gemini_api_key=gemini_key,
        )
        return {"description": description}
    except Exception as e:
        logger.warning("Scenario description generation failed: %s", e)
        return {"description": "", "error": str(e)}


# ---------------------------------------------------------------------------
# Create project only (no prompt generation) — used by the new interactive flow
# ---------------------------------------------------------------------------


@router.post("/create-project")
async def create_project(
    body: OnboardingRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Create a project from wizard data WITHOUT generating prompts.

    V3 interactive workflow:
      1. User fills wizard steps 1-3 (per-scenario prompt generation & review)
      2. Step 4: POST /onboarding/create-project → returns project_id
      3. POST /prompt-engine/save/{project_id} → saves curated prompts
    """
    project = Project(
        tenant_id=tenant_id,
        name=body.brand_name,
        domain=body.domain,
        search_engine="both",
        region_yandex=213,
        track_llm=True,
        track_ai_overview=False,
        llm_providers=body.target_llms,
        brand_name=body.brand_name,
        brand_description=body.brand_description or None,
        brand_aliases=body.brand_aliases,
        competitors=body.competitors,
        categories=body.categories,
        golden_facts=body.golden_facts,
        personas=body.personas,
        market=body.market,
        geo=body.geo,
        brand_sub_brands=body.brand_sub_brands or {},
        competitor_sub_brands=body.competitor_sub_brands or {},
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    logger.info("Project created (no prompts): project=%d, tenant=%s", project.id, tenant_id)

    return {
        "project_id": project.id,
        "project_name": project.name,
    }


# ---------------------------------------------------------------------------
# Onboarding complete endpoint (legacy — atomic create + generate + save)
# ---------------------------------------------------------------------------


@router.post("/complete")
async def onboarding_complete(
    body: OnboardingRequest,
    max_prompts: int = Query(500, ge=1, le=5000),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Atomic onboarding: create project + generate & save LLM queries.

    Called when the user clicks "Generate & Launch" on the last wizard step.
    """

    # 1. Create project with all audit-profile fields
    project = Project(
        tenant_id=tenant_id,
        name=body.brand_name,
        domain=body.domain,
        search_engine="both",
        region_yandex=213,
        track_llm=True,
        track_ai_overview=False,
        llm_providers=body.target_llms,
        brand_name=body.brand_name,
        brand_description=body.brand_description or None,
        brand_aliases=body.brand_aliases,
        competitors=body.competitors,
        categories=body.categories,
        golden_facts=body.golden_facts,
        personas=body.personas,
        market=body.market,
        geo=body.geo,
        brand_sub_brands=body.brand_sub_brands or {},
        competitor_sub_brands=body.competitor_sub_brands or {},
    )
    db.add(project)
    await db.flush()  # get project.id

    # 2. Create scenario tags from categories
    await _create_scenario_tags(db, project.id, tenant_id, body.categories)

    # 3. Build AuditProfileCreate and run the pipeline
    audit_input = AuditProfileCreate(
        brand=body.brand_name,
        brand_aliases=body.brand_aliases,
        competitors=body.competitors,
        categories=body.categories,
        market=body.market,
        geo=body.geo,
        golden_facts=body.golden_facts,
        brand_description=body.brand_description,
        personas=body.personas,
        intents=body.intents,
        target_llms=body.target_llms,
        enable_expansion=body.enable_expansion,
        enable_pivots=body.enable_pivots,
    )

    # Load Gemini API key for LLM-based prompt generation
    tenant = await db.get(Tenant, tenant_id)
    gemini_key = decrypt_value(tenant.gemini_api_key) if tenant and tenant.gemini_api_key else None

    pipeline = PromptPipeline()
    result = await pipeline.run_async(audit_input, expansion_api_key=gemini_key)

    # 4. Save prompts as LlmQuery rows with V2 metadata
    _subtype_to_query_type = {
        "comparison": "comparison",
        "reputation": "brand_check",
        "negative": "brand_check",
        "fact_check": "brand_check",
        "category": "recommendation",
        "attribute": "recommendation",
        "scenario": "recommendation",
        "event": "recommendation",
    }
    _intent_to_query_type = {
        "informational": "brand_check",
        "navigational": "brand_check",
        "comparative": "comparison",
        "transactional": "recommendation",
    }

    prompts_to_save = result.prompts[:max_prompts]

    created = 0
    skipped = 0
    seen_texts: set[str] = set()

    for prompt in prompts_to_save:
        text_key = prompt.user_prompt.strip().lower()
        if text_key in seen_texts:
            skipped += 1
            continue
        seen_texts.add(text_key)

        query_type = _subtype_to_query_type.get(prompt.query_subtype) or _intent_to_query_type.get(
            prompt.intent.value, "custom"
        )

        cat_value = getattr(prompt, "category", None) or None
        stmt = (
            pg_insert(LlmQuery)
            .values(
                tenant_id=tenant_id,
                project_id=project.id,
                query_text=prompt.user_prompt[:2000],
                query_type=query_type,
                target_brand=prompt.brand or body.brand_name,
                competitors=prompt.competitors if prompt.competitors else None,
                measurement_type=prompt.measurement_type or None,
                query_class=prompt.query_class or None,
                query_subtype=prompt.query_subtype or None,
                category=cat_value,
            )
            .on_conflict_do_update(
                constraint="uq_project_llm_query",
                set_={"category": cat_value},
                where=LlmQuery.category.is_(None),
            )
        )
        exec_result = await db.execute(stmt)
        if exec_result.rowcount > 0:
            created += 1
            # Dual write: link tags
            id_result = await db.execute(
                select(LlmQuery.id).where(
                    LlmQuery.project_id == project.id,
                    LlmQuery.query_text == prompt.user_prompt[:2000],
                )
            )
            qid = id_result.scalar_one_or_none()
            if qid:
                await link_tags_for_prompt(
                    db,
                    qid,
                    project.id,
                    tenant_id,
                    query_class=prompt.query_class,
                    category=cat_value,
                )
        else:
            skipped += 1

    await db.commit()

    # 5. Pick up to 3 sample prompts for the response
    sample_prompts = []
    for p in result.prompts[:3]:
        sample_prompts.append(
            {
                "text": p.user_prompt,
                "persona": p.persona.value if hasattr(p.persona, "value") else str(p.persona),
                "intent": p.intent.value if hasattr(p.intent, "value") else str(p.intent),
                "target_llm": p.target_llm.value if hasattr(p.target_llm, "value") else str(p.target_llm),
            }
        )

    logger.info(
        "Onboarding complete: project=%d, prompts created=%d, skipped=%d",
        project.id,
        created,
        skipped,
    )

    return {
        "project_id": project.id,
        "project_name": project.name,
        "generation": result.to_summary_dict(),
        "saved": {
            "created": created,
            "skipped": skipped,
            "total_in_batch": len(prompts_to_save),
        },
        "sample_prompts": sample_prompts,
    }
