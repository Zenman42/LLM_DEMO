"""API endpoints for the Prompt Engineering Module.

Provides:
  - POST /prompt-engine/generate — generate a prompt test suite (preview, no save)
  - POST /prompt-engine/save/{project_id} — save user-curated prompts to project
  - POST /prompt-engine/generate-and-save — generate + save (legacy, still works)
  - POST /prompt-engine/preview — preview (summary only, no full prompts)
"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id
from app.core.encryption import decrypt_value
from app.core.exceptions import NotFoundError
from app.core.plan_limits import check_llm_query_limit
from app.db.postgres import get_db
from app.models.llm_query import LlmQuery
from app.models.project import Project
from app.models.tenant import Tenant
from app.prompt_engine.pipeline import PromptPipeline
from app.schemas.prompt_engine import (
    AuditProfileCreate,
    PromptSuiteSummary,
    SavePromptsRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prompt-engine", tags=["prompt-engine"])


async def _get_project(project_id: int, tenant_id: UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    return project


@router.post("/generate")
async def generate_prompt_suite(
    body: AuditProfileCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Generate a full prompt test suite without saving.

    Uses Gemini to generate natural prompts per scenario.
    Falls back to rule-based templates if no API key is available.
    """
    tenant = await db.get(Tenant, tenant_id)
    gemini_key = decrypt_value(tenant.gemini_api_key) if tenant and tenant.gemini_api_key else None

    pipeline = PromptPipeline()
    result = await pipeline.run_async(body, expansion_api_key=gemini_key)
    return result.to_response_dict()


@router.post("/preview", response_model=PromptSuiteSummary)
async def preview_prompt_suite(
    body: AuditProfileCreate,
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Preview prompt suite — returns summary statistics without full prompts.

    Use this to estimate the size of the prompt suite before generating.
    """
    pipeline = PromptPipeline()
    result = pipeline.run(body)
    return result.to_summary_dict()


@router.post("/save/{project_id}")
async def save_prompts(
    project_id: int,
    body: SavePromptsRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Save user-curated prompts to a project.

    This endpoint accepts prompts that were previously generated and
    reviewed/edited by the user in the UI. Each prompt is saved as
    an LlmQuery with V2 metadata (query_class, query_subtype).

    Workflow:
      1. User calls POST /generate to get prompt candidates
      2. User reviews, edits, adds, removes prompts in the UI
      3. User calls POST /save/{project_id} with the final list
    """
    project = await _get_project(project_id, tenant_id, db)

    # Check plan limits
    await check_llm_query_limit(db, tenant_id, adding=len(body.prompts))

    # Map V2 query_class to query_type for backward compatibility
    _class_to_query_type = {
        "thematic": "brand_check",
        "branded": "brand_check",
    }
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

    created = 0
    skipped = 0
    seen_texts: set[str] = set()

    for prompt in body.prompts:
        # Deduplicate within this batch
        text_key = prompt.user_prompt.strip().lower()
        if text_key in seen_texts:
            skipped += 1
            continue
        seen_texts.add(text_key)

        # Determine query_type from V2 subtype, fall back to class, then default
        query_type = (
            _subtype_to_query_type.get(prompt.query_subtype) or _class_to_query_type.get(prompt.query_class) or "custom"
        )

        cat_value = getattr(prompt, "category", None) or None
        stmt = (
            pg_insert(LlmQuery)
            .values(
                tenant_id=tenant_id,
                project_id=project_id,
                query_text=prompt.user_prompt[:2000],
                query_type=query_type,
                target_brand=body.brand or project.brand_name,
                competitors=body.competitors if body.competitors else None,
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
        else:
            skipped += 1

    await db.commit()

    return {
        "saved": {
            "created": created,
            "skipped_duplicates": skipped,
            "total_in_request": len(body.prompts),
        },
        "project_id": project_id,
    }


@router.post("/generate-and-save/{project_id}")
async def generate_and_save(
    project_id: int,
    body: AuditProfileCreate,
    max_prompts: int = Query(500, ge=1, le=5000, description="Max prompts to save"),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Generate prompt suite and save as LLM queries for the project.

    Legacy endpoint — generates and saves in one step.
    For interactive workflow, use /generate + /save/{project_id} instead.
    """
    project = await _get_project(project_id, tenant_id, db)

    # Load Gemini API key for LLM-based prompt generation
    tenant = await db.get(Tenant, tenant_id)
    gemini_key = decrypt_value(tenant.gemini_api_key) if tenant and tenant.gemini_api_key else None

    # Run pipeline with LLM generation (or template fallback)
    pipeline = PromptPipeline()
    result = await pipeline.run_async(body, expansion_api_key=gemini_key)

    # Map V2 subtype to query_type for backward compatibility
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
    # Legacy intent mapping as fallback
    _intent_to_query_type = {
        "informational": "brand_check",
        "navigational": "brand_check",
        "comparative": "comparison",
        "transactional": "recommendation",
    }

    # Limit prompts to save
    prompts_to_save = result.prompts[:max_prompts]

    # Check plan limits
    await check_llm_query_limit(db, tenant_id, adding=len(prompts_to_save))

    # Deduplicate and save
    created = 0
    skipped = 0
    seen_texts: set[str] = set()

    for prompt in prompts_to_save:
        # Deduplicate within this batch
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
                project_id=project_id,
                query_text=prompt.user_prompt[:2000],
                query_type=query_type,
                target_brand=prompt.brand or project.brand_name,
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
        else:
            skipped += 1

    await db.commit()

    return {
        "generation": result.to_summary_dict(),
        "saved": {
            "created": created,
            "skipped": skipped,
            "total_in_batch": len(prompts_to_save),
        },
        "project_id": project_id,
    }
