"""Scenarios API — CRUD, prompt generation, and hierarchy management.

Replaces the broken query_tags system. Stores scenario metadata in
Project.categories (JSONB) and binds prompts via LlmQuery.category (string).
"""

from __future__ import annotations

import logging
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.dependencies import get_current_tenant_id
from app.core.encryption import decrypt_value
from app.core.exceptions import NotFoundError
from app.core.plan_limits import check_llm_query_limit
from app.db.postgres import get_db
from app.models.llm_query import LlmQuery
from app.models.project import Project
from app.models.tenant import Tenant
from app.prompt_engine.pipeline import PromptPipeline
from app.schemas.prompt_engine import AuditProfileCreate
from app.schemas.scenario import (
    MovePromptsRequest,
    ScenarioCreate,
    ScenarioListResponse,
    ScenarioResponse,
    ScenarioSavePromptsRequest,
    ScenarioTreeItem,
    ScenarioUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects/{project_id}/scenarios", tags=["scenarios"])


# ── Helpers ────────────────────────────────────────────────────────


async def _get_project(project_id: int, tenant_id: UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    return project


def _normalize_categories(raw: list | None) -> list[dict]:
    """Ensure every item in categories is a dict with name/description/parent."""
    result = []
    for item in raw or []:
        if isinstance(item, str):
            result.append({"name": item, "description": "", "parent": None})
        elif isinstance(item, dict):
            result.append(
                {
                    "name": item.get("name") or item.get("category") or "",
                    "description": item.get("description") or "",
                    "parent": item.get("parent") or None,
                }
            )
        else:
            result.append({"name": str(item), "description": "", "parent": None})
    return result


def _find_scenario(cats: list[dict], name: str) -> dict | None:
    for cat in cats:
        if cat["name"] == name:
            return cat
    return None


def _save_categories(project: Project, cats: list[dict]) -> None:
    """Assign categories and mark JSONB as modified."""
    project.categories = cats
    flag_modified(project, "categories")


# ── GET /scenarios ─────────────────────────────────────────────────


@router.get("", response_model=ScenarioListResponse)
async def list_scenarios(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Return all scenarios as a tree with prompt counts."""
    project = await _get_project(project_id, tenant_id, db)
    cats = _normalize_categories(project.categories)

    # Get prompt counts per category
    rows = await db.execute(
        select(LlmQuery.category, func.count(LlmQuery.id))
        .where(
            LlmQuery.project_id == project_id,
            LlmQuery.tenant_id == tenant_id,
        )
        .group_by(LlmQuery.category)
    )
    count_map: dict[str | None, int] = {}
    for cat_name, cnt in rows.all():
        count_map[cat_name] = cnt

    uncategorized = count_map.get(None, 0) + count_map.get("", 0)

    # Build tree: top-level items (no parent) with children
    top_level: list[ScenarioTreeItem] = []
    children_map: dict[str, list[ScenarioResponse]] = {}

    for cat in cats:
        parent = cat.get("parent")
        resp = ScenarioResponse(
            name=cat["name"],
            description=cat.get("description", ""),
            parent=parent,
            prompt_count=count_map.get(cat["name"], 0),
        )
        if parent:
            children_map.setdefault(parent, []).append(resp)
        else:
            top_level.append(
                ScenarioTreeItem(
                    name=cat["name"],
                    description=cat.get("description", ""),
                    parent=None,
                    prompt_count=count_map.get(cat["name"], 0),
                    children=[],
                )
            )

    # Attach children
    for item in top_level:
        item.children = children_map.pop(item.name, [])

    # Orphaned children (parent doesn't exist) — promote to top-level
    for parent_name, orphans in children_map.items():
        for orphan in orphans:
            top_level.append(
                ScenarioTreeItem(
                    name=orphan.name,
                    description=orphan.description,
                    parent=orphan.parent,
                    prompt_count=orphan.prompt_count,
                    children=[],
                )
            )

    return ScenarioListResponse(scenarios=top_level, uncategorized_count=uncategorized)


# ── POST /scenarios ────────────────────────────────────────────────


@router.post("", status_code=201, response_model=ScenarioResponse)
async def create_scenario(
    project_id: int,
    body: ScenarioCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Create a new scenario."""
    project = await _get_project(project_id, tenant_id, db)
    cats = _normalize_categories(project.categories)

    # Check uniqueness
    if _find_scenario(cats, body.name):
        raise HTTPException(status_code=409, detail=f"Scenario '{body.name}' already exists")

    # Validate parent
    if body.parent:
        parent_cat = _find_scenario(cats, body.parent)
        if not parent_cat:
            raise HTTPException(status_code=400, detail=f"Parent scenario '{body.parent}' not found")

    cats.append(
        {
            "name": body.name,
            "description": body.description,
            "parent": body.parent,
        }
    )
    _save_categories(project, cats)
    await db.flush()

    return ScenarioResponse(
        name=body.name,
        description=body.description,
        parent=body.parent,
        prompt_count=0,
    )


# ── PUT /scenarios/{scenario_name} ─────────────────────────────────


@router.put("/{scenario_name}", response_model=ScenarioResponse)
async def update_scenario(
    project_id: int,
    scenario_name: str,
    body: ScenarioUpdate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Update scenario name, description, or parent."""
    scenario_name = unquote(scenario_name)
    project = await _get_project(project_id, tenant_id, db)
    cats = _normalize_categories(project.categories)

    target = _find_scenario(cats, scenario_name)
    if not target:
        raise NotFoundError(f"Scenario '{scenario_name}' not found")

    new_name = body.name if body.name is not None else target["name"]

    # If renaming, check uniqueness and update related data
    if body.name is not None and body.name != scenario_name:
        if _find_scenario(cats, body.name):
            raise HTTPException(status_code=409, detail=f"Scenario '{body.name}' already exists")

        # Update all prompts with old category name
        await db.execute(
            update(LlmQuery)
            .where(
                LlmQuery.project_id == project_id,
                LlmQuery.tenant_id == tenant_id,
                LlmQuery.category == scenario_name,
            )
            .values(category=body.name)
        )

        # Update children's parent references
        for cat in cats:
            if cat.get("parent") == scenario_name:
                cat["parent"] = body.name

        target["name"] = body.name

    if body.description is not None:
        target["description"] = body.description

    # Handle parent update (explicit None in body means "remove parent")
    if "parent" in body.model_fields_set:
        if body.parent is not None:
            parent_cat = _find_scenario(cats, body.parent)
            if not parent_cat:
                raise HTTPException(status_code=400, detail=f"Parent scenario '{body.parent}' not found")
            if body.parent == new_name:
                raise HTTPException(status_code=400, detail="Scenario cannot be its own parent")
        target["parent"] = body.parent

    _save_categories(project, cats)
    await db.flush()

    # Get prompt count
    count_result = await db.execute(
        select(func.count(LlmQuery.id)).where(
            LlmQuery.project_id == project_id,
            LlmQuery.tenant_id == tenant_id,
            LlmQuery.category == new_name,
        )
    )

    return ScenarioResponse(
        name=target["name"],
        description=target["description"],
        parent=target.get("parent"),
        prompt_count=count_result.scalar() or 0,
    )


# ── DELETE /scenarios/{scenario_name} ──────────────────────────────


@router.delete("/{scenario_name}", status_code=204)
async def delete_scenario(
    project_id: int,
    scenario_name: str,
    delete_prompts: bool = Query(False, description="Delete prompts too (otherwise uncategorize)"),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Delete a scenario. Optionally delete or uncategorize its prompts."""
    scenario_name = unquote(scenario_name)
    project = await _get_project(project_id, tenant_id, db)
    cats = _normalize_categories(project.categories)

    target = _find_scenario(cats, scenario_name)
    if not target:
        raise NotFoundError(f"Scenario '{scenario_name}' not found")

    # Handle prompts
    if delete_prompts:
        await db.execute(
            delete(LlmQuery).where(
                LlmQuery.project_id == project_id,
                LlmQuery.tenant_id == tenant_id,
                LlmQuery.category == scenario_name,
            )
        )
    else:
        await db.execute(
            update(LlmQuery)
            .where(
                LlmQuery.project_id == project_id,
                LlmQuery.tenant_id == tenant_id,
                LlmQuery.category == scenario_name,
            )
            .values(category=None)
        )

    # Re-parent children to null (make standalone)
    for cat in cats:
        if cat.get("parent") == scenario_name:
            cat["parent"] = None

    # Remove from categories
    cats = [c for c in cats if c["name"] != scenario_name]
    _save_categories(project, cats)
    await db.flush()


# ── POST /scenarios/move-prompts ───────────────────────────────────


@router.post("/move-prompts")
async def move_prompts(
    project_id: int,
    body: MovePromptsRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Move prompts to a different scenario (or uncategorize)."""
    project = await _get_project(project_id, tenant_id, db)

    target_scenario = body.target_scenario.strip() or None

    # Validate target scenario exists (unless uncategorizing)
    if target_scenario:
        cats = _normalize_categories(project.categories)
        if not _find_scenario(cats, target_scenario):
            raise HTTPException(status_code=400, detail=f"Target scenario '{target_scenario}' not found")

    result = await db.execute(
        update(LlmQuery)
        .where(
            LlmQuery.project_id == project_id,
            LlmQuery.tenant_id == tenant_id,
            LlmQuery.id.in_(body.query_ids),
        )
        .values(category=target_scenario)
    )
    await db.flush()

    return {"moved": result.rowcount}


# ── POST /scenarios/{scenario_name}/generate ───────────────────────


@router.post("/{scenario_name}/generate")
async def generate_scenario_prompts(
    project_id: int,
    scenario_name: str,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Generate prompts for a specific scenario using the prompt engine."""
    scenario_name = unquote(scenario_name)
    project = await _get_project(project_id, tenant_id, db)
    cats = _normalize_categories(project.categories)

    target = _find_scenario(cats, scenario_name)
    if not target:
        raise NotFoundError(f"Scenario '{scenario_name}' not found")

    # Build AuditProfileCreate from project data
    competitors = project.competitors or []
    if not competitors:
        raise HTTPException(status_code=400, detail="Project has no competitors configured")

    try:
        body = AuditProfileCreate(
            brand=project.brand_name or "Brand",
            brand_aliases=project.brand_aliases or [],
            competitors=competitors,
            categories=[{"name": target["name"], "description": target.get("description", "")}],
            market=project.market or "ru",
            geo=project.geo or "",
            golden_facts=project.golden_facts or [],
            brand_description=project.brand_description or "",
            target_llms=["chatgpt"],  # single LLM — UI shows user_prompt only
            enable_expansion=True,
            enable_pivots=False,
        )
    except Exception as e:
        logger.warning("Invalid project data for generation: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid project data: {e}")

    # Get Gemini API key
    tenant = await db.get(Tenant, tenant_id)
    gemini_key = decrypt_value(tenant.gemini_api_key) if tenant and tenant.gemini_api_key else None

    try:
        pipeline = PromptPipeline()
        result = await pipeline.run_async(body, expansion_api_key=gemini_key)
        return result.to_response_dict()
    except Exception as e:
        logger.exception(
            "Prompt generation failed for project=%d scenario=%s",
            project_id,
            scenario_name,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prompt generation failed: {e}",
        )


# ── POST /scenarios/{scenario_name}/save-prompts ───────────────────


@router.post("/{scenario_name}/save-prompts")
async def save_scenario_prompts(
    project_id: int,
    scenario_name: str,
    body: ScenarioSavePromptsRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Save generated prompts to a scenario."""
    scenario_name = unquote(scenario_name)
    project = await _get_project(project_id, tenant_id, db)
    cats = _normalize_categories(project.categories)

    if not _find_scenario(cats, scenario_name):
        raise NotFoundError(f"Scenario '{scenario_name}' not found")

    # Check plan limits
    await check_llm_query_limit(db, tenant_id, adding=len(body.prompts))

    _class_to_query_type = {
        "thematic": "recommendation",
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
        text_key = prompt.user_prompt.strip().lower()
        if text_key in seen_texts:
            skipped += 1
            continue
        seen_texts.add(text_key)

        query_type = (
            _subtype_to_query_type.get(prompt.query_subtype) or _class_to_query_type.get(prompt.query_class) or "custom"
        )

        stmt = (
            pg_insert(LlmQuery)
            .values(
                tenant_id=tenant_id,
                project_id=project_id,
                query_text=prompt.user_prompt[:2000],
                query_type=query_type,
                target_brand=project.brand_name,
                measurement_type=prompt.measurement_type or None,
                query_class=prompt.query_class or None,
                query_subtype=prompt.query_subtype or None,
                category=scenario_name,
            )
            .on_conflict_do_update(
                constraint="uq_project_llm_query",
                set_={"category": scenario_name},
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
    }
