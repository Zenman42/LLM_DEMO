"""Competitors CRUD — manage project competitors and discovered entities."""

from datetime import date, datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.dependencies import get_current_tenant_id
from app.core.exceptions import BadRequestError, NotFoundError
from app.db.postgres import get_db
from app.models.discovered_entity import DiscoveredEntity
from app.models.llm_query import LlmQuery
from app.models.llm_snapshot import LlmSnapshot
from app.models.project import Project
from app.models.tenant import Tenant
from app.schemas.competitor import (
    AddCompetitorRequest,
    AliasInfo,
    BrandInfo,
    CompetitorInfo,
    CompetitorListResponse,
    DiscoveredEntityResponse,
    MergeBrandRequest,
    PromoteRequest,
    RejectRequest,
    RemoveAliasRequest,
    ResetEntityAliasesRequest,
    SetAliasRequest,
    SubBrandCreate,
    SubBrandResponse,
    VerifyApplyRequest,
    VerifyRequest,
    VerifyResponse,
)

router = APIRouter(prefix="/projects/{project_id}/competitors", tags=["competitors"])


# ── Helpers ──────────────────────────────────────────────────────


async def _get_project(project_id: int, tenant_id: UUID, db: AsyncSession) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    return project


async def _find_entity(db: AsyncSession, project_id: int, entity_name: str) -> DiscoveredEntity | None:
    """Find a DiscoveredEntity by name within project."""
    result = await db.execute(
        select(DiscoveredEntity).where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.entity_name == entity_name,
        )
    )
    return result.scalar_one_or_none()


async def _ensure_entity(
    db: AsyncSession,
    project_id: int,
    tenant_id: UUID,
    entity_name: str,
    status: str = "promoted",
) -> DiscoveredEntity:
    """Find or create a DiscoveredEntity with the given status."""
    entity = await _find_entity(db, project_id, entity_name)
    if entity:
        if entity.status != status:
            entity.status = status
        return entity
    today = date.today()
    entity = DiscoveredEntity(
        tenant_id=tenant_id,
        project_id=project_id,
        entity_name=entity_name,
        mention_count=0,
        first_seen=today,
        last_seen=today,
        status=status,
    )
    db.add(entity)
    await db.flush()
    return entity


async def _get_aggregated_mentions(db: AsyncSession, entity_id: int) -> int:
    """Get total mentions: own + all aliases."""
    result = await db.execute(
        select(func.coalesce(func.sum(DiscoveredEntity.mention_count), 0)).where(
            (DiscoveredEntity.id == entity_id) | (DiscoveredEntity.alias_of_id == entity_id)
        )
    )
    return result.scalar() or 0


async def _get_entity_mention_counts(db: AsyncSession, entity: DiscoveredEntity | None) -> tuple[int, int]:
    """Return (own_mentions, with_aliases_mentions) for an entity.

    own = entity's DiscoveredEntity.mention_count
    with_aliases = own + sum(alias.mention_count for each alias)
    """
    if not entity:
        return 0, 0
    own = entity.mention_count or 0
    with_aliases = await _get_aggregated_mentions(db, entity.id)
    return own, with_aliases


def _sum_alias_mentions(aliases: list[AliasInfo]) -> int:
    """Sum mention_count from alias info list."""
    return sum(a.mention_count for a in aliases)


async def _build_alias_info_list(db: AsyncSession, entity_id: int) -> list[AliasInfo]:
    """Build list of AliasInfo for entity's aliases (entities with alias_of_id pointing here)."""
    result = await db.execute(
        select(DiscoveredEntity)
        .where(DiscoveredEntity.alias_of_id == entity_id)
        .order_by(DiscoveredEntity.mention_count.desc())
    )
    aliases = result.scalars().all()
    return [AliasInfo(id=a.id, entity_name=a.entity_name, mention_count=a.mention_count) for a in aliases]


async def _release_entity_aliases(db: AsyncSession, project_id: int, entity_id: int) -> int:
    """Clear alias_of_id on all aliases of the given entity, setting them to confirmed.

    Returns the number of aliases released.
    """
    result = await db.execute(
        update(DiscoveredEntity)
        .where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.alias_of_id == entity_id,
        )
        .values(alias_of_id=None, status="confirmed")
    )
    return result.rowcount


# ── List competitors ────────────────────────────────────────────


@router.get("/", response_model=CompetitorListResponse)
async def list_competitors(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)

    competitors_list = project.competitors or []

    # Count discovered entities (pending + confirmed, excluding aliases)
    discovered_q = await db.execute(
        select(func.count(DiscoveredEntity.id)).where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.tenant_id == tenant_id,
            DiscoveredEntity.status.in_(["pending", "confirmed"]),
            DiscoveredEntity.alias_of_id.is_(None),
        )
    )
    discovered_count = discovered_q.scalar() or 0

    # --- Brand info ---
    brand_name = project.brand_name or project.name
    brand_aliases = project.brand_aliases or []

    # Count brand mentions from snapshots (join LlmQuery for project_id)
    brand_mention_q = await db.execute(
        select(func.count(LlmSnapshot.id))
        .join(LlmQuery, LlmQuery.id == LlmSnapshot.llm_query_id)
        .where(
            LlmQuery.project_id == project_id,
            LlmSnapshot.tenant_id == tenant_id,
            LlmSnapshot.brand_mentioned.is_(True),
        )
    )
    brand_mention_count = brand_mention_q.scalar() or 0

    # Entity-based aliases for the brand
    brand_entity = await _find_entity(db, project_id, brand_name)
    if brand_entity:
        brand_entity_aliases = await _build_alias_info_list(db, brand_entity.id)
    else:
        brand_entity_aliases = []

    # Mention counts: use snapshot-based count as "own" for brand
    # (DiscoveredEntity.mention_count may be 0 for manually-added brands)
    brand_own = brand_mention_count
    brand_with_aliases = brand_own + _sum_alias_mentions(brand_entity_aliases)

    # Total: brand + aliases + all brand sub-brands with their aliases
    brand_total = brand_with_aliases
    brand_subs = project.brand_sub_brands or {}
    for sb_name in brand_subs:
        # Snapshot-based count for brand sub-brand (join LlmQuery for project_id)
        sb_key = f"__brand__::{sb_name}"
        sb_q = await db.execute(
            select(func.count(LlmSnapshot.id))
            .join(LlmQuery, LlmQuery.id == LlmSnapshot.llm_query_id)
            .where(
                LlmQuery.project_id == project_id,
                LlmSnapshot.tenant_id == tenant_id,
                LlmSnapshot.competitor_mentions[sb_key].as_string() == "true",
            )
        )
        sb_snap_count = sb_q.scalar() or 0
        sb_entity = await _find_entity(db, project_id, sb_name)
        sb_aliases = await _build_alias_info_list(db, sb_entity.id) if sb_entity else []
        brand_total += sb_snap_count + _sum_alias_mentions(sb_aliases)

    brand_info = BrandInfo(
        name=brand_name,
        mention_count=brand_mention_count,
        aliases=brand_aliases,
        entity_aliases=brand_entity_aliases,
        entity_mention_count=brand_own,
        entity_mention_count_with_aliases=brand_with_aliases,
        entity_mention_count_total=brand_total,
    )

    # --- Competitor info with snapshot-based mentions and entity-based aliases ---
    items = []
    for name in competitors_list:
        # Mention count from snapshots (join LlmQuery for project_id)
        comp_mention_q = await db.execute(
            select(func.count(LlmSnapshot.id))
            .join(LlmQuery, LlmQuery.id == LlmSnapshot.llm_query_id)
            .where(
                LlmQuery.project_id == project_id,
                LlmSnapshot.tenant_id == tenant_id,
                LlmSnapshot.competitor_mentions[name].as_string() == "true",
            )
        )
        mention_count = comp_mention_q.scalar() or 0

        entity = await _find_entity(db, project_id, name)
        if entity:
            aliases = await _build_alias_info_list(db, entity.id)
        else:
            aliases = []

        # Mention counts: use snapshot-based count as "own" for competitor
        # (DiscoveredEntity.mention_count may be 0 for manually-added competitors)
        comp_own = mention_count
        comp_with_aliases = comp_own + _sum_alias_mentions(aliases)

        # Total: competitor + aliases + all its sub-brands with their aliases
        comp_total = comp_with_aliases
        comp_subs_dict = (project.competitor_sub_brands or {}).get(name, {})
        if isinstance(comp_subs_dict, dict):
            for sb_name in comp_subs_dict:
                # Snapshot-based count for competitor sub-brand (join LlmQuery for project_id)
                sb_key = f"{name}::{sb_name}"
                sb_q = await db.execute(
                    select(func.count(LlmSnapshot.id))
                    .join(LlmQuery, LlmQuery.id == LlmSnapshot.llm_query_id)
                    .where(
                        LlmQuery.project_id == project_id,
                        LlmSnapshot.tenant_id == tenant_id,
                        LlmSnapshot.competitor_mentions[sb_key].as_string() == "true",
                    )
                )
                sb_snap_count = sb_q.scalar() or 0
                sb_entity = await _find_entity(db, project_id, sb_name)
                sb_aliases = await _build_alias_info_list(db, sb_entity.id) if sb_entity else []
                comp_total += sb_snap_count + _sum_alias_mentions(sb_aliases)

        items.append(
            CompetitorInfo(
                name=name,
                mention_count=mention_count,
                aliases=aliases,
                entity_mention_count=comp_own,
                entity_mention_count_with_aliases=comp_with_aliases,
                entity_mention_count_total=comp_total,
            )
        )

    return CompetitorListResponse(brand=brand_info, competitors=items, discovered_count=discovered_count)


# ── Add competitor ──────────────────────────────────────────────


@router.post("/", response_model=CompetitorListResponse, status_code=201)
async def add_competitor(
    project_id: int,
    body: AddCompetitorRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    import logging as _logging

    _log = _logging.getLogger(__name__)

    project = await _get_project(project_id, tenant_id, db)

    current = project.competitors or []
    _log.info("[add-competitor] project=%d adding=%r current(%d)=%s", project_id, body.name, len(current), current)
    if body.name in current:
        _log.info("[add-competitor] %r already in list, skipping", body.name)
        return await list_competitors(project_id, db, tenant_id)

    current.append(body.name)
    project.competitors = current
    flag_modified(project, "competitors")
    await db.flush()

    # Verify it was saved
    await db.refresh(project)
    _log.info("[add-competitor] After save: competitors(%d)=%s", len(project.competitors or []), project.competitors)

    # If entity exists in discovered — mark as promoted
    await db.execute(
        update(DiscoveredEntity)
        .where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.entity_name == body.name,
        )
        .values(status="promoted")
    )
    await db.commit()

    return await list_competitors(project_id, db, tenant_id)


# ── Delete competitor ───────────────────────────────────────────


@router.delete("/{name}")
async def delete_competitor(
    project_id: int,
    name: str,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)

    current = project.competitors or []
    if name not in current:
        raise NotFoundError("Competitor not found")

    current.remove(name)
    project.competitors = current
    flag_modified(project, "competitors")

    # Ensure entity exists (onboarding competitors may lack a DiscoveredEntity)
    entity = await _find_entity(db, project_id, name)
    if not entity:
        entity = await _ensure_entity(db, project_id, tenant_id, name, status="confirmed")
    else:
        # Release aliases and demote
        await _release_entity_aliases(db, project_id, entity.id)
        entity.status = "confirmed"

    await db.commit()

    return {"deleted": name}


# ── Discovered entities ─────────────────────────────────────────


@router.get("/discovered")
async def list_discovered(
    project_id: int,
    status: str | None = Query(None, pattern=r"^(pending|confirmed|rejected|promoted|alias)$"),
    min_mentions: int = Query(1, ge=1),
    search: str | None = Query(None),
    suggested_only: bool = Query(False),
    limit: int = Query(30, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    where = (DiscoveredEntity.project_id == project_id) & (DiscoveredEntity.tenant_id == tenant_id)

    # Hide aliased entities by default (they appear under their parent)
    where = where & (DiscoveredEntity.alias_of_id.is_(None))

    if status:
        where = where & (DiscoveredEntity.status == status)
    else:
        # By default, show pending + confirmed
        where = where & (DiscoveredEntity.status.in_(["pending", "confirmed"]))

    if suggested_only:
        where = where & (DiscoveredEntity.suggested_parent.isnot(None))

    if min_mentions > 1:
        where = where & (DiscoveredEntity.mention_count >= min_mentions)

    if search:
        where = where & (DiscoveredEntity.entity_name.ilike(f"%{search}%"))

    # Get total count
    count_result = await db.execute(select(func.count()).select_from(DiscoveredEntity).where(where))
    total = count_result.scalar() or 0

    result = await db.execute(
        select(DiscoveredEntity)
        .where(where)
        .order_by(DiscoveredEntity.mention_count.desc(), DiscoveredEntity.last_seen.desc())
        .limit(limit)
        .offset(offset)
    )
    entities = result.scalars().all()

    return {
        "items": [DiscoveredEntityResponse.model_validate(e) for e in entities],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ── Verify entities via LLM ────────────────────────────────────


@router.post("/verify", response_model=VerifyResponse)
async def verify_entities(
    project_id: int,
    body: VerifyRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    import logging as _logging

    _log = _logging.getLogger(__name__)

    project = await _get_project(project_id, tenant_id, db)

    # Direct SQL to bypass any ORM caching — guarantees fresh data
    from sqlalchemy import text

    raw_q = await db.execute(
        text("SELECT competitors FROM projects WHERE id = :pid AND tenant_id = :tid"),
        {"pid": project_id, "tid": str(tenant_id)},
    )
    raw_row = raw_q.fetchone()
    competitors_list = (raw_row[0] if raw_row and raw_row[0] else []) or []
    _log.info(
        "[verify-endpoint] project=%d entities=%s competitors(%d)=%s",
        project_id,
        body.entities[:5],
        len(competitors_list),
        competitors_list,
    )

    # Collect all sub-brand names for alias detection
    brand_subs = project.brand_sub_brands or {}
    comp_subs = project.competitor_sub_brands or {}
    all_sub_brands: list[str] = list(brand_subs.keys())
    for _comp_name, subs_dict in comp_subs.items():
        if isinstance(subs_dict, dict):
            all_sub_brands.extend(subs_dict.keys())

    # Import here to avoid circular imports
    from app.services.competitor_verification import verify_competitors

    try:
        results = await verify_competitors(
            brand_name=project.brand_name or project.name,
            brand_description=project.brand_description or "",
            categories=project.categories or [],
            entities=body.entities,
            tenant_id=tenant_id,
            db=db,
            competitors=competitors_list,
            sub_brands=all_sub_brands,
        )
    except Exception as exc:
        _log.exception("[verify-endpoint] verify_competitors failed: %s", exc)
        raise

    _log.info("[verify-endpoint] Got %d results", len(results))

    # NOTE: We do NOT auto-apply statuses anymore.
    # Results are returned to the UI; the user decides per-entity via /verify-apply.
    return VerifyResponse(results=results)


@router.post("/verify-apply")
async def verify_apply(
    project_id: int,
    body: VerifyApplyRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Apply user decisions on verification results.

    Actions:
    - accept: agree with LLM verdict, save explanation, update status
    - accept_and_add: agree + immediately promote as competitor or add as sub-brand
    - disagree: discard LLM verdict, keep entity as-is (no explanation saved)
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)

    project = await _get_project(project_id, tenant_id, db)
    now = datetime.now(timezone.utc)

    promoted_names = []
    sub_brand_added = []
    alias_added = []

    for d in body.decisions:
        if d.action == "disagree":
            # User disagrees — don't save anything, entity stays as-is
            _log.info("[verify-apply] disagree for %s", d.entity)
            continue

        # "accept" or "accept_and_add" — save LLM explanation and update status
        if d.is_alias and d.alias_of:
            new_status = "confirmed"
            suggested_parent = None
        elif d.is_sub_brand and d.parent_brand:
            new_status = "confirmed"
            suggested_parent = d.parent_brand
        elif d.is_competitor:
            new_status = "confirmed"
            suggested_parent = None
        else:
            new_status = "rejected"
            suggested_parent = None

        await db.execute(
            update(DiscoveredEntity)
            .where(
                DiscoveredEntity.project_id == project_id,
                DiscoveredEntity.entity_name == d.entity,
            )
            .values(
                status=new_status,
                verified_at=now,
                verified_by="llm",
                suggested_parent=suggested_parent,
                llm_explanation=d.explanation if d.explanation else None,
            )
        )

        if d.action == "accept_and_add":
            if d.is_alias and d.alias_of:
                # Add as alias via set-alias logic
                target = await _find_entity(db, project_id, d.alias_of)
                if not target:
                    # Target might be brand name — ensure entity exists
                    target = await _ensure_entity(db, project_id, tenant_id, d.alias_of, status="promoted")
                source = await _ensure_entity(db, project_id, tenant_id, d.entity, status="alias")
                source.alias_of_id = target.id
                source.status = "alias"
                alias_added.append(d.entity)
            elif d.is_competitor:
                # Promote to competitor
                comps = list(project.competitors or [])
                if d.entity not in comps:
                    comps.append(d.entity)
                    project.competitors = comps
                    flag_modified(project, "competitors")
                # Also mark as promoted
                await db.execute(
                    update(DiscoveredEntity)
                    .where(
                        DiscoveredEntity.project_id == project_id,
                        DiscoveredEntity.entity_name == d.entity,
                    )
                    .values(status="promoted")
                )
                promoted_names.append(d.entity)
            elif d.is_sub_brand and d.parent_brand:
                # Add as sub-brand (store [] in JSONB, aliases are entity-based)
                brand_name = project.brand_name or project.name
                if d.parent_brand == brand_name:
                    subs = dict(project.brand_sub_brands or {})
                    if d.entity not in subs:
                        subs[d.entity] = []
                    project.brand_sub_brands = subs
                    flag_modified(project, "brand_sub_brands")
                else:
                    comp_subs = dict(project.competitor_sub_brands or {})
                    parent_subs = dict(comp_subs.get(d.parent_brand, {}))
                    if d.entity not in parent_subs:
                        parent_subs[d.entity] = []
                    comp_subs[d.parent_brand] = parent_subs
                    project.competitor_sub_brands = comp_subs
                    flag_modified(project, "competitor_sub_brands")
                # Mark entity as promoted so it disappears from discovered
                await db.execute(
                    update(DiscoveredEntity)
                    .where(
                        DiscoveredEntity.project_id == project_id,
                        DiscoveredEntity.entity_name == d.entity,
                    )
                    .values(status="promoted")
                )
                sub_brand_added.append(d.entity)

    await db.commit()

    return {
        "applied": len([d for d in body.decisions if d.action != "disagree"]),
        "disagreed": len([d for d in body.decisions if d.action == "disagree"]),
        "promoted": promoted_names,
        "sub_brands_added": sub_brand_added,
        "aliases_added": alias_added,
    }


# ── Promote entities to competitors ─────────────────────────────


@router.post("/promote")
async def promote_entities(
    project_id: int,
    body: PromoteRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)

    current = project.competitors or []
    added = []
    for name in body.entities:
        if name not in current:
            current.append(name)
            added.append(name)

    project.competitors = current
    flag_modified(project, "competitors")
    await db.flush()

    # Update discovered entities status
    if added:
        await db.execute(
            update(DiscoveredEntity)
            .where(
                DiscoveredEntity.project_id == project_id,
                DiscoveredEntity.entity_name.in_(added),
            )
            .values(status="promoted")
        )
    await db.commit()

    return {"promoted": added}


# ── Reject entities ─────────────────────────────────────────────


@router.post("/reject")
async def reject_entities(
    project_id: int,
    body: RejectRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    await _get_project(project_id, tenant_id, db)

    result = await db.execute(
        update(DiscoveredEntity)
        .where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.entity_name.in_(body.entities),
        )
        .values(status="rejected")
    )
    await db.commit()

    return {"rejected": result.rowcount}


# ── Unreject (reset rejected → pending) ──────────────────────


@router.post("/unreject")
async def unreject_entities(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Reset ALL rejected entities back to pending status."""
    await _get_project(project_id, tenant_id, db)

    result = await db.execute(
        update(DiscoveredEntity)
        .where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.tenant_id == tenant_id,
            DiscoveredEntity.status == "rejected",
        )
        .values(status="pending", verified_at=None, verified_by=None)
    )
    await db.commit()

    return {"unrejected": result.rowcount}


# ── Reset all sub-brands ──────────────────────────────────────


@router.post("/reset-sub-brands")
async def reset_sub_brands(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Remove ALL sub-brands and return their entities to discovered (confirmed).

    - Collects all sub-brand names from brand_sub_brands + competitor_sub_brands
    - Releases all entity aliases (clears alias_of_id)
    - Sets their discovered entity status back to 'confirmed'
    - Clears suggested_parent on ALL entities
    - Empties brand_sub_brands and competitor_sub_brands in the project
    """
    project = await _get_project(project_id, tenant_id, db)

    # Collect all sub-brand names
    all_names: list[str] = []
    brand_subs = project.brand_sub_brands or {}
    comp_subs = project.competitor_sub_brands or {}

    all_names.extend(brand_subs.keys())
    for comp_name, subs_dict in comp_subs.items():
        if isinstance(subs_dict, dict):
            all_names.extend(subs_dict.keys())

    restored = 0
    if all_names:
        # Get sub-brand entity IDs to release their aliases
        sub_ids_q = await db.execute(
            select(DiscoveredEntity.id).where(
                DiscoveredEntity.project_id == project_id,
                DiscoveredEntity.entity_name.in_(all_names),
            )
        )
        sub_ids = [r[0] for r in sub_ids_q.fetchall()]

        # Release aliases of all sub-brands
        if sub_ids:
            await db.execute(
                update(DiscoveredEntity)
                .where(
                    DiscoveredEntity.project_id == project_id,
                    DiscoveredEntity.alias_of_id.in_(sub_ids),
                )
                .values(alias_of_id=None, status="confirmed")
            )

        # Reset sub-brand entities status back to confirmed
        result = await db.execute(
            update(DiscoveredEntity)
            .where(
                DiscoveredEntity.project_id == project_id,
                DiscoveredEntity.tenant_id == tenant_id,
                DiscoveredEntity.entity_name.in_(all_names),
            )
            .values(status="confirmed")
        )
        restored = result.rowcount

    # Clear suggested_parent on ALL entities
    await db.execute(
        update(DiscoveredEntity)
        .where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.tenant_id == tenant_id,
        )
        .values(suggested_parent=None)
    )

    # Empty project sub-brand fields
    project.brand_sub_brands = {}
    project.competitor_sub_brands = {}
    flag_modified(project, "brand_sub_brands")
    flag_modified(project, "competitor_sub_brands")
    await db.commit()

    return {
        "reset": True,
        "sub_brand_names": all_names,
        "entities_restored": restored,
    }


# ── Sync entity statuses ──────────────────────────────────────


@router.post("/sync-statuses")
async def sync_entity_statuses(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Sync discovered entity statuses with actual competitors/sub-brands.

    Ensures every competitor, sub-brand, and the brand itself has a
    DiscoveredEntity record (creates one if missing).
    Marks them as 'promoted'. Does not override 'alias' status.
    Also normalises Unicode dash variants in entity names (U+2011, U+2013, etc.)
    to ASCII hyphen-minus so that visually identical names don't create duplicates.
    """
    import re as _re

    project = await _get_project(project_id, tenant_id, db)

    # --- Phase 0: normalise Unicode dashes in existing entity names ----------
    _dash_re = _re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00AD\uFE58\uFE63\uFF0D]")
    all_entities_q = await db.execute(
        select(DiscoveredEntity).where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.tenant_id == tenant_id,
        )
    )
    all_entities = all_entities_q.scalars().all()

    # Find entities whose names contain non-ASCII dashes
    renamed = 0
    merged_away = 0
    for entity in all_entities:
        normalised = _dash_re.sub("-", entity.entity_name)
        if normalised != entity.entity_name:
            # Check if a canonical entity with the normalised name already exists
            canonical_q = await db.execute(
                select(DiscoveredEntity).where(
                    DiscoveredEntity.project_id == project_id,
                    DiscoveredEntity.tenant_id == tenant_id,
                    DiscoveredEntity.entity_name == normalised,
                    DiscoveredEntity.id != entity.id,
                )
            )
            canonical = canonical_q.scalar_one_or_none()
            if canonical:
                # Merge: add mentions to canonical, alias the non-ASCII variant
                canonical.mention_count = (canonical.mention_count or 0) + (entity.mention_count or 0)
                if entity.last_seen and (not canonical.last_seen or entity.last_seen > canonical.last_seen):
                    canonical.last_seen = entity.last_seen
                entity.alias_of_id = canonical.id
                entity.status = "alias"
                merged_away += 1
            else:
                # No canonical exists — just rename in place
                entity.entity_name = normalised
                renamed += 1

    if renamed or merged_away:
        await db.flush()

    # Collect all names that should be "promoted" — brand + competitors + sub-brands
    promoted_names: list[str] = []

    # Include the brand itself so it can receive entity-based aliases
    brand_name = project.brand_name or project.name
    if brand_name:
        promoted_names.append(brand_name)

    promoted_names.extend(project.competitors or [])

    brand_subs = project.brand_sub_brands or {}
    promoted_names.extend(brand_subs.keys())

    comp_subs = project.competitor_sub_brands or {}
    for comp_name, subs_dict in comp_subs.items():
        if isinstance(subs_dict, dict):
            promoted_names.extend(subs_dict.keys())

    updated = 0
    created = 0
    if promoted_names:
        # Update existing entities
        result = await db.execute(
            update(DiscoveredEntity)
            .where(
                DiscoveredEntity.project_id == project_id,
                DiscoveredEntity.tenant_id == tenant_id,
                DiscoveredEntity.entity_name.in_(promoted_names),
                DiscoveredEntity.status.notin_(["promoted", "alias"]),
            )
            .values(status="promoted")
        )
        updated = result.rowcount

        # Create missing entities (e.g. onboarding competitors that the
        # consolidator never stored because they matched known names)
        existing_q = await db.execute(
            select(DiscoveredEntity.entity_name).where(
                DiscoveredEntity.project_id == project_id,
                DiscoveredEntity.entity_name.in_(promoted_names),
            )
        )
        existing_names = {row[0] for row in existing_q.fetchall()}
        missing = set(promoted_names) - existing_names
        if missing:
            today = date.today()
            for name in missing:
                db.add(
                    DiscoveredEntity(
                        tenant_id=tenant_id,
                        project_id=project_id,
                        entity_name=name,
                        mention_count=0,
                        first_seen=today,
                        last_seen=today,
                        status="promoted",
                    )
                )
            created = len(missing)

    await db.commit()

    return {
        "synced": True,
        "total_promoted_names": len(set(promoted_names)),
        "entities_updated": updated,
        "entities_created": created,
        "dash_normalised": renamed,
        "dash_merged": merged_away,
    }


# ── Brand sub-brands ────────────────────────────────────────────


@router.get("/brand-sub-brands", response_model=list[SubBrandResponse])
async def list_brand_sub_brands(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)
    subs = project.brand_sub_brands or {}
    items = []
    for name in subs:
        # Mention count from snapshots (sub-brands use "__brand__::SubName" key, join LlmQuery for project_id)
        sb_key = f"__brand__::{name}"
        sb_mention_q = await db.execute(
            select(func.count(LlmSnapshot.id))
            .join(LlmQuery, LlmQuery.id == LlmSnapshot.llm_query_id)
            .where(
                LlmQuery.project_id == project_id,
                LlmSnapshot.tenant_id == tenant_id,
                LlmSnapshot.competitor_mentions[sb_key].as_string() == "true",
            )
        )
        mention_count = sb_mention_q.scalar() or 0

        entity = await _find_entity(db, project_id, name)
        if entity:
            aliases = await _build_alias_info_list(db, entity.id)
        else:
            aliases = []

        # Use snapshot-based count as "own", add alias mentions on top
        sb_own = mention_count
        sb_with_aliases = sb_own + _sum_alias_mentions(aliases)
        items.append(
            SubBrandResponse(
                name=name,
                mention_count=mention_count,
                aliases=aliases,
                entity_mention_count=sb_own,
                entity_mention_count_with_aliases=sb_with_aliases,
            )
        )
    return items


@router.post("/brand-sub-brands", response_model=list[SubBrandResponse], status_code=201)
async def add_brand_sub_brand(
    project_id: int,
    body: SubBrandCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)
    subs = dict(project.brand_sub_brands or {})

    # Store only the key in JSONB (aliases are entity-based relationships now)
    subs[body.name] = []
    project.brand_sub_brands = subs
    flag_modified(project, "brand_sub_brands")

    # Ensure sub-brand entity exists and is promoted
    sub_entity = await _ensure_entity(db, project_id, tenant_id, body.name, status="promoted")

    # Create alias entities if provided
    for alias_name in body.aliases:
        alias_entity = await _ensure_entity(db, project_id, tenant_id, alias_name, status="alias")
        alias_entity.alias_of_id = sub_entity.id
        alias_entity.status = "alias"

    await db.commit()
    return await list_brand_sub_brands(project_id, db, tenant_id)


@router.delete("/brand-sub-brands/{name}")
async def delete_brand_sub_brand(
    project_id: int,
    name: str,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)
    subs = dict(project.brand_sub_brands or {})
    if name not in subs:
        raise NotFoundError("Sub-brand not found")
    del subs[name]
    project.brand_sub_brands = subs
    flag_modified(project, "brand_sub_brands")

    # Find sub-brand entity and release its aliases
    entity = await _find_entity(db, project_id, name)
    if entity:
        await _release_entity_aliases(db, project_id, entity.id)
        # Reset entity status from "promoted" back to "confirmed"
        if entity.status == "promoted":
            entity.status = "confirmed"

    await db.commit()
    return {"deleted": name}


# ── Competitor sub-brands ───────────────────────────────────────


@router.get("/{competitor_name}/sub-brands", response_model=list[SubBrandResponse])
async def list_competitor_sub_brands(
    project_id: int,
    competitor_name: str,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)
    competitors = project.competitors or []
    if competitor_name not in competitors:
        raise NotFoundError("Competitor not found")
    comp_subs = (project.competitor_sub_brands or {}).get(competitor_name, {})
    if not isinstance(comp_subs, dict):
        comp_subs = {}
    items = []
    for name in comp_subs:
        # Mention count from snapshots (competitor sub-brands use "CompName::SubName" key, join LlmQuery for project_id)
        sb_key = f"{competitor_name}::{name}"
        sb_mention_q = await db.execute(
            select(func.count(LlmSnapshot.id))
            .join(LlmQuery, LlmQuery.id == LlmSnapshot.llm_query_id)
            .where(
                LlmQuery.project_id == project_id,
                LlmSnapshot.tenant_id == tenant_id,
                LlmSnapshot.competitor_mentions[sb_key].as_string() == "true",
            )
        )
        mention_count = sb_mention_q.scalar() or 0

        entity = await _find_entity(db, project_id, name)
        if entity:
            aliases = await _build_alias_info_list(db, entity.id)
        else:
            aliases = []

        # Use snapshot-based count as "own", add alias mentions on top
        sb_own = mention_count
        sb_with_aliases = sb_own + _sum_alias_mentions(aliases)
        items.append(
            SubBrandResponse(
                name=name,
                mention_count=mention_count,
                aliases=aliases,
                entity_mention_count=sb_own,
                entity_mention_count_with_aliases=sb_with_aliases,
            )
        )
    return items


@router.post("/{competitor_name}/sub-brands", response_model=list[SubBrandResponse], status_code=201)
async def add_competitor_sub_brand(
    project_id: int,
    competitor_name: str,
    body: SubBrandCreate,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)
    competitors = project.competitors or []
    if competitor_name not in competitors:
        raise NotFoundError("Competitor not found")

    all_comp_subs = dict(project.competitor_sub_brands or {})
    comp_subs = dict(all_comp_subs.get(competitor_name, {}))

    # Store only the key in JSONB (aliases are entity-based relationships now)
    comp_subs[body.name] = []
    all_comp_subs[competitor_name] = comp_subs
    project.competitor_sub_brands = all_comp_subs
    flag_modified(project, "competitor_sub_brands")

    # Ensure sub-brand entity exists and is promoted
    sub_entity = await _ensure_entity(db, project_id, tenant_id, body.name, status="promoted")

    # Create alias entities if provided
    for alias_name in body.aliases:
        alias_entity = await _ensure_entity(db, project_id, tenant_id, alias_name, status="alias")
        alias_entity.alias_of_id = sub_entity.id
        alias_entity.status = "alias"

    await db.commit()
    return await list_competitor_sub_brands(project_id, competitor_name, db, tenant_id)


@router.delete("/{competitor_name}/sub-brands/{name}")
async def delete_competitor_sub_brand(
    project_id: int,
    competitor_name: str,
    name: str,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    project = await _get_project(project_id, tenant_id, db)
    competitors = project.competitors or []
    if competitor_name not in competitors:
        raise NotFoundError("Competitor not found")

    all_comp_subs = dict(project.competitor_sub_brands or {})
    comp_subs = dict(all_comp_subs.get(competitor_name, {}))
    if name not in comp_subs:
        raise NotFoundError("Sub-brand not found")
    del comp_subs[name]
    all_comp_subs[competitor_name] = comp_subs
    project.competitor_sub_brands = all_comp_subs
    flag_modified(project, "competitor_sub_brands")

    # Find sub-brand entity and release its aliases
    entity = await _find_entity(db, project_id, name)
    if entity:
        await _release_entity_aliases(db, project_id, entity.id)
        # Reset entity status from "promoted" back to "confirmed"
        if entity.status == "promoted":
            entity.status = "confirmed"

    await db.commit()
    return {"deleted": name}


# ── Move sub-brand between parents ────────────────────────────


@router.post("/move-sub-brand")
async def move_sub_brand(
    project_id: int,
    body: dict,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Move a sub-brand from one parent to another.

    Body: {"name": "parkside", "from_parent": "Кортрос", "to_parent": "ЛСР"}
    from_parent / to_parent can be the brand name (stored in brand_sub_brands)
    or a competitor name (stored in competitor_sub_brands).

    Entity aliases (alias_of_id) automatically follow the sub-brand entity.
    """
    project = await _get_project(project_id, tenant_id, db)
    name = body.get("name", "")
    from_parent = body.get("from_parent", "")
    to_parent = body.get("to_parent", "")

    if not name or not from_parent or not to_parent:
        raise BadRequestError("name, from_parent, and to_parent are required")
    if from_parent == to_parent:
        return {"moved": name, "from": from_parent, "to": to_parent}

    brand_name = project.brand_name or project.name
    brand_subs = dict(project.brand_sub_brands or {})
    comp_subs = dict(project.competitor_sub_brands or {})

    # --- Remove from old parent ---
    if from_parent == brand_name:
        if name in brand_subs:
            brand_subs.pop(name)
        else:
            raise NotFoundError(f"Sub-brand '{name}' not found under '{from_parent}'")
    else:
        parent_dict = dict(comp_subs.get(from_parent, {}))
        if name in parent_dict:
            parent_dict.pop(name)
            comp_subs[from_parent] = parent_dict
        else:
            raise NotFoundError(f"Sub-brand '{name}' not found under '{from_parent}'")

    # --- Add to new parent (JSONB key only, aliases are entity-based) ---
    if to_parent == brand_name:
        brand_subs[name] = []
    else:
        if to_parent not in (project.competitors or []):
            raise NotFoundError(f"Competitor '{to_parent}' not found")
        parent_dict = dict(comp_subs.get(to_parent, {}))
        parent_dict[name] = []
        comp_subs[to_parent] = parent_dict

    project.brand_sub_brands = brand_subs
    project.competitor_sub_brands = comp_subs
    flag_modified(project, "brand_sub_brands")
    flag_modified(project, "competitor_sub_brands")
    await db.commit()
    return {"moved": name, "from": from_parent, "to": to_parent}


# ── Merge brand (entity-to-entity alias) ──────────────────────


@router.post("/merge-brand")
async def merge_brand(
    project_id: int,
    body: MergeBrandRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Merge source brand/competitor into target.

    Source becomes an alias of target via alias_of_id (entity-to-entity relationship).
    Source's sub-brands are moved to target.
    Source is removed from competitors list.
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)

    project = await _get_project(project_id, tenant_id, db)
    brand_name = project.brand_name or project.name
    competitors_list = list(project.competitors or [])
    comp_subs = dict(project.competitor_sub_brands or {})
    brand_subs = dict(project.brand_sub_brands or {})

    source = body.source
    target = body.target

    if source == target:
        raise BadRequestError("Source and target must be different")

    # Validate source exists (must be a competitor or brand sub-brand)
    source_is_competitor = source in competitors_list
    source_is_brand_sub = source in brand_subs
    if not source_is_competitor and not source_is_brand_sub:
        raise NotFoundError(f"Source '{source}' is not a competitor or brand sub-brand")

    # Validate target exists (must be a competitor or the brand itself)
    target_is_brand = target == brand_name
    target_is_competitor = target in competitors_list
    if not target_is_brand and not target_is_competitor:
        raise NotFoundError(f"Target '{target}' is not the brand or a competitor")

    # Ensure both entities exist in discovered_entities
    target_entity = await _ensure_entity(db, project_id, tenant_id, target, status="promoted")
    source_entity = await _ensure_entity(db, project_id, tenant_id, source, status="alias")

    # Set source as alias of target (entity-to-entity relationship)
    source_entity.alias_of_id = target_entity.id
    source_entity.status = "alias"

    # Re-point source's own aliases to target
    await db.execute(
        update(DiscoveredEntity)
        .where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.alias_of_id == source_entity.id,
        )
        .values(alias_of_id=target_entity.id)
    )

    # Move source's sub-brands to target in JSONB
    if source_is_competitor:
        source_sub_brands = comp_subs.pop(source, {})
        if isinstance(source_sub_brands, dict):
            if target_is_brand:
                for sb_name in source_sub_brands:
                    brand_subs[sb_name] = []
            else:
                target_sub_brands = dict(comp_subs.get(target, {}))
                for sb_name in source_sub_brands:
                    target_sub_brands[sb_name] = []
                comp_subs[target] = target_sub_brands
    elif source_is_brand_sub:
        # Source was a brand sub-brand — remove from brand_sub_brands
        del brand_subs[source]

    # Remove source from competitors list
    if source_is_competitor:
        competitors_list.remove(source)
        project.competitors = competitors_list
        flag_modified(project, "competitors")

    project.brand_sub_brands = brand_subs
    project.competitor_sub_brands = comp_subs
    flag_modified(project, "brand_sub_brands")
    flag_modified(project, "competitor_sub_brands")

    await db.commit()

    _log.info(
        "[merge-brand] project=%d source=%r -> target=%r (alias_of_id=%d)", project_id, source, target, target_entity.id
    )

    return {
        "merged": True,
        "source": source,
        "target": target,
    }


# ── Entity alias management ───────────────────────────────────


@router.post("/set-alias")
async def set_alias(
    project_id: int,
    body: SetAliasRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Mark entity_name as alias of target_entity_name.

    Sets alias_of_id and status='alias'. The entity will be hidden
    from the discovered list and its mentions will aggregate to the target.
    """
    await _get_project(project_id, tenant_id, db)

    # Find target entity
    target = await _find_entity(db, project_id, body.target_entity_name)
    if not target:
        raise NotFoundError(f"Target entity '{body.target_entity_name}' not found")
    if target.alias_of_id is not None:
        raise BadRequestError(f"Target '{body.target_entity_name}' is itself an alias. Cannot alias to an alias.")

    # Find or create source entity
    source = await _ensure_entity(db, project_id, tenant_id, body.entity_name, status="alias")
    source.alias_of_id = target.id
    source.status = "alias"

    await db.commit()
    return {"entity": body.entity_name, "alias_of": body.target_entity_name}


@router.post("/remove-alias")
async def remove_alias(
    project_id: int,
    body: RemoveAliasRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Remove alias relationship, returning entity to standalone confirmed status."""
    await _get_project(project_id, tenant_id, db)

    entity = await _find_entity(db, project_id, body.entity_name)
    if not entity:
        raise NotFoundError(f"Entity '{body.entity_name}' not found")
    if entity.alias_of_id is None:
        raise BadRequestError(f"Entity '{body.entity_name}' is not an alias")

    entity.alias_of_id = None
    entity.status = "confirmed"

    await db.commit()
    return {"entity": body.entity_name, "status": "confirmed"}


@router.post("/reset-entity-aliases")
async def reset_entity_aliases(
    project_id: int,
    body: ResetEntityAliasesRequest,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Bulk-release ALL aliases of a specific entity (brand, competitor, or sub-brand).

    The entity itself is NOT removed or modified — only its aliases are
    detached (alias_of_id cleared, status set to 'confirmed').
    """
    await _get_project(project_id, tenant_id, db)

    entity = await _find_entity(db, project_id, body.entity_name)
    if not entity:
        raise NotFoundError(f"Entity '{body.entity_name}' not found")

    released = await _release_entity_aliases(db, project_id, entity.id)

    await db.commit()
    return {
        "entity_name": body.entity_name,
        "aliases_released": released,
    }


# ── Auto-detect sub-brands via LLM ─────────────────────────────


@router.post("/detect-sub-brands")
async def detect_sub_brands_endpoint(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Ask LLM to identify potential sub-brands among discovered entities."""
    project = await _get_project(project_id, tenant_id, db)

    # Get Gemini API key from tenant
    from app.core.encryption import decrypt_value

    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise NotFoundError("Tenant not found")

    encrypted_key = getattr(tenant, "gemini_api_key", None)
    gemini_api_key = decrypt_value(encrypted_key) if encrypted_key else None
    if not gemini_api_key:
        return {"error": "Gemini API key not configured", "suggestions": []}

    from app.analysis.entity_consolidator import detect_sub_brands

    suggestions = await detect_sub_brands(
        db=db,
        project_id=project_id,
        tenant_id=tenant_id,
        brand_name=project.brand_name or project.name,
        competitors=project.competitors or [],
        gemini_api_key=gemini_api_key,
    )

    return {"suggestions": suggestions}


# ── Update suggested parent ───────────────────────────────────


@router.patch("/discovered/{entity_name}/suggested-parent")
async def update_suggested_parent(
    project_id: int,
    entity_name: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Update or clear the suggested_parent for a discovered entity."""
    await _get_project(project_id, tenant_id, db)

    new_parent = body.get("suggested_parent")  # can be None to clear

    stmt = select(DiscoveredEntity).where(
        DiscoveredEntity.project_id == project_id,
        DiscoveredEntity.tenant_id == tenant_id,
        DiscoveredEntity.entity_name == entity_name,
    )
    result = await db.execute(stmt)
    entity = result.scalar_one_or_none()
    if not entity:
        raise NotFoundError("Entity not found")

    entity.suggested_parent = new_parent if new_parent else None
    await db.commit()
    return {"entity_name": entity_name, "suggested_parent": entity.suggested_parent}
