"""LLM Results API — dashboard stats, charts, details, citations."""

import uuid
from collections import defaultdict
from datetime import date, timedelta
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import check_project_access, get_current_tenant_id, get_current_user, require_role
from app.core.encryption import decrypt_value
from app.core.exceptions import NotFoundError
from app.db.postgres import get_db
from app.models.llm_query import LlmQuery
from app.models.llm_snapshot import LlmSnapshot
from app.models.project import Project
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.bi_dashboard import BiDashboardResponse, GeoActionPlan, HeatmapResponse
from app.schemas.llm_debug import (
    LlmDebugResponse,
    PipelineTraceResponse,
    SnapshotWithQueryText,
    TraceCitation,
    TraceCompetitor,
    TraceContext,
    TraceEntityResolution,
    TraceLlmQuery,
    TraceSanitization,
    TraceScoring,
    TraceStructure,
    TraceTargetBrand,
)
from app.schemas.llm_snapshot import (
    LlmChartDataResponse,
    LlmChartDataset,
    LlmCitationItem,
    LlmCitationsResponse,
    LlmDashboardStats,
    LlmSnapshotDetail,
)
from app.services import bi_service

router = APIRouter(prefix="/llm", tags=["llm-results"])


def _category_filter(project_id: int, category: str):
    """Filter LlmQuery by category (scenario name) stored directly on the row."""
    return LlmQuery.category == category


async def _get_project(
    project_id: int, tenant_id: uuid.UUID, db: AsyncSession, user: User | None = None
) -> Project:
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")
    if user is not None:
        await check_project_access(project_id, user, db)
    return project


async def _get_judge_api_key(db: AsyncSession, tenant_id: uuid.UUID) -> str | None:
    """Get the OpenAI API key from tenant for LLM-as-a-Judge calls."""
    result = await db.execute(select(Tenant.openai_api_key).where(Tenant.id == tenant_id))
    encrypted_key = result.scalar_one_or_none()
    if encrypted_key:
        return decrypt_value(encrypted_key) or None
    return None


@router.get("/categories/{project_id}")
async def llm_categories(
    project_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    """Distinct scenario/category names for this project (from llm_queries.category)."""
    await _get_project(project_id, tenant_id, db, user)
    result = await db.execute(
        select(LlmQuery.category)
        .where(
            LlmQuery.project_id == project_id,
            LlmQuery.tenant_id == tenant_id,
            LlmQuery.category.isnot(None),
            LlmQuery.category != "",
        )
        .distinct()
        .order_by(LlmQuery.category)
    )
    return {"categories": [r[0] for r in result.all()]}


@router.get("/dashboard/{project_id}", response_model=LlmDashboardStats)
async def llm_dashboard(
    project_id: int,
    days: int = Query(30, ge=1, le=365),
    category: str | None = Query(None),
    detail_level: str | None = Query(None, pattern=r"^(rollup|detailed)$"),
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    """Aggregated LLM visibility stats for a project."""
    project = await _get_project(project_id, tenant_id, db, user)

    # Build a set of all sub-brand names (brand + competitor sub-brands)
    # so we can exclude them from the top-level SOV chart.
    # These may exist as plain keys in older snapshots (before being moved to sub-brands).
    sub_brand_names: set[str] = set()
    for sb_name in project.brand_sub_brands or {}:
        sub_brand_names.add(sb_name)
    for comp_subs in (project.competitor_sub_brands or {}).values():
        for sb_name in comp_subs if isinstance(comp_subs, dict) else {}:
            sub_brand_names.add(sb_name)

    date_from = date.today() - timedelta(days=days)

    # Get all query IDs for this project (optionally filtered by category/scenario)
    query_ids_stmt = select(LlmQuery.id).where(LlmQuery.project_id == project_id, LlmQuery.tenant_id == tenant_id)
    if category:
        query_ids_stmt = query_ids_stmt.where(_category_filter(project_id, category))

    # Aggregate snapshots
    snap_where = and_(
        LlmSnapshot.llm_query_id.in_(query_ids_stmt),
        LlmSnapshot.tenant_id == tenant_id,
        LlmSnapshot.date >= date_from,
    )

    # Total queries and checks
    total_queries_where = [
        LlmQuery.project_id == project_id,
        LlmQuery.tenant_id == tenant_id,
        LlmQuery.is_active == True,  # noqa: E712
    ]
    if category:
        total_queries_where.append(_category_filter(project_id, category))
    total_queries_result = await db.execute(select(func.count()).select_from(LlmQuery).where(*total_queries_where))
    total_queries = total_queries_result.scalar() or 0

    total_checks_result = await db.execute(select(func.count()).select_from(LlmSnapshot).where(snap_where))
    total_checks = total_checks_result.scalar() or 0

    # Brand mention rate overall
    mentioned_count_result = await db.execute(
        select(func.count()).select_from(LlmSnapshot).where(snap_where, LlmSnapshot.brand_mentioned == True)  # noqa: E712
    )
    mentioned_count = mentioned_count_result.scalar() or 0
    brand_mention_rate = mentioned_count / total_checks if total_checks > 0 else 0.0

    # Mention rate by provider
    provider_stats = await db.execute(
        select(
            LlmSnapshot.llm_provider,
            func.count().label("total"),
            func.sum(case((LlmSnapshot.brand_mentioned == True, 1), else_=0)).label("mentioned"),  # noqa: E712
        )
        .where(snap_where)
        .group_by(LlmSnapshot.llm_provider)
    )
    mention_rate_by_provider = {}
    for row in provider_stats:
        mention_rate_by_provider[row.llm_provider] = row.mentioned / row.total if row.total > 0 else 0.0

    # Total cost
    cost_result = await db.execute(select(func.coalesce(func.sum(LlmSnapshot.cost_usd), 0.0)).where(snap_where))
    total_cost = cost_result.scalar() or 0.0

    # SOV calculation: brand mentions / (brand + competitor mentions)
    # Competitor mentions are tracked in competitor_mentions JSONB
    all_snapshots = await db.execute(
        select(LlmSnapshot.brand_mentioned, LlmSnapshot.competitor_mentions, LlmSnapshot.llm_provider).where(snap_where)
    )
    all_snaps = all_snapshots.all()

    competitor_mention_counts: dict[str, int] = defaultdict(int)
    competitor_mention_counts_by_provider: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_brand_mentions = 0
    total_all_mentions = 0
    sov_provider_data: dict[str, dict] = defaultdict(lambda: {"brand": 0, "total": 0})

    for snap in all_snaps:
        if snap.brand_mentioned:
            total_brand_mentions += 1
            total_all_mentions += 1
            sov_provider_data[snap.llm_provider]["brand"] += 1
            sov_provider_data[snap.llm_provider]["total"] += 1

        if snap.competitor_mentions:
            for comp, mentioned in snap.competitor_mentions.items():
                if mentioned:
                    # In rollup mode (default), skip sub-brand :: keys
                    # (parent brand is already True, so no double-counting)
                    if detail_level != "detailed" and "::" in comp:
                        continue
                    # Skip entities that are now sub-brands but stored as
                    # top-level keys in older snapshots (before reclassification)
                    if detail_level != "detailed" and comp in sub_brand_names:
                        continue
                    competitor_mention_counts[comp] += 1
                    competitor_mention_counts_by_provider[snap.llm_provider][comp] += 1
                    total_all_mentions += 1
                    sov_provider_data[snap.llm_provider]["total"] += 1

    sov = total_brand_mentions / total_all_mentions if total_all_mentions > 0 else 0.0
    sov_by_provider = {p: d["brand"] / d["total"] if d["total"] > 0 else 0.0 for p, d in sov_provider_data.items()}
    competitor_sov = dict(
        sorted(
            (
                (c, cnt / total_all_mentions if total_all_mentions > 0 else 0.0)
                for c, cnt in competitor_mention_counts.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )
    )

    # Competitor SOV broken down by provider
    competitor_sov_by_provider: dict[str, dict[str, float]] = {}
    for prov, comp_counts in competitor_mention_counts_by_provider.items():
        prov_total = sov_provider_data[prov]["total"]
        competitor_sov_by_provider[prov] = dict(
            sorted(
                ((c, cnt / prov_total if prov_total > 0 else 0.0) for c, cnt in comp_counts.items()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    # Top cited URLs
    cited_url_counts: dict[str, int] = defaultdict(int)
    cited_snaps = await db.execute(select(LlmSnapshot.cited_urls).where(snap_where, LlmSnapshot.cited_urls.isnot(None)))
    for row in cited_snaps:
        if row.cited_urls:
            for url in row.cited_urls:
                cited_url_counts[url] += 1

    top_cited = sorted(cited_url_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return LlmDashboardStats(
        total_queries=total_queries,
        total_checks=total_checks,
        brand_mention_rate=round(brand_mention_rate, 4),
        mention_rate_by_provider={k: round(v, 4) for k, v in mention_rate_by_provider.items()},
        sov=round(sov, 4),
        sov_by_provider={k: round(v, 4) for k, v in sov_by_provider.items()},
        top_cited_urls=[{"url": url, "count": cnt} for url, cnt in top_cited],
        total_cost_usd=round(total_cost, 4),
        competitor_sov={k: round(v, 4) for k, v in competitor_sov.items()},
        competitor_sov_by_provider={
            prov: {k: round(v, 4) for k, v in comps.items()} for prov, comps in competitor_sov_by_provider.items()
        },
    )


@router.get("/chart/{project_id}", response_model=LlmChartDataResponse)
async def llm_chart(
    project_id: int,
    days: int = Query(30, ge=1, le=365),
    provider: str | None = Query(None),
    category: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    """Mention rate by day, grouped by provider. For Chart.js."""
    await _get_project(project_id, tenant_id, db, user)

    date_from = date.today() - timedelta(days=days)

    query_ids_stmt = select(LlmQuery.id).where(LlmQuery.project_id == project_id, LlmQuery.tenant_id == tenant_id)
    if category:
        query_ids_stmt = query_ids_stmt.where(_category_filter(project_id, category))

    snap_where = [
        LlmSnapshot.llm_query_id.in_(query_ids_stmt),
        LlmSnapshot.tenant_id == tenant_id,
        LlmSnapshot.date >= date_from,
    ]
    if provider:
        snap_where.append(LlmSnapshot.llm_provider == provider)

    rows = await db.execute(
        select(
            LlmSnapshot.date,
            LlmSnapshot.llm_provider,
            func.count().label("total"),
            func.sum(case((LlmSnapshot.brand_mentioned == True, 1), else_=0)).label("mentioned"),  # noqa: E712
        )
        .where(and_(*snap_where))
        .group_by(LlmSnapshot.date, LlmSnapshot.llm_provider)
        .order_by(LlmSnapshot.date)
    )

    # Build date labels
    date_list = []
    d = date_from
    today = date.today()
    while d <= today:
        date_list.append(d)
        d += timedelta(days=1)

    labels = [d.isoformat() for d in date_list]

    # Organize data by provider
    provider_data: dict[str, dict[date, float]] = defaultdict(dict)
    for row in rows:
        rate = row.mentioned / row.total if row.total > 0 else 0.0
        provider_data[row.llm_provider][row.date] = round(rate, 4)

    datasets = []
    for prov, date_rates in sorted(provider_data.items()):
        data = [date_rates.get(d) for d in date_list]
        datasets.append(LlmChartDataset(label=prov, data=data))

    return LlmChartDataResponse(labels=labels, datasets=datasets)


@router.get("/details/{project_id}")
async def llm_details(
    project_id: int,
    query_id: int | None = Query(None),
    provider: str | None = Query(None),
    category: str | None = Query(None),
    days: int = Query(7, ge=1, le=90),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    """Detailed LLM responses with full data."""
    await _get_project(project_id, tenant_id, db, user)

    date_from = date.today() - timedelta(days=days)

    query_ids_stmt = select(LlmQuery.id).where(LlmQuery.project_id == project_id, LlmQuery.tenant_id == tenant_id)
    if category:
        query_ids_stmt = query_ids_stmt.where(_category_filter(project_id, category))

    snap_where = [
        LlmSnapshot.llm_query_id.in_(query_ids_stmt),
        LlmSnapshot.tenant_id == tenant_id,
        LlmSnapshot.date >= date_from,
    ]
    if query_id:
        snap_where.append(LlmSnapshot.llm_query_id == query_id)
    if provider:
        snap_where.append(LlmSnapshot.llm_provider == provider)

    count_result = await db.execute(select(func.count()).select_from(LlmSnapshot).where(and_(*snap_where)))
    total = count_result.scalar() or 0

    result = await db.execute(
        select(LlmSnapshot)
        .where(and_(*snap_where))
        .order_by(LlmSnapshot.date.desc(), LlmSnapshot.llm_provider)
        .offset(offset)
        .limit(limit)
    )
    items = result.scalars().all()

    return {
        "items": [LlmSnapshotDetail.model_validate(s) for s in items],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@router.get("/citations/{project_id}", response_model=LlmCitationsResponse)
async def llm_citations(
    project_id: int,
    days: int = Query(30, ge=1, le=365),
    category: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    """Aggregated citation analysis — which URLs are most cited by LLMs."""
    await _get_project(project_id, tenant_id, db, user)

    date_from = date.today() - timedelta(days=days)

    query_ids_stmt = select(LlmQuery.id).where(LlmQuery.project_id == project_id, LlmQuery.tenant_id == tenant_id)
    if category:
        query_ids_stmt = query_ids_stmt.where(_category_filter(project_id, category))

    result = await db.execute(
        select(LlmSnapshot.cited_urls, LlmSnapshot.llm_provider).where(
            LlmSnapshot.llm_query_id.in_(query_ids_stmt),
            LlmSnapshot.tenant_id == tenant_id,
            LlmSnapshot.date >= date_from,
            LlmSnapshot.cited_urls.isnot(None),
        )
    )

    url_data: dict[str, dict] = {}
    for row in result:
        if not row.cited_urls:
            continue
        for url in row.cited_urls:
            if url not in url_data:
                try:
                    domain = urlparse(url).netloc
                except Exception:
                    domain = ""
                url_data[url] = {"url": url, "domain": domain, "count": 0, "providers": set()}
            url_data[url]["count"] += 1
            url_data[url]["providers"].add(row.llm_provider)

    items = sorted(url_data.values(), key=lambda x: x["count"], reverse=True)[:100]

    return LlmCitationsResponse(
        citations=[
            LlmCitationItem(
                url=item["url"],
                domain=item["domain"],
                count=item["count"],
                providers=sorted(item["providers"]),
            )
            for item in items
        ],
        total=len(url_data),
    )


# ---------------------------------------------------------------------------
# BI Dashboard endpoints (Module 4)
# ---------------------------------------------------------------------------


@router.get("/bi-dashboard/{project_id}", response_model=BiDashboardResponse)
async def llm_bi_dashboard(
    project_id: int,
    days: int = Query(30, ge=1, le=365),
    category: str | None = Query(None),
    provider: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    """Full BI dashboard: AIVS, SoM, Resilience, Heatmap, Citations, GEO advice."""
    project = await _get_project(project_id, tenant_id, db, user)
    return await bi_service.get_bi_dashboard(
        project_id,
        tenant_id,
        db,
        days,
        brand_name=project.brand_name or "",
        category=category,
        provider=provider,
    )


@router.get("/heatmap/{project_id}", response_model=HeatmapResponse)
async def llm_heatmap(
    project_id: int,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    """Scenario x Vendor heatmap with AIVS per cell."""
    await _get_project(project_id, tenant_id, db, user)
    return await bi_service.get_heatmap(project_id, tenant_id, db, days)


@router.get(
    "/geo-advisor/{project_id}",
    response_model=GeoActionPlan,
    dependencies=[Depends(require_role("admin"))],
)
async def llm_geo_advisor(
    project_id: int,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
):
    """GEO strategy recommendations based on current visibility metrics. Admin only."""
    await _get_project(project_id, tenant_id, db)
    return await bi_service.get_geo_advisor(project_id, tenant_id, db, days)


# ---------------------------------------------------------------------------
# Debug Console endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/snapshot/{snapshot_id}",
    response_model=SnapshotWithQueryText,
    dependencies=[Depends(require_role("admin"))],
)
async def llm_snapshot_by_id(
    snapshot_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
):
    """Get a single snapshot by ID with query_text. Fixes the provider bug in Inspector."""
    result = await db.execute(
        select(LlmSnapshot, LlmQuery.query_text)
        .join(LlmQuery, LlmSnapshot.llm_query_id == LlmQuery.id)
        .where(
            LlmSnapshot.id == snapshot_id,
            LlmSnapshot.tenant_id == tenant_id,
        )
    )
    row = result.one_or_none()
    if not row:
        raise NotFoundError("Snapshot not found")

    snap, query_text = row.tuple()
    return SnapshotWithQueryText(
        id=snap.id,
        llm_query_id=snap.llm_query_id,
        query_text=query_text,
        date=snap.date,
        llm_provider=snap.llm_provider,
        llm_model=snap.llm_model,
        brand_mentioned=snap.brand_mentioned,
        mention_type=snap.mention_type,
        mention_context=snap.mention_context,
        competitor_mentions=snap.competitor_mentions,
        cited_urls=snap.cited_urls,
        response_tokens=snap.response_tokens,
        cost_usd=snap.cost_usd,
        raw_response=snap.raw_response,
        collected_at=snap.collected_at,
    )


def _build_trace_scoring(analyzed) -> TraceScoring:
    """Build a TraceScoring with the full RVS formula breakdown from AnalyzedResponse."""
    tb = analyzed.target_brand
    presence = 1.0 if tb.is_mentioned else 0.0
    position_weight = tb.position_weight
    position_rank = tb.position_rank
    sentiment_mult = tb.sentiment_multiplier
    sentiment_lbl = tb.sentiment_label.value if hasattr(tb.sentiment_label, "value") else str(tb.sentiment_label)
    structure_type = (
        analyzed.structure.structure_type.value
        if hasattr(analyzed.structure.structure_type, "value")
        else str(analyzed.structure.structure_type)
    )

    rvs = round(analyzed.response_visibility_score, 4)
    share = round(analyzed.share_of_model_local, 4)

    # Build human-readable position_source
    if not tb.is_mentioned:
        position_source = "not mentioned → 0.0"
    elif structure_type in ("numbered_list", "table", "mixed"):
        if position_rank > 0:
            position_source = (
                f"{structure_type}: rank #{position_rank} → 1.0 - ({position_rank}-1)×0.1 = {position_weight}"
            )
        else:
            position_source = f"{structure_type}: brand not found in list items → default 0.3"
    elif structure_type == "bulleted_list":
        if position_rank > 0:
            position_source = f"bulleted_list: uniform weight → {position_weight}"
        else:
            position_source = "bulleted_list: brand not found in list items → default 0.3"
    else:  # narrative
        offset = tb.char_offset if tb.char_offset >= 0 else 0
        position_source = f"narrative: char_offset={offset} → proximity weight {position_weight}"

    # Build human-readable sentiment_source
    _SENTIMENT_MAP = {"positive": 1.2, "neutral": 1.0, "mixed": 0.8, "negative": 0.5}
    expected_mult = _SENTIMENT_MAP.get(sentiment_lbl, sentiment_mult)
    sentiment_source = f"{sentiment_lbl} → {expected_mult}"

    # Build RVS formula string
    if tb.is_mentioned:
        rvs_formula = f"RVS = {presence} × {position_weight} × {sentiment_mult} = {rvs}"
    else:
        rvs_formula = "RVS = 0.0 (brand not mentioned)"

    # Build SoM local breakdown
    mentioned_comps = [c.name for c in analyzed.competitors if c.is_mentioned]
    total_mentioned = 1 + len(mentioned_comps) if tb.is_mentioned else len(mentioned_comps)
    if tb.is_mentioned and total_mentioned > 0:
        som_formula = f"SoM_local = 1 / {total_mentioned} = {share}"
    else:
        som_formula = "SoM_local = 0.0 (brand not mentioned)"

    return TraceScoring(
        rvs=rvs,
        share_of_model_local=share,
        presence=presence,
        position_rank=position_rank,
        position_weight=position_weight,
        position_source=position_source,
        sentiment_multiplier=sentiment_mult,
        sentiment_label=sentiment_lbl,
        sentiment_source=sentiment_source,
        structure_type=structure_type,
        rvs_formula=rvs_formula,
        total_mentioned_brands=total_mentioned,
        mentioned_competitors=mentioned_comps,
        som_formula=som_formula,
    )


@router.get(
    "/debug/trace/{snapshot_id}",
    response_model=PipelineTraceResponse,
    dependencies=[Depends(require_role("admin"))],
)
async def llm_debug_trace(
    snapshot_id: int,
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
):
    """Re-run the analysis pipeline on a single snapshot and return every step. Admin only."""
    # Fetch the snapshot with its query context
    result = await db.execute(
        select(LlmSnapshot, LlmQuery)
        .join(LlmQuery, LlmSnapshot.llm_query_id == LlmQuery.id)
        .where(
            LlmSnapshot.id == snapshot_id,
            LlmSnapshot.tenant_id == tenant_id,
        )
    )
    row = result.one_or_none()
    if not row:
        raise NotFoundError("Snapshot not found")

    snap, query = row.tuple()

    # Step 0: reconstruct the LLM query with full prompt origin trace
    from app.prompt_engine.trace import trace_prompt_origin
    from app.schemas.llm_debug import TracePromptOrigin

    _competitors_list = []
    if query.competitors:
        if isinstance(query.competitors, list):
            _competitors_list = query.competitors
        elif isinstance(query.competitors, dict):
            _competitors_list = list(query.competitors.keys())

    # Reconstruct how the prompt was generated
    origin = trace_prompt_origin(
        query_text=query.query_text,
        provider=snap.llm_provider,
        model=snap.llm_model,
        target_brand=query.target_brand or "",
        competitors=_competitors_list,
        market="ru",  # TODO: store market in LlmQuery
    )

    step_0 = TraceLlmQuery(
        system_prompt=origin.system_prompt,
        user_prompt=origin.user_prompt,
        provider=snap.llm_provider,
        model=snap.llm_model,
        temperature=0.0,
        max_tokens=2048,
        query_type=query.query_type or "",
        target_brand=query.target_brand or "",
        competitors=_competitors_list,
        api_payload=origin.api_payload,
        prompt_origin=TracePromptOrigin(
            matched_template=origin.matched_template,
            persona=origin.persona,
            intent=origin.intent,
            match_confidence=origin.match_confidence,
        ),
    )

    if not snap.raw_response:
        # Cannot re-analyze without raw_response — return stored data only
        # Build fallback scoring from stored data
        _fallback_rvs = bi_service.rvs_for_mention_type(snap.mention_type)
        return PipelineTraceResponse(
            snapshot_id=snap.id,
            query_text=query.query_text,
            provider=snap.llm_provider,
            raw_response="",
            step_0_llm_query=step_0,
            step_1_sanitization=TraceSanitization(flag="no_raw_response"),
            step_2_entity_resolution=TraceEntityResolution(
                target_brand=TraceTargetBrand(
                    name=query.target_brand or "",
                    is_mentioned=snap.brand_mentioned,
                    mention_type=snap.mention_type,
                    mention_context=snap.mention_context or "",
                ),
                competitors=[
                    TraceCompetitor(name=name, is_mentioned=bool(mentioned))
                    for name, mentioned in (snap.competitor_mentions or {}).items()
                ],
            ),
            step_3_structure=TraceStructure(),
            step_4_context=TraceContext(),
            step_5_citations=[TraceCitation(url=url) for url in (snap.cited_urls or [])],
            step_6_scoring=TraceScoring(
                rvs=_fallback_rvs,
                presence=1.0 if snap.brand_mentioned else 0.0,
                rvs_formula=f"RVS = {_fallback_rvs} (from mention_type lookup, no raw_response for full recalc)",
            ),
        )

    # Re-run the full pipeline (with LLM-as-a-Judge)
    from app.analysis.pipeline import analyze_response
    from app.gateway.types import GatewayResponse, GatewayVendor

    # Build competitors dict: {"Name": []} for the pipeline
    competitors_dict: dict[str, list[str]] = {}
    if query.competitors:
        # query.competitors is stored as list of names
        if isinstance(query.competitors, list):
            for c in query.competitors:
                competitors_dict[c] = []
        elif isinstance(query.competitors, dict):
            competitors_dict = {k: (v if isinstance(v, list) else []) for k, v in query.competitors.items()}

    try:
        vendor = GatewayVendor(snap.llm_provider)
    except ValueError:
        vendor = GatewayVendor.CHATGPT

    gateway_resp = GatewayResponse(
        request_id=f"trace-{snapshot_id}",
        vendor=vendor,
        model_version=snap.llm_model,
        raw_response_text=snap.raw_response,
        cited_urls=snap.cited_urls or [],
        total_tokens=snap.response_tokens or 0,
        cost_usd=snap.cost_usd or 0.0,
        tenant_id=str(tenant_id),
        project_id=query.project_id,
        llm_query_id=query.id,
    )

    # Get OpenAI API key for LLM-as-a-Judge from tenant credentials
    judge_api_key = await _get_judge_api_key(db, tenant_id)

    analyzed = await analyze_response(
        response=gateway_resp,
        target_brand=query.target_brand or "",
        target_aliases=[],
        competitors=competitors_dict,
        intent=query.query_type,
        judge_api_key=judge_api_key,
    )

    return PipelineTraceResponse(
        snapshot_id=snap.id,
        query_text=query.query_text,
        provider=snap.llm_provider,
        raw_response=snap.raw_response or "",
        step_0_llm_query=step_0,
        step_1_sanitization=TraceSanitization(
            flag=analyzed.sanitization.flag.value,
            stripped_chars=analyzed.sanitization.stripped_chars,
            think_content=analyzed.sanitization.think_content,
        ),
        step_2_entity_resolution=TraceEntityResolution(
            target_brand=TraceTargetBrand(
                name=analyzed.target_brand.name,
                is_mentioned=analyzed.target_brand.is_mentioned,
                position_rank=analyzed.target_brand.position_rank,
                position_weight=analyzed.target_brand.position_weight,
                mention_type=analyzed.target_brand.mention_type.value,
                mention_context=analyzed.target_brand.mention_context,
                char_offset=analyzed.target_brand.char_offset,
                sentence_index=analyzed.target_brand.sentence_index,
            ),
            competitors=[
                TraceCompetitor(
                    name=c.name,
                    is_mentioned=c.is_mentioned,
                    position_rank=c.position_rank,
                    position_weight=c.position_weight,
                    mention_type=c.mention_type.value,
                )
                for c in analyzed.competitors
            ],
        ),
        step_3_structure=TraceStructure(
            type=analyzed.structure.structure_type.value,
            total_items=analyzed.structure.total_items,
            brands_in_list=analyzed.structure.brands_in_list,
        ),
        step_4_context=TraceContext(
            mention_type=analyzed.target_brand.mention_type.value,
            sentiment_score=analyzed.target_brand.sentiment_score,
            sentiment_label=analyzed.target_brand.sentiment_label.value,
            sentiment_multiplier=analyzed.target_brand.sentiment_multiplier,
            context_tags=analyzed.target_brand.context_tags,
            is_recommended=analyzed.target_brand.is_recommended,
            is_hallucination=analyzed.target_brand.is_hallucination,
            judge_method=analyzed.target_brand.judge_method,
            judge_prompt_system=analyzed.target_brand.judge_prompt_system,
            judge_prompt_user=analyzed.target_brand.judge_prompt_user,
            judge_raw_response=analyzed.target_brand.judge_raw_response,
        ),
        step_5_citations=[
            TraceCitation(
                url=c.url,
                domain=c.domain,
                is_native=c.is_native,
            )
            for c in analyzed.citations
        ],
        step_6_scoring=_build_trace_scoring(analyzed),
    )


@router.get(
    "/debug/{project_id}",
    response_model=LlmDebugResponse,
    dependencies=[Depends(require_role("admin"))],
)
async def llm_debug(
    project_id: int,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
):
    """Full debug trace — shows exactly how every metric is computed. Admin only."""
    await _get_project(project_id, tenant_id, db)
    return await bi_service.get_debug_trace(project_id, tenant_id, db, days)
