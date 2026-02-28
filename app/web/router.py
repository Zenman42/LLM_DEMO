"""Web UI routes — serves Jinja2 templates.

Demo mode is always on. Project pages pre-render data server-side so that
tools like Lovable (which don't execute JS) can see actual content.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
templates.env.globals["demo_mode"] = True

web_router = APIRouter(tags=["web"])


# ---------------------------------------------------------------------------
# Demo-mode helper: fetch project data for server-side rendering
# ---------------------------------------------------------------------------

async def _demo_project_context(project_id: int) -> dict:
    """Return project + keywords + LLM data for Jinja2 SSR in demo mode."""
    from sqlalchemy import select, func, and_, case

    from app.db.postgres import async_session_factory
    from app.models.discovered_entity import DiscoveredEntity
    from app.models.keyword import Keyword
    from app.models.llm_query import LlmQuery
    from app.models.llm_snapshot import LlmSnapshot
    from app.models.project import Project
    from app.models.serp import SerpSnapshot

    async with async_session_factory() as db:
        # 1. Project
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()
        if not project:
            return {}

        engine = "google" if project.search_engine in ("google", "both") else "yandex"

        # 2. Active keywords
        result = await db.execute(
            select(Keyword).where(Keyword.project_id == project_id, Keyword.is_active.is_(True))
        )
        keywords = result.scalars().all()

        # 3. Latest SERP snapshot per keyword for chosen engine
        max_date_sq = (
            select(
                SerpSnapshot.keyword_id,
                func.max(SerpSnapshot.date).label("max_date"),
            )
            .where(SerpSnapshot.search_engine == engine)
            .group_by(SerpSnapshot.keyword_id)
            .subquery()
        )
        result = await db.execute(
            select(SerpSnapshot).join(
                max_date_sq,
                and_(
                    SerpSnapshot.keyword_id == max_date_sq.c.keyword_id,
                    SerpSnapshot.date == max_date_sq.c.max_date,
                    SerpSnapshot.search_engine == engine,
                ),
            )
        )
        snapshots = {s.keyword_id: s for s in result.scalars().all()}

        # Build keyword rows
        kw_rows = []
        for kw in keywords:
            snap = snapshots.get(kw.id)
            pos = snap.position if snap else None
            prev = snap.previous_position if snap else None
            change = (prev - pos) if (pos and prev) else None
            found_url = snap.found_url if snap else None
            match = None
            if kw.target_url and found_url:
                match = found_url.rstrip("/") == kw.target_url.rstrip("/")

            kw_rows.append(
                {
                    "keyword_id": kw.id,
                    "keyword": kw.keyword,
                    "position": pos,
                    "change": change,
                    "found_url": found_url,
                    "match": match,
                }
            )

        # ---------------------------------------------------------------
        # LLM data (only if project tracks LLM)
        # ---------------------------------------------------------------
        llm_stats: dict = {}
        llm_responses: list[dict] = []
        llm_entities: list[dict] = []
        llm_provider_rates: dict[str, float] = {}
        llm_competitor_sov: dict[str, float] = {}

        if project.track_llm:
            date_from = date.today() - timedelta(days=30)

            # Query IDs for this project
            query_ids_stmt = select(LlmQuery.id).where(
                LlmQuery.project_id == project_id,
                LlmQuery.is_active.is_(True),
            )

            # Total active queries
            total_q_result = await db.execute(
                select(func.count()).select_from(LlmQuery).where(
                    LlmQuery.project_id == project_id,
                    LlmQuery.is_active.is_(True),
                )
            )
            total_queries = total_q_result.scalar() or 0

            # Snapshot aggregates (last 30 days)
            snap_where = and_(
                LlmSnapshot.llm_query_id.in_(query_ids_stmt),
                LlmSnapshot.date >= date_from,
            )

            total_checks_r = await db.execute(
                select(func.count()).select_from(LlmSnapshot).where(snap_where)
            )
            total_checks = total_checks_r.scalar() or 0

            mentioned_r = await db.execute(
                select(func.count()).select_from(LlmSnapshot).where(
                    snap_where, LlmSnapshot.brand_mentioned.is_(True)
                )
            )
            mentioned_count = mentioned_r.scalar() or 0
            mention_rate = mentioned_count / total_checks if total_checks > 0 else 0.0

            # Cost
            cost_r = await db.execute(
                select(func.coalesce(func.sum(LlmSnapshot.cost_usd), 0.0)).where(snap_where)
            )
            total_cost = cost_r.scalar() or 0.0

            # Mention rate by provider
            prov_stats = await db.execute(
                select(
                    LlmSnapshot.llm_provider,
                    func.count().label("total"),
                    func.sum(case((LlmSnapshot.brand_mentioned.is_(True), 1), else_=0)).label("mentioned"),
                )
                .where(snap_where)
                .group_by(LlmSnapshot.llm_provider)
            )
            for row in prov_stats:
                llm_provider_rates[row.llm_provider] = (
                    round(row.mentioned / row.total * 100, 1) if row.total > 0 else 0.0
                )

            # SOV: brand vs competitors
            all_snaps_r = await db.execute(
                select(
                    LlmSnapshot.brand_mentioned,
                    LlmSnapshot.competitor_mentions,
                ).where(snap_where)
            )
            total_brand = 0
            total_all = 0
            comp_counts: dict[str, int] = defaultdict(int)
            for snap_row in all_snaps_r:
                if snap_row.brand_mentioned:
                    total_brand += 1
                    total_all += 1
                if snap_row.competitor_mentions:
                    for comp, m in snap_row.competitor_mentions.items():
                        if m and "::" not in comp:
                            comp_counts[comp] += 1
                            total_all += 1

            sov = round(total_brand / total_all * 100, 1) if total_all > 0 else 0.0
            for comp, cnt in sorted(comp_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                llm_competitor_sov[comp] = round(cnt / total_all * 100, 1) if total_all > 0 else 0.0

            llm_stats = {
                "total_queries": total_queries,
                "total_checks": total_checks,
                "mention_rate": round(mention_rate * 100, 1),
                "sov": sov,
                "total_cost": round(total_cost, 4),
            }

            # Recent LLM responses (last 20)
            recent_r = await db.execute(
                select(
                    LlmSnapshot.id,
                    LlmSnapshot.date,
                    LlmSnapshot.llm_provider,
                    LlmSnapshot.llm_model,
                    LlmSnapshot.brand_mentioned,
                    LlmSnapshot.mention_type,
                    LlmSnapshot.cost_usd,
                    LlmSnapshot.sentiment_label,
                    LlmQuery.query_text,
                )
                .join(LlmQuery, LlmSnapshot.llm_query_id == LlmQuery.id)
                .where(snap_where)
                .order_by(LlmSnapshot.date.desc(), LlmSnapshot.id.desc())
                .limit(20)
            )
            for r in recent_r:
                llm_responses.append({
                    "id": r.id,
                    "date": r.date.isoformat() if r.date else "",
                    "provider": r.llm_provider,
                    "model": r.llm_model or "",
                    "brand_mentioned": r.brand_mentioned,
                    "mention_type": r.mention_type or "none",
                    "cost": round(r.cost_usd, 4) if r.cost_usd else 0.0,
                    "sentiment": r.sentiment_label or "",
                    "query_text": (r.query_text or "")[:80],
                })

            # Discovered entities (top 20 by mention count)
            ent_r = await db.execute(
                select(DiscoveredEntity)
                .where(
                    DiscoveredEntity.project_id == project_id,
                    DiscoveredEntity.status != "rejected",
                    DiscoveredEntity.alias_of_id.is_(None),
                )
                .order_by(DiscoveredEntity.mention_count.desc())
                .limit(20)
            )
            for e in ent_r.scalars():
                llm_entities.append({
                    "name": e.entity_name,
                    "count": e.mention_count,
                    "type": e.entity_type or "",
                    "status": e.status,
                    "first_seen": e.first_seen.isoformat() if e.first_seen else "",
                    "last_seen": e.last_seen.isoformat() if e.last_seen else "",
                })

        return {
            "demo_project": {
                "id": project.id,
                "name": project.name,
                "domain": project.domain or "",
                "search_engine": project.search_engine,
                "track_llm": project.track_llm,
                "brand_name": project.brand_name or "",
                "competitors": project.competitors or [],
            },
            "demo_keywords": kw_rows,
            "demo_engine": engine,
            "llm_stats": llm_stats,
            "llm_responses": llm_responses,
            "llm_entities": llm_entities,
            "llm_provider_rates": llm_provider_rates,
            "llm_competitor_sov": llm_competitor_sov,
        }


async def _demo_dashboard_context() -> dict:
    """Return list of projects with LLM stats for dashboard SSR in demo mode."""
    from sqlalchemy import select, func, case, and_

    from app.db.postgres import async_session_factory
    from app.models.discovered_entity import DiscoveredEntity
    from app.models.llm_query import LlmQuery
    from app.models.llm_snapshot import LlmSnapshot
    from app.models.project import Project

    async with async_session_factory() as db:
        result = await db.execute(
            select(Project).where(Project.is_active.is_(True)).order_by(Project.id)
        )
        projects = result.scalars().all()

        date_from = date.today() - timedelta(days=30)
        project_list = []

        for p in projects:
            item: dict = {
                "id": p.id,
                "name": p.name,
                "domain": p.domain or "",
                "search_engine": p.search_engine,
                "track_llm": p.track_llm,
                "llm_queries": 0,
                "llm_mention_rate": 0.0,
                "llm_total_checks": 0,
                "llm_entities": 0,
            }

            if p.track_llm:
                # Query count
                q_r = await db.execute(
                    select(func.count()).select_from(LlmQuery).where(
                        LlmQuery.project_id == p.id, LlmQuery.is_active.is_(True)
                    )
                )
                item["llm_queries"] = q_r.scalar() or 0

                # Snapshot stats (last 30 days)
                query_ids = select(LlmQuery.id).where(
                    LlmQuery.project_id == p.id, LlmQuery.is_active.is_(True)
                )
                snap_where = and_(
                    LlmSnapshot.llm_query_id.in_(query_ids),
                    LlmSnapshot.date >= date_from,
                )

                stats_r = await db.execute(
                    select(
                        func.count().label("total"),
                        func.sum(case((LlmSnapshot.brand_mentioned.is_(True), 1), else_=0)).label("mentioned"),
                    ).where(snap_where)
                )
                row = stats_r.one()
                total = row.total or 0
                mentioned = row.mentioned or 0
                item["llm_total_checks"] = total
                item["llm_mention_rate"] = round(mentioned / total * 100, 1) if total > 0 else 0.0

                # Entity count
                ent_r = await db.execute(
                    select(func.count()).select_from(DiscoveredEntity).where(
                        DiscoveredEntity.project_id == p.id,
                        DiscoveredEntity.status != "rejected",
                        DiscoveredEntity.alias_of_id.is_(None),
                    )
                )
                item["llm_entities"] = ent_r.scalar() or 0

            project_list.append(item)

        return {"demo_projects": project_list}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@web_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Redirect to dashboard — no login needed."""
    return RedirectResponse(url="/", status_code=302)


@web_router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    ctx = await _demo_dashboard_context()
    return templates.TemplateResponse(request, "dashboard.html", ctx)


@web_router.get("/project/new", response_class=HTMLResponse)
async def onboarding_page(request: Request):
    return templates.TemplateResponse(request, "llm_onboarding.html")


@web_router.get("/project/{project_id}", response_class=HTMLResponse)
async def project_page(request: Request, project_id: int):
    ctx: dict = {"project_id": project_id}
    ctx.update(await _demo_project_context(project_id))
    return templates.TemplateResponse(request, "project.html", ctx)


@web_router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html")


@web_router.get("/users", response_class=HTMLResponse)
async def users_page(request: Request):
    return templates.TemplateResponse(request, "users.html")


@web_router.get("/api-keys", response_class=HTMLResponse)
async def api_keys_page(request: Request):
    return templates.TemplateResponse(request, "api_keys.html")


@web_router.get("/export", response_class=HTMLResponse)
async def export_page(request: Request):
    return templates.TemplateResponse(request, "export.html")


@web_router.get("/account", response_class=HTMLResponse)
async def account_page(request: Request):
    return templates.TemplateResponse(request, "account.html")


@web_router.get("/project/{project_id}/llm-dashboard", response_class=HTMLResponse)
async def llm_dashboard_page(request: Request, project_id: int):
    return templates.TemplateResponse(request, "llm_dashboard.html", {"project_id": project_id})


@web_router.get("/project/{project_id}/llm-debug", response_class=HTMLResponse)
async def llm_debug_page(request: Request, project_id: int):
    return templates.TemplateResponse(request, "llm_debug.html", {"project_id": project_id})


@web_router.get("/project/{project_id}/prompts", response_class=HTMLResponse)
async def prompts_page(request: Request, project_id: int):
    return templates.TemplateResponse(request, "prompts.html", {"project_id": project_id})


@web_router.get("/project/{project_id}/competitors", response_class=HTMLResponse)
async def competitors_page(request: Request, project_id: int):
    return templates.TemplateResponse(request, "competitors.html", {"project_id": project_id})


@web_router.get("/project/{project_id}/settings", response_class=HTMLResponse)
async def project_settings_page(request: Request, project_id: int):
    return templates.TemplateResponse(request, "project_settings.html", {"project_id": project_id})


@web_router.get("/project/{project_id}/geo-command-center", response_class=HTMLResponse)
async def geo_command_center_page(request: Request, project_id: int):
    return templates.TemplateResponse(request, "geo_command_center.html", {"project_id": project_id})
