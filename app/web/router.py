"""Web UI routes — serves Jinja2 templates.

Demo mode is always on. Project pages pre-render data server-side so that
tools like Lovable (which don't execute JS) can see actual content.
"""

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
    """Return project + keywords data for Jinja2 SSR in demo mode."""
    from sqlalchemy import select, func, and_

    from app.db.postgres import async_session_factory
    from app.models.keyword import Keyword
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

        return {
            "demo_project": {
                "id": project.id,
                "name": project.name,
                "domain": project.domain or "",
                "search_engine": project.search_engine,
                "track_llm": project.track_llm,
                "brand_name": project.brand_name or "",
            },
            "demo_keywords": kw_rows,
            "demo_engine": engine,
        }


async def _demo_dashboard_context() -> dict:
    """Return list of projects for dashboard SSR in demo mode."""
    from sqlalchemy import select

    from app.db.postgres import async_session_factory
    from app.models.project import Project

    async with async_session_factory() as db:
        result = await db.execute(
            select(Project).where(Project.is_active.is_(True)).order_by(Project.id)
        )
        projects = result.scalars().all()
        return {
            "demo_projects": [
                {
                    "id": p.id,
                    "name": p.name,
                    "domain": p.domain or "",
                    "search_engine": p.search_engine,
                    "track_llm": p.track_llm,
                }
                for p in projects
            ]
        }


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
