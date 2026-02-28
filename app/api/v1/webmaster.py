"""Webmaster data API endpoints: GSC daily breakdown, GSC semantics."""

import logging
from datetime import date, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id
from app.core.encryption import decrypt_value
from app.core.exceptions import BadRequestError, NotFoundError
from app.db.postgres import get_db
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.serp import SerpSnapshot
from app.models.tenant import Tenant
from app.models.webmaster import WebmasterData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webmaster", tags=["webmaster"])


@router.get("/gsc/{keyword_id}")
async def get_gsc_daily(
    keyword_id: int,
    days: int = Query(30, ge=7, le=90),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """GSC performance data by day for a specific keyword (last N days)."""
    # Verify keyword belongs to tenant
    kw_result = await db.execute(select(Keyword).where(Keyword.id == keyword_id, Keyword.tenant_id == tenant_id))
    keyword = kw_result.scalar_one_or_none()
    if not keyword:
        raise NotFoundError("Keyword not found")

    date_from = date.today() - timedelta(days=days)

    result = await db.execute(
        select(WebmasterData)
        .where(
            WebmasterData.keyword_id == keyword_id,
            WebmasterData.search_engine == "google",
            WebmasterData.date >= date_from,
            WebmasterData.tenant_id == tenant_id,
        )
        .order_by(WebmasterData.date.desc())
    )
    rows = result.scalars().all()

    return {
        "keyword": keyword.keyword,
        "days": [
            {
                "date": r.date.isoformat(),
                "position": round(r.position, 1) if r.position else None,
                "url": r.page_url,
                "impressions": r.impressions,
                "clicks": r.clicks,
                "ctr": round(r.ctr, 2) if r.ctr else None,
            }
            for r in rows
        ],
    }


@router.get("/gsc-semantics/{project_id}")
async def get_gsc_semantics(
    project_id: int,
    min_impressions: int = Query(300, ge=0),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Advanced GSC analysis — find additional queries ranking on project pages.

    Fetches queries from GSC API (last 28 days) for pages where tracked keywords rank,
    excluding already-tracked keywords. Useful for discovering new content opportunities.

    Requires GSC credentials configured for the tenant.
    """
    # Verify project
    proj_result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = proj_result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")

    if not project.gsc_site_url:
        raise BadRequestError("GSC site URL not configured for this project")

    # Get tenant credentials
    tenant_result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    tenant = tenant_result.scalar_one_or_none()
    if not tenant or not tenant.gsc_credentials_json:
        raise BadRequestError("GSC credentials not configured")

    gsc_creds_json = decrypt_value(tenant.gsc_credentials_json)
    if not gsc_creds_json:
        raise BadRequestError("GSC credentials invalid")

    # Collect unique found_urls from latest Google snapshots
    kw_result = await db.execute(
        select(Keyword.id, Keyword.keyword).where(
            Keyword.project_id == project_id,
            Keyword.tenant_id == tenant_id,
            Keyword.is_active == True,  # noqa: E712
        )
    )
    keywords = kw_result.all()
    if not keywords:
        return {"pages": []}

    kw_ids = [kw.id for kw in keywords]
    tracked_kw_set = {kw.keyword.lower() for kw in keywords}

    # Get unique found URLs from latest snapshots
    latest_sub = (
        select(
            SerpSnapshot.keyword_id,
            func.max(SerpSnapshot.date).label("max_date"),
        )
        .where(
            SerpSnapshot.keyword_id.in_(kw_ids),
            SerpSnapshot.search_engine == "google",
            SerpSnapshot.tenant_id == tenant_id,
        )
        .group_by(SerpSnapshot.keyword_id)
        .subquery()
    )

    url_stmt = (
        select(SerpSnapshot.found_url)
        .join(
            latest_sub,
            (SerpSnapshot.keyword_id == latest_sub.c.keyword_id) & (SerpSnapshot.date == latest_sub.c.max_date),
        )
        .where(
            SerpSnapshot.search_engine == "google",
            SerpSnapshot.tenant_id == tenant_id,
            SerpSnapshot.found_url.is_not(None),
        )
        .distinct()
    )
    url_result = await db.execute(url_stmt)
    tracked_urls = [r[0] for r in url_result.all()]

    if not tracked_urls:
        return {"pages": []}

    # Query GSC API for these pages
    import httpx
    import json

    try:
        creds = json.loads(gsc_creds_json)
    except (json.JSONDecodeError, TypeError):
        raise BadRequestError("GSC credentials JSON is malformed")

    access_token = creds.get("token", creds.get("access_token", ""))
    if not access_token:
        raise BadRequestError("GSC access token not found in credentials")

    date_to = date.today() - timedelta(days=2)  # GSC data has ~2-day lag
    date_from = date_to - timedelta(days=28)

    gsc_api = f"https://www.googleapis.com/webmasters/v3/sites/{project.gsc_site_url}/searchAnalytics/query"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                gsc_api,
                headers={"Authorization": f"Bearer {access_token}"},
                json={
                    "startDate": date_from.isoformat(),
                    "endDate": date_to.isoformat(),
                    "dimensions": ["query", "page"],
                    "rowLimit": 5000,
                    "dataState": "final",
                },
            )
            if resp.status_code != 200:
                logger.warning("GSC semantics API error: %s %s", resp.status_code, resp.text[:200])
                raise BadRequestError(f"GSC API error: {resp.status_code}")

            data = resp.json()
    except httpx.HTTPError as e:
        raise BadRequestError(f"GSC API connection error: {e}")

    # Filter and group results
    pages: dict[str, list[dict]] = {}
    for row in data.get("rows", []):
        keys = row.get("keys", [])
        if len(keys) < 2:
            continue

        query = keys[0]
        page = keys[1]
        impressions = row.get("impressions", 0)

        # Filter: must be on tracked page, not a tracked keyword, enough impressions
        if page not in tracked_urls:
            continue
        if query.lower() in tracked_kw_set:
            continue
        if impressions < min_impressions:
            continue

        pages.setdefault(page, []).append(
            {
                "query": query,
                "impressions": impressions,
                "clicks": row.get("clicks", 0),
                "position": round(row.get("position", 0), 1),
                "ctr": round(row.get("ctr", 0) * 100, 2),
            }
        )

    # Sort queries within each page by impressions desc
    result_pages = []
    for url in sorted(pages.keys()):
        queries = sorted(pages[url], key=lambda x: x["impressions"], reverse=True)
        result_pages.append({"url": url, "queries": queries})

    return {"pages": result_pages}
