"""SERP data API endpoints: chart data, SERP details, URL-keywords."""

from datetime import date, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Float, and_, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id
from app.core.exceptions import BadRequestError, NotFoundError
from app.db.clickhouse import get_ch_client
from app.db.postgres import get_db
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.serp import SerpSnapshot
from app.models.webmaster import WebmasterData
from app.schemas.serp import ChartDataResponse, ChartDataset, SerpDetailsResponse, SerpDetailItem

router = APIRouter(prefix="/serp", tags=["serp"])


@router.get("/chart/{project_id}", response_model=ChartDataResponse)
async def get_chart_data(
    project_id: int,
    engine: str = Query("google", pattern=r"^(google|yandex)$"),
    days: int = Query(30, ge=7, le=90),
    keyword_id: int | None = Query(None, description="Single keyword mode"),
    keyword_ids: str | None = Query(None, description="Comma-separated keyword IDs for URL mode"),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Get chart data for position history visualization.

    Two modes:
    - keyword mode: single keyword_id → one SERP line + optional GSC line (Google only)
    - url mode: keyword_ids (comma-separated) → multiple SERP lines
    """
    # Verify project belongs to tenant
    proj = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    if not proj.scalar_one_or_none():
        raise NotFoundError("Project not found")

    # Parse keyword IDs
    kw_ids: list[int] = []
    if keyword_id is not None:
        kw_ids = [keyword_id]
    elif keyword_ids:
        try:
            kw_ids = [int(x.strip()) for x in keyword_ids.split(",") if x.strip()]
        except ValueError:
            raise BadRequestError("Invalid keyword_ids format")

    if not kw_ids:
        raise BadRequestError("Provide keyword_id or keyword_ids")

    # Verify keywords belong to project + tenant
    kw_result = await db.execute(
        select(Keyword).where(
            Keyword.id.in_(kw_ids),
            Keyword.project_id == project_id,
            Keyword.tenant_id == tenant_id,
        )
    )
    keywords = {kw.id: kw for kw in kw_result.scalars().all()}
    if not keywords:
        raise NotFoundError("No valid keywords found")

    # Date range
    date_to = date.today()
    date_from = date_to - timedelta(days=days)

    # Build date labels
    labels = []
    d = date_from
    while d <= date_to:
        labels.append(d.isoformat())
        d += timedelta(days=1)

    # Fetch SERP snapshots
    snap_result = await db.execute(
        select(SerpSnapshot).where(
            SerpSnapshot.keyword_id.in_(list(keywords.keys())),
            SerpSnapshot.search_engine == engine,
            SerpSnapshot.date >= date_from,
            SerpSnapshot.date <= date_to,
            SerpSnapshot.tenant_id == tenant_id,
        )
    )
    snapshots = snap_result.scalars().all()

    # Index: {keyword_id: {date_str: position}}
    snap_index: dict[int, dict[str, float | None]] = {}
    for s in snapshots:
        snap_index.setdefault(s.keyword_id, {})[s.date.isoformat()] = (
            float(s.position) if s.position and s.position > 0 else None
        )

    datasets: list[ChartDataset] = []

    for kw_id, kw in keywords.items():
        kw_data = snap_index.get(kw_id, {})
        data = [kw_data.get(lbl) for lbl in labels]
        datasets.append(ChartDataset(label=kw.keyword, data=data, type="serp"))

    # For single keyword + Google engine → add GSC avg_position as overlay
    if len(kw_ids) == 1 and engine == "google":
        the_kw_id = kw_ids[0]
        gsc_result = await db.execute(
            select(WebmasterData).where(
                WebmasterData.keyword_id == the_kw_id,
                WebmasterData.search_engine == "google",
                WebmasterData.date >= date_from,
                WebmasterData.date <= date_to,
                WebmasterData.tenant_id == tenant_id,
            )
        )
        gsc_rows = gsc_result.scalars().all()
        if gsc_rows:
            gsc_index = {r.date.isoformat(): r.position for r in gsc_rows}
            gsc_data = [gsc_index.get(lbl) for lbl in labels]
            datasets.append(
                ChartDataset(
                    label=f"GSC: {keywords[the_kw_id].keyword}",
                    data=gsc_data,
                    type="gsc",
                )
            )

    return ChartDataResponse(labels=labels, datasets=datasets)


@router.get("/details/{keyword_id}/{engine}", response_model=SerpDetailsResponse)
async def get_serp_details(
    keyword_id: int,
    engine: str,
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Get full SERP top-50 details for a keyword + engine from ClickHouse."""
    if engine not in ("google", "yandex"):
        raise BadRequestError("Engine must be 'google' or 'yandex'")

    # Verify keyword belongs to tenant
    kw_result = await db.execute(select(Keyword).where(Keyword.id == keyword_id, Keyword.tenant_id == tenant_id))
    keyword = kw_result.scalar_one_or_none()
    if not keyword:
        raise NotFoundError("Keyword not found")

    # Get latest snapshot date from PG
    snap_result = await db.execute(
        select(SerpSnapshot.date)
        .where(
            SerpSnapshot.keyword_id == keyword_id,
            SerpSnapshot.search_engine == engine,
            SerpSnapshot.tenant_id == tenant_id,
        )
        .order_by(SerpSnapshot.date.desc())
        .limit(1)
    )
    latest_date = snap_result.scalar_one_or_none()
    if not latest_date:
        return SerpDetailsResponse(keyword=keyword.keyword, date="", search_engine=engine, details=[])

    # Fetch details from ClickHouse
    try:
        ch = get_ch_client()
        result = ch.query(
            """
            SELECT position, url, domain, title, snippet
            FROM serp_details
            WHERE keyword_id = %(kw_id)s
              AND search_engine = %(engine)s
              AND date = %(dt)s
              AND tenant_id = %(tid)s
            ORDER BY position ASC
            """,
            parameters={
                "kw_id": keyword_id,
                "engine": engine,
                "dt": latest_date.isoformat(),
                "tid": str(tenant_id),
            },
        )
        details = [
            SerpDetailItem(position=row[0], url=row[1], domain=row[2], title=row[3], snippet=row[4])
            for row in result.result_rows
        ]
    except Exception:
        # ClickHouse unavailable — return empty
        details = []

    return SerpDetailsResponse(
        keyword=keyword.keyword,
        date=latest_date.isoformat(),
        search_engine=engine,
        details=details,
    )


@router.get("/url-keywords/{project_id}")
async def get_url_keywords(
    project_id: int,
    page_url: str = Query(..., description="URL to find keywords for"),
    engine: str = Query("google", pattern=r"^(google|yandex)$"),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Get keywords that rank on a specific URL — for chart URL-mode selector."""
    # Verify project
    proj = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    if not proj.scalar_one_or_none():
        raise NotFoundError("Project not found")

    # Find latest snapshot per keyword where found_url matches
    # Subquery: latest date per keyword+engine
    latest_sub = (
        select(
            SerpSnapshot.keyword_id,
            func.max(SerpSnapshot.date).label("max_date"),
        )
        .where(
            SerpSnapshot.tenant_id == tenant_id,
            SerpSnapshot.search_engine == engine,
        )
        .join(Keyword, Keyword.id == SerpSnapshot.keyword_id)
        .where(Keyword.project_id == project_id)
        .group_by(SerpSnapshot.keyword_id)
        .subquery()
    )

    # Join to get snapshots with matching URL
    stmt = (
        select(SerpSnapshot, Keyword.keyword)
        .join(Keyword, Keyword.id == SerpSnapshot.keyword_id)
        .join(
            latest_sub,
            and_(
                SerpSnapshot.keyword_id == latest_sub.c.keyword_id,
                SerpSnapshot.date == latest_sub.c.max_date,
            ),
        )
        .where(
            SerpSnapshot.search_engine == engine,
            SerpSnapshot.found_url == page_url,
            SerpSnapshot.tenant_id == tenant_id,
        )
    )
    result = await db.execute(stmt)
    rows = result.all()

    # For Google, enrich with 30-day impressions from GSC
    kw_impressions: dict[int, int] = {}
    if engine == "google" and rows:
        kw_ids_list = [r.SerpSnapshot.keyword_id for r in rows]
        date_from = date.today() - timedelta(days=30)
        imp_result = await db.execute(
            select(
                WebmasterData.keyword_id,
                func.sum(WebmasterData.impressions).label("total_impressions"),
            )
            .where(
                WebmasterData.keyword_id.in_(kw_ids_list),
                WebmasterData.search_engine == "google",
                WebmasterData.date >= date_from,
                WebmasterData.tenant_id == tenant_id,
            )
            .group_by(WebmasterData.keyword_id)
        )
        for r in imp_result.all():
            kw_impressions[r.keyword_id] = r.total_impressions or 0

    keywords_out = []
    for row in rows:
        snap = row.SerpSnapshot
        keywords_out.append(
            {
                "id": snap.keyword_id,
                "keyword": row.keyword,
                "impressions": kw_impressions.get(snap.keyword_id, 0),
            }
        )

    # Sort by impressions descending
    keywords_out.sort(key=lambda x: x["impressions"], reverse=True)

    return {"keywords": keywords_out}


@router.get("/project-keywords/{project_id}")
async def get_project_keywords_with_positions(
    project_id: int,
    engine: str = Query("google", pattern=r"^(google|yandex)$"),
    page_url: str | None = Query(None, description="Filter by found URL"),
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
):
    """Full keyword ranking table for project detail view.

    Returns for each keyword:
    - position, change, found_url, match status
    - GSC metrics (Google only): position, impressions, clicks, ctr (7-day aggregated)
    """
    # Verify project
    proj_result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = proj_result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")

    # Get all active keywords for project
    kw_result = await db.execute(
        select(Keyword).where(
            Keyword.project_id == project_id,
            Keyword.tenant_id == tenant_id,
            Keyword.is_active == True,  # noqa: E712
        )
    )
    keywords = kw_result.scalars().all()
    if not keywords:
        return {"keywords": [], "all_urls": []}

    kw_map = {kw.id: kw for kw in keywords}
    kw_ids = list(kw_map.keys())

    # Latest snapshot per keyword for this engine
    latest_sub = (
        select(
            SerpSnapshot.keyword_id,
            func.max(SerpSnapshot.date).label("max_date"),
        )
        .where(
            SerpSnapshot.keyword_id.in_(kw_ids),
            SerpSnapshot.search_engine == engine,
            SerpSnapshot.tenant_id == tenant_id,
        )
        .group_by(SerpSnapshot.keyword_id)
        .subquery()
    )

    snap_stmt = (
        select(SerpSnapshot)
        .join(
            latest_sub,
            and_(
                SerpSnapshot.keyword_id == latest_sub.c.keyword_id,
                SerpSnapshot.date == latest_sub.c.max_date,
            ),
        )
        .where(SerpSnapshot.search_engine == engine, SerpSnapshot.tenant_id == tenant_id)
    )
    snap_result = await db.execute(snap_stmt)
    latest_snaps = {s.keyword_id: s for s in snap_result.scalars().all()}

    # Collect all unique found URLs for the filter dropdown
    all_urls = sorted({s.found_url for s in latest_snaps.values() if s.found_url})

    # GSC data (7-day aggregate) for Google engine
    gsc_data: dict[int, dict] = {}
    if engine == "google":
        date_from_gsc = date.today() - timedelta(days=7)
        gsc_stmt = (
            select(
                WebmasterData.keyword_id,
                func.sum(WebmasterData.impressions).label("impressions"),
                func.sum(WebmasterData.clicks).label("clicks"),
                # Weighted average position by impressions
                case(
                    (
                        func.sum(WebmasterData.impressions) > 0,
                        func.sum(WebmasterData.position * WebmasterData.impressions)
                        / func.sum(WebmasterData.impressions),
                    ),
                    else_=None,
                ).label("avg_position"),
                # Weighted CTR
                case(
                    (
                        func.sum(WebmasterData.impressions) > 0,
                        cast(func.sum(WebmasterData.clicks), Float) / func.sum(WebmasterData.impressions) * 100,
                    ),
                    else_=None,
                ).label("ctr"),
            )
            .where(
                WebmasterData.keyword_id.in_(kw_ids),
                WebmasterData.search_engine == "google",
                WebmasterData.date >= date_from_gsc,
                WebmasterData.tenant_id == tenant_id,
            )
            .group_by(WebmasterData.keyword_id)
        )
        gsc_result = await db.execute(gsc_stmt)
        for r in gsc_result.all():
            gsc_data[r.keyword_id] = {
                "impressions": r.impressions or 0,
                "clicks": r.clicks or 0,
                "position": round(r.avg_position, 1) if r.avg_position else None,
                "ctr": round(r.ctr, 2) if r.ctr else None,
            }

        # Also get latest GSC page URL per keyword
        gsc_url_sub = (
            select(
                WebmasterData.keyword_id,
                func.max(WebmasterData.date).label("max_date"),
            )
            .where(
                WebmasterData.keyword_id.in_(kw_ids),
                WebmasterData.search_engine == "google",
                WebmasterData.tenant_id == tenant_id,
            )
            .group_by(WebmasterData.keyword_id)
            .subquery()
        )
        gsc_url_stmt = (
            select(WebmasterData.keyword_id, WebmasterData.page_url)
            .join(
                gsc_url_sub,
                and_(
                    WebmasterData.keyword_id == gsc_url_sub.c.keyword_id,
                    WebmasterData.date == gsc_url_sub.c.max_date,
                ),
            )
            .where(WebmasterData.search_engine == "google", WebmasterData.tenant_id == tenant_id)
        )
        gsc_url_result = await db.execute(gsc_url_stmt)
        for r in gsc_url_result.all():
            if r.keyword_id in gsc_data:
                gsc_data[r.keyword_id]["page_url"] = r.page_url

    # Build response
    kw_list = []
    for kw_id, kw in kw_map.items():
        snap = latest_snaps.get(kw_id)
        pos = snap.position if snap and snap.position and snap.position > 0 else None
        prev_pos = snap.previous_position if snap and snap.previous_position and snap.previous_position > 0 else None
        found_url = snap.found_url if snap else None

        # Compute change (positive = improved = position went down numerically)
        change = None
        if pos is not None and prev_pos is not None:
            change = prev_pos - pos  # positive = got better

        # URL match check
        match = None
        if found_url and kw.target_url:
            match = found_url == kw.target_url

        # Apply URL filter
        if page_url and found_url != page_url:
            continue

        entry = {
            "keyword_id": kw_id,
            "keyword": kw.keyword,
            "target_url": kw.target_url,
            "position": pos,
            "previous_position": prev_pos,
            "change": change,
            "found_url": found_url,
            "match": match,
        }

        # Add GSC data for Google
        if engine == "google":
            gsc = gsc_data.get(kw_id, {})
            entry["gsc_position"] = gsc.get("position")
            entry["gsc_impressions"] = gsc.get("impressions", 0)
            entry["gsc_clicks"] = gsc.get("clicks", 0)
            entry["gsc_ctr"] = gsc.get("ctr")
            entry["gsc_page_url"] = gsc.get("page_url")

        kw_list.append(entry)

    return {"keywords": kw_list, "all_urls": all_urls}
