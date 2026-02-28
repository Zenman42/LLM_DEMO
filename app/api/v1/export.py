"""Data export endpoints — CSV export of position and LLM data."""

import csv
import io
import json
import uuid
from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id, require_role
from app.core.exceptions import NotFoundError
from app.core.rate_limit import limiter
from app.db.postgres import get_db
from app.models.keyword import Keyword
from app.models.llm_query import LlmQuery
from app.models.llm_snapshot import LlmSnapshot
from app.models.project import Project
from app.models.serp import SerpSnapshot

router = APIRouter(prefix="/export", tags=["export"])


@router.get(
    "/projects/{project_id}/csv",
    dependencies=[Depends(require_role("member"))],
)
@limiter.limit("10/minute")
async def export_project_csv(
    request: Request,
    project_id: int,
    date_from: date | None = Query(None, description="Start date (default: 30 days ago)"),
    date_to: date | None = Query(None, description="End date (default: today)"),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Export position data as CSV. Requires member role."""
    # Verify project belongs to tenant
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")

    # Default date range: last 30 days
    if date_to is None:
        date_to = date.today()
    if date_from is None:
        date_from = date_to - timedelta(days=30)

    # Query SERP data with keyword info
    stmt = (
        select(
            Keyword.keyword,
            SerpSnapshot.search_engine,
            SerpSnapshot.date,
            SerpSnapshot.position,
            SerpSnapshot.found_url,
        )
        .join(Keyword, Keyword.id == SerpSnapshot.keyword_id)
        .where(
            Keyword.project_id == project_id,
            SerpSnapshot.tenant_id == tenant_id,
            SerpSnapshot.date >= date_from,
            SerpSnapshot.date <= date_to,
        )
        .order_by(Keyword.keyword, SerpSnapshot.date.desc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    # Generate CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["keyword", "search_engine", "date", "position", "url"])

    for row in rows:
        writer.writerow(
            [
                row.keyword,
                row.search_engine,
                row.date.isoformat(),
                row.position if row.position else "",
                row.found_url or "",
            ]
        )

    output.seek(0)

    filename = f"{project.domain}_{date_from}_{date_to}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/projects/{project_id}/llm-csv",
    dependencies=[Depends(require_role("member"))],
)
@limiter.limit("10/minute")
async def export_llm_csv(
    request: Request,
    project_id: int,
    date_from: date | None = Query(None, description="Start date (default: 30 days ago)"),
    date_to: date | None = Query(None, description="End date (default: today)"),
    provider: str | None = Query(None, description="Filter by LLM provider"),
    tenant_id: uuid.UUID = Depends(get_current_tenant_id),
    db: AsyncSession = Depends(get_db),
):
    """Export LLM visibility data as CSV. Requires member role."""
    result = await db.execute(select(Project).where(Project.id == project_id, Project.tenant_id == tenant_id))
    project = result.scalar_one_or_none()
    if not project:
        raise NotFoundError("Project not found")

    if date_to is None:
        date_to = date.today()
    if date_from is None:
        date_from = date_to - timedelta(days=30)

    conditions = [
        LlmQuery.project_id == project_id,
        LlmSnapshot.tenant_id == tenant_id,
        LlmSnapshot.date >= date_from,
        LlmSnapshot.date <= date_to,
    ]
    if provider:
        conditions.append(LlmSnapshot.llm_provider == provider)

    stmt = (
        select(
            LlmQuery.query_text,
            LlmQuery.query_type,
            LlmQuery.target_brand,
            LlmSnapshot.date,
            LlmSnapshot.llm_provider,
            LlmSnapshot.llm_model,
            LlmSnapshot.brand_mentioned,
            LlmSnapshot.mention_type,
            LlmSnapshot.mention_context,
            LlmSnapshot.competitor_mentions,
            LlmSnapshot.cited_urls,
            LlmSnapshot.response_tokens,
            LlmSnapshot.cost_usd,
        )
        .join(LlmQuery, LlmQuery.id == LlmSnapshot.llm_query_id)
        .where(*conditions)
        .order_by(LlmSnapshot.date.desc(), LlmQuery.query_text, LlmSnapshot.llm_provider)
    )

    result = await db.execute(stmt)
    rows = result.all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "query",
            "query_type",
            "target_brand",
            "date",
            "provider",
            "model",
            "brand_mentioned",
            "mention_type",
            "mention_context",
            "competitor_mentions",
            "cited_urls",
            "tokens",
            "cost_usd",
        ]
    )

    for row in rows:
        writer.writerow(
            [
                row.query_text,
                row.query_type,
                row.target_brand or "",
                row.date.isoformat(),
                row.llm_provider,
                row.llm_model,
                row.brand_mentioned,
                row.mention_type,
                row.mention_context or "",
                json.dumps(row.competitor_mentions) if row.competitor_mentions else "",
                json.dumps(row.cited_urls) if row.cited_urls else "",
                row.response_tokens or "",
                f"{row.cost_usd:.6f}" if row.cost_usd else "",
            ]
        )

    output.seek(0)

    name = project.brand_name or project.domain or project.name
    filename = f"llm_{name}_{date_from}_{date_to}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
