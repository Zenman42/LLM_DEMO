"""Tests for Webmaster API endpoints."""

from datetime import date, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.keyword import Keyword
from app.models.project import Project
from app.models.webmaster import WebmasterData


@pytest.fixture
async def keyword_with_gsc(db: AsyncSession, tenant_and_user) -> dict:
    """Create a keyword with GSC data for testing."""
    tenant, user = tenant_and_user

    project = Project(tenant_id=tenant.id, name="GSC Project", domain="example.com")
    db.add(project)
    await db.flush()

    kw = Keyword(tenant_id=tenant.id, project_id=project.id, keyword="seo tool")
    db.add(kw)
    await db.flush()

    today = date.today()
    # Create 7 days of GSC data
    gsc_rows = []
    for i in range(7):
        d = today - timedelta(days=i)
        gsc_rows.append(
            WebmasterData(
                tenant_id=tenant.id,
                keyword_id=kw.id,
                date=d,
                search_engine="google",
                impressions=100 + i * 10,
                clicks=10 + i,
                ctr=10.0 + i,
                position=5.0 - i * 0.1,
                page_url="https://example.com/seo",
            )
        )
    db.add_all(gsc_rows)
    await db.commit()

    return {"tenant": tenant, "project": project, "keyword": kw}


@pytest.mark.asyncio
async def test_gsc_daily(client: AsyncClient, auth_headers, keyword_with_gsc):
    """GSC daily endpoint returns day-by-day data."""
    data = keyword_with_gsc
    response = await client.get(
        f"/api/v1/webmaster/gsc/{data['keyword'].id}",
        params={"days": 30},
        headers=auth_headers,
    )
    assert response.status_code == 200
    result = response.json()

    assert result["keyword"] == "seo tool"
    assert len(result["days"]) == 7  # We created 7 days of data
    # Verify first entry (most recent — ordered desc)
    first = result["days"][0]
    assert "date" in first
    assert "position" in first
    assert "impressions" in first
    assert "clicks" in first
    assert "ctr" in first
    assert "url" in first
    assert first["url"] == "https://example.com/seo"


@pytest.mark.asyncio
async def test_gsc_daily_not_found(client: AsyncClient, auth_headers):
    """GSC daily for non-existent keyword returns 404."""
    response = await client.get(
        "/api/v1/webmaster/gsc/99999",
        params={"days": 30},
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_gsc_semantics_no_gsc_url(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """GSC semantics returns 400 when GSC URL not configured."""
    tenant, user = tenant_and_user

    project = Project(tenant_id=tenant.id, name="No GSC", domain="example.com")
    db.add(project)
    await db.commit()

    response = await client.get(
        f"/api/v1/webmaster/gsc-semantics/{project.id}",
        headers=auth_headers,
    )
    assert response.status_code == 400
