"""Tests for dashboard API endpoint."""

from datetime import date

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.keyword import Keyword
from app.models.serp import SerpSnapshot


@pytest.mark.asyncio
async def test_dashboard_empty(client: AsyncClient, auth_headers):
    """Dashboard with no projects returns empty list."""
    response = await client.get("/api/v1/dashboard/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["projects"] == []


@pytest.mark.asyncio
async def test_dashboard_with_project(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Dashboard returns project stats correctly."""
    tenant, user = tenant_and_user

    # Create project via API
    proj_resp = await client.post(
        "/api/v1/projects/",
        json={"name": "Test Site", "domain": "example.com"},
        headers=auth_headers,
    )
    project_id = proj_resp.json()["id"]

    # Add keywords directly via DB
    kw1 = Keyword(tenant_id=tenant.id, project_id=project_id, keyword="buy laptop")
    kw2 = Keyword(tenant_id=tenant.id, project_id=project_id, keyword="best laptop")
    db.add_all([kw1, kw2])
    await db.flush()

    # Add SERP snapshots for today
    today = date.today()
    snap1 = SerpSnapshot(
        tenant_id=tenant.id,
        keyword_id=kw1.id,
        date=today,
        search_engine="google",
        position=5,
        found_url="https://example.com/laptops",
    )
    snap2 = SerpSnapshot(
        tenant_id=tenant.id,
        keyword_id=kw2.id,
        date=today,
        search_engine="google",
        position=10,
        found_url="https://example.com/laptops",
    )
    db.add_all([snap1, snap2])
    await db.commit()

    response = await client.get("/api/v1/dashboard/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["projects"]) == 1

    proj = data["projects"][0]
    assert proj["name"] == "Test Site"
    assert proj["keyword_count"] == 2
    assert proj["avg_position"] == 7.5  # (5+10)/2
    assert proj["last_collected"] == today.isoformat()
