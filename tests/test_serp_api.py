"""Tests for SERP API endpoints."""

from datetime import date, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.keyword import Keyword
from app.models.project import Project
from app.models.serp import SerpSnapshot
from app.models.webmaster import WebmasterData


@pytest.fixture
async def project_with_data(db: AsyncSession, tenant_and_user) -> dict:
    """Create a project with keywords and SERP data for testing."""
    tenant, user = tenant_and_user

    project = Project(tenant_id=tenant.id, name="SEO Project", domain="example.com")
    db.add(project)
    await db.flush()

    kw1 = Keyword(
        tenant_id=tenant.id, project_id=project.id, keyword="buy shoes", target_url="https://example.com/shoes"
    )
    kw2 = Keyword(tenant_id=tenant.id, project_id=project.id, keyword="best shoes")
    db.add_all([kw1, kw2])
    await db.flush()

    today = date.today()
    yesterday = today - timedelta(days=1)

    # Create snapshots for multiple days
    snapshots = [
        SerpSnapshot(
            tenant_id=tenant.id,
            keyword_id=kw1.id,
            date=today,
            search_engine="google",
            position=5,
            found_url="https://example.com/shoes",
            previous_position=7,
        ),
        SerpSnapshot(
            tenant_id=tenant.id,
            keyword_id=kw1.id,
            date=yesterday,
            search_engine="google",
            position=7,
            found_url="https://example.com/shoes",
            previous_position=10,
        ),
        SerpSnapshot(
            tenant_id=tenant.id,
            keyword_id=kw2.id,
            date=today,
            search_engine="google",
            position=12,
            found_url="https://example.com/shoes",
            previous_position=15,
        ),
    ]
    db.add_all(snapshots)

    # Add GSC data
    gsc_data = [
        WebmasterData(
            tenant_id=tenant.id,
            keyword_id=kw1.id,
            date=today,
            search_engine="google",
            impressions=100,
            clicks=10,
            ctr=10.0,
            position=4.5,
            page_url="https://example.com/shoes",
        ),
        WebmasterData(
            tenant_id=tenant.id,
            keyword_id=kw1.id,
            date=yesterday,
            search_engine="google",
            impressions=80,
            clicks=8,
            ctr=10.0,
            position=5.0,
            page_url="https://example.com/shoes",
        ),
    ]
    db.add_all(gsc_data)
    await db.commit()

    return {
        "tenant": tenant,
        "project": project,
        "kw1": kw1,
        "kw2": kw2,
    }


@pytest.mark.asyncio
async def test_chart_data_single_keyword(client: AsyncClient, auth_headers, project_with_data):
    """Chart data returns positions for a single keyword."""
    data = project_with_data
    response = await client.get(
        f"/api/v1/serp/chart/{data['project'].id}",
        params={"engine": "google", "days": 7, "keyword_id": data["kw1"].id},
        headers=auth_headers,
    )
    assert response.status_code == 200
    result = response.json()

    assert "labels" in result
    assert "datasets" in result
    assert len(result["labels"]) == 8  # 7 days + today

    # Should have SERP dataset + GSC dataset for single keyword Google
    serp_datasets = [d for d in result["datasets"] if d["type"] == "serp"]
    gsc_datasets = [d for d in result["datasets"] if d["type"] == "gsc"]
    assert len(serp_datasets) == 1
    assert len(gsc_datasets) == 1
    assert serp_datasets[0]["label"] == "buy shoes"


@pytest.mark.asyncio
async def test_chart_data_multiple_keywords(client: AsyncClient, auth_headers, project_with_data):
    """Chart data with multiple keyword IDs (URL mode)."""
    data = project_with_data
    kw_ids = f"{data['kw1'].id},{data['kw2'].id}"
    response = await client.get(
        f"/api/v1/serp/chart/{data['project'].id}",
        params={"engine": "google", "days": 7, "keyword_ids": kw_ids},
        headers=auth_headers,
    )
    assert response.status_code == 200
    result = response.json()

    serp_datasets = [d for d in result["datasets"] if d["type"] == "serp"]
    assert len(serp_datasets) == 2  # Two keywords
    # No GSC overlay in multi-keyword mode
    gsc_datasets = [d for d in result["datasets"] if d["type"] == "gsc"]
    assert len(gsc_datasets) == 0


@pytest.mark.asyncio
async def test_chart_data_no_keyword_ids(client: AsyncClient, auth_headers, project_with_data):
    """Chart data without keyword IDs returns 400."""
    data = project_with_data
    response = await client.get(
        f"/api/v1/serp/chart/{data['project'].id}",
        params={"engine": "google", "days": 7},
        headers=auth_headers,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_project_keywords_with_positions(client: AsyncClient, auth_headers, project_with_data):
    """Project keywords endpoint returns positions and GSC data."""
    data = project_with_data
    response = await client.get(
        f"/api/v1/serp/project-keywords/{data['project'].id}",
        params={"engine": "google"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    result = response.json()

    assert "keywords" in result
    assert "all_urls" in result
    assert len(result["keywords"]) == 2

    # Find kw1 entry
    kw1_entry = next(k for k in result["keywords"] if k["keyword"] == "buy shoes")
    assert kw1_entry["position"] == 5
    assert kw1_entry["change"] == 2  # was 7, now 5 → improved by 2
    assert kw1_entry["found_url"] == "https://example.com/shoes"
    assert kw1_entry["match"] is True  # target_url matches found_url
    assert kw1_entry["gsc_impressions"] > 0
    assert kw1_entry["gsc_clicks"] > 0


@pytest.mark.asyncio
async def test_project_keywords_url_filter(client: AsyncClient, auth_headers, project_with_data):
    """Project keywords filtered by URL."""
    data = project_with_data
    response = await client.get(
        f"/api/v1/serp/project-keywords/{data['project'].id}",
        params={"engine": "google", "page_url": "https://example.com/shoes"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    result = response.json()
    # Both keywords have found_url = "https://example.com/shoes"
    assert len(result["keywords"]) == 2


@pytest.mark.asyncio
async def test_url_keywords(client: AsyncClient, auth_headers, project_with_data):
    """URL-keywords endpoint returns keywords ranking on a specific URL."""
    data = project_with_data
    response = await client.get(
        f"/api/v1/serp/url-keywords/{data['project'].id}",
        params={"page_url": "https://example.com/shoes", "engine": "google"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    result = response.json()
    assert "keywords" in result
    assert len(result["keywords"]) == 2
    # Should be sorted by impressions desc
    assert result["keywords"][0]["keyword"] == "buy shoes"  # has GSC impressions


@pytest.mark.asyncio
async def test_serp_details_no_clickhouse(client: AsyncClient, auth_headers, project_with_data):
    """SERP details returns keyword info even when ClickHouse is unavailable."""
    data = project_with_data
    response = await client.get(
        f"/api/v1/serp/details/{data['kw1'].id}/google",
        headers=auth_headers,
    )
    assert response.status_code == 200
    result = response.json()
    assert result["keyword"] == "buy shoes"
    assert result["search_engine"] == "google"
    # Details may be empty if CH is not available in tests
    assert isinstance(result["details"], list)


@pytest.mark.asyncio
async def test_serp_chart_wrong_project(client: AsyncClient, auth_headers, project_with_data):
    """Chart data for non-existent project returns 404."""
    response = await client.get(
        "/api/v1/serp/chart/99999",
        params={"engine": "google", "days": 7, "keyword_id": 1},
        headers=auth_headers,
    )
    assert response.status_code == 404
