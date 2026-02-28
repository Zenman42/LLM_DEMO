"""Tests for data export API."""

from datetime import date, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.keyword import Keyword
from app.models.llm_query import LlmQuery
from app.models.llm_snapshot import LlmSnapshot
from app.models.project import Project
from app.models.serp import SerpSnapshot


@pytest.mark.asyncio
async def test_export_csv_success(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Should export CSV with position data."""
    tenant, user = tenant_and_user

    # Create project and keyword
    project = Project(tenant_id=tenant.id, name="Export Test", domain="example.com")
    db.add(project)
    await db.flush()

    keyword = Keyword(tenant_id=tenant.id, project_id=project.id, keyword="test keyword")
    db.add(keyword)
    await db.flush()

    # Add SERP snapshot
    snapshot = SerpSnapshot(
        tenant_id=tenant.id,
        keyword_id=keyword.id,
        date=date.today(),
        search_engine="google",
        position=5,
        found_url="https://example.com/page",
    )
    db.add(snapshot)
    await db.commit()

    response = await client.get(
        f"/api/v1/export/projects/{project.id}/csv",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers.get("content-disposition", "")

    # Parse CSV content (csv module uses \r\n line endings)
    content = response.text.replace("\r\n", "\n").strip()
    lines = content.split("\n")
    assert len(lines) == 2  # header + 1 data row
    assert lines[0] == "keyword,search_engine,date,position,url"
    assert "test keyword" in lines[1]
    assert "google" in lines[1]
    assert "5" in lines[1]


@pytest.mark.asyncio
async def test_export_csv_empty(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Empty project should return CSV with only headers."""
    tenant, user = tenant_and_user

    project = Project(tenant_id=tenant.id, name="Empty Project", domain="empty.com")
    db.add(project)
    await db.commit()

    response = await client.get(
        f"/api/v1/export/projects/{project.id}/csv",
        headers=auth_headers,
    )
    assert response.status_code == 200
    content = response.text.replace("\r\n", "\n").strip()
    lines = content.split("\n")
    assert len(lines) == 1  # only header
    assert lines[0] == "keyword,search_engine,date,position,url"


@pytest.mark.asyncio
async def test_export_csv_date_filter(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Date filter should work correctly."""
    tenant, user = tenant_and_user

    project = Project(tenant_id=tenant.id, name="Date Filter Test", domain="filter.com")
    db.add(project)
    await db.flush()

    keyword = Keyword(tenant_id=tenant.id, project_id=project.id, keyword="filtered kw")
    db.add(keyword)
    await db.flush()

    today = date.today()
    old_date = today - timedelta(days=60)

    # Add old snapshot (outside default 30-day range)
    old_snap = SerpSnapshot(
        tenant_id=tenant.id,
        keyword_id=keyword.id,
        date=old_date,
        search_engine="yandex",
        position=10,
    )
    # Add recent snapshot
    recent_snap = SerpSnapshot(
        tenant_id=tenant.id,
        keyword_id=keyword.id,
        date=today,
        search_engine="yandex",
        position=3,
    )
    db.add_all([old_snap, recent_snap])
    await db.commit()

    # Default range (30 days) should only return recent
    response = await client.get(
        f"/api/v1/export/projects/{project.id}/csv",
        headers=auth_headers,
    )
    assert response.status_code == 200
    content = response.text.replace("\r\n", "\n").strip()
    lines = content.split("\n")
    assert len(lines) == 2  # header + 1 recent row

    # Explicit wide range should return both
    response2 = await client.get(
        f"/api/v1/export/projects/{project.id}/csv",
        params={"date_from": str(old_date), "date_to": str(today)},
        headers=auth_headers,
    )
    assert response2.status_code == 200
    content2 = response2.text.replace("\r\n", "\n").strip()
    lines2 = content2.split("\n")
    assert len(lines2) == 3  # header + 2 rows


@pytest.mark.asyncio
async def test_export_wrong_project(client: AsyncClient, auth_headers):
    """Exporting from non-existent project should return 404."""
    response = await client.get(
        "/api/v1/export/projects/99999/csv",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_export_csv_content_headers(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Response should have correct content-type and disposition."""
    tenant, user = tenant_and_user

    project = Project(tenant_id=tenant.id, name="Headers Test", domain="headers.com")
    db.add(project)
    await db.commit()

    response = await client.get(
        f"/api/v1/export/projects/{project.id}/csv",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "headers.com" in response.headers["content-disposition"]
    assert ".csv" in response.headers["content-disposition"]


# ---- LLM CSV Export Tests ----


@pytest.mark.asyncio
async def test_export_llm_csv_success(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Should export LLM visibility data as CSV."""
    tenant, user = tenant_and_user

    project = Project(tenant_id=tenant.id, name="LLM Export", domain="llm.com", track_llm=True, brand_name="TestBrand")
    db.add(project)
    await db.flush()

    query = LlmQuery(
        tenant_id=tenant.id,
        project_id=project.id,
        query_text="best seo tool",
        query_type="brand_check",
        target_brand="TestBrand",
        competitors=["CompA", "CompB"],
    )
    db.add(query)
    await db.flush()

    snapshot = LlmSnapshot(
        tenant_id=tenant.id,
        llm_query_id=query.id,
        date=date.today(),
        llm_provider="chatgpt",
        llm_model="gpt-4o-mini",
        brand_mentioned=True,
        mention_type="recommended",
        mention_context="TestBrand is recommended",
        competitor_mentions={"CompA": True, "CompB": False},
        cited_urls=["https://testbrand.com", "https://example.com"],
        response_tokens=150,
        cost_usd=0.001,
    )
    db.add(snapshot)
    await db.commit()

    response = await client.get(
        f"/api/v1/export/projects/{project.id}/llm-csv",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers.get("content-disposition", "")

    content = response.text.replace("\r\n", "\n").strip()
    lines = content.split("\n")
    assert len(lines) == 2  # header + 1 data row
    assert "query,query_type,target_brand" in lines[0]
    assert "best seo tool" in lines[1]
    assert "chatgpt" in lines[1]
    assert "True" in lines[1]
    assert "recommended" in lines[1]


@pytest.mark.asyncio
async def test_export_llm_csv_empty(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Empty LLM project should return CSV with only headers."""
    tenant, user = tenant_and_user

    project = Project(tenant_id=tenant.id, name="Empty LLM", domain="empty-llm.com", track_llm=True)
    db.add(project)
    await db.commit()

    response = await client.get(
        f"/api/v1/export/projects/{project.id}/llm-csv",
        headers=auth_headers,
    )
    assert response.status_code == 200
    content = response.text.replace("\r\n", "\n").strip()
    lines = content.split("\n")
    assert len(lines) == 1
    assert "query,query_type,target_brand" in lines[0]


@pytest.mark.asyncio
async def test_export_llm_csv_provider_filter(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Provider filter should return only matching rows."""
    tenant, user = tenant_and_user

    project = Project(tenant_id=tenant.id, name="Filter LLM", domain="filter-llm.com", track_llm=True)
    db.add(project)
    await db.flush()

    query = LlmQuery(tenant_id=tenant.id, project_id=project.id, query_text="test query", query_type="custom")
    db.add(query)
    await db.flush()

    today = date.today()
    snap1 = LlmSnapshot(
        tenant_id=tenant.id,
        llm_query_id=query.id,
        date=today,
        llm_provider="chatgpt",
        llm_model="gpt-4o-mini",
        brand_mentioned=False,
        mention_type="none",
    )
    snap2 = LlmSnapshot(
        tenant_id=tenant.id,
        llm_query_id=query.id,
        date=today,
        llm_provider="perplexity",
        llm_model="sonar",
        brand_mentioned=True,
        mention_type="direct",
    )
    db.add_all([snap1, snap2])
    await db.commit()

    # No filter — both rows
    resp_all = await client.get(
        f"/api/v1/export/projects/{project.id}/llm-csv",
        headers=auth_headers,
    )
    content_all = resp_all.text.replace("\r\n", "\n").strip()
    assert len(content_all.split("\n")) == 3  # header + 2

    # Filter by perplexity
    resp_pplx = await client.get(
        f"/api/v1/export/projects/{project.id}/llm-csv",
        params={"provider": "perplexity"},
        headers=auth_headers,
    )
    content_pplx = resp_pplx.text.replace("\r\n", "\n").strip()
    lines = content_pplx.split("\n")
    assert len(lines) == 2  # header + 1
    assert "perplexity" in lines[1]
    assert "chatgpt" not in lines[1]


@pytest.mark.asyncio
async def test_export_llm_csv_wrong_project(client: AsyncClient, auth_headers):
    """Non-existent project should return 404."""
    response = await client.get(
        "/api/v1/export/projects/99999/llm-csv",
        headers=auth_headers,
    )
    assert response.status_code == 404
