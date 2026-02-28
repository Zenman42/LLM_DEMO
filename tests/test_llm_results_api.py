"""Tests for LLM Results API — dashboard, chart, details, citations."""

from datetime import date

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.llm_query import LlmQuery
from app.models.llm_snapshot import LlmSnapshot
from app.models.project import Project


@pytest.fixture
async def llm_project_with_data(db: AsyncSession, tenant_and_user):
    """Create project + queries + snapshots for LLM results testing."""
    tenant, user = tenant_and_user

    project = Project(
        tenant_id=tenant.id,
        name="LLM Results Test",
        domain="example.com",
        track_llm=True,
        brand_name="TestBrand",
        llm_providers=["chatgpt"],
    )
    db.add(project)
    await db.flush()

    query1 = LlmQuery(
        tenant_id=tenant.id,
        project_id=project.id,
        query_text="What is TestBrand?",
        query_type="brand_check",
        target_brand="TestBrand",
        competitors=["CompA", "CompB"],
    )
    query2 = LlmQuery(
        tenant_id=tenant.id,
        project_id=project.id,
        query_text="Best tools in the industry?",
        query_type="recommendation",
        target_brand="TestBrand",
    )
    db.add_all([query1, query2])
    await db.flush()

    # Add snapshots
    today = date.today()
    snapshots = [
        LlmSnapshot(
            tenant_id=tenant.id,
            llm_query_id=query1.id,
            date=today,
            llm_provider="chatgpt",
            llm_model="gpt-4o-mini",
            brand_mentioned=True,
            mention_type="recommended",
            mention_context="TestBrand is recommended for its features.",
            competitor_mentions={"CompA": True, "CompB": False},
            cited_urls=["https://testbrand.com", "https://example.com"],
            raw_response="TestBrand is recommended...",
            response_tokens=100,
            cost_usd=0.001,
        ),
        LlmSnapshot(
            tenant_id=tenant.id,
            llm_query_id=query2.id,
            date=today,
            llm_provider="chatgpt",
            llm_model="gpt-4o-mini",
            brand_mentioned=False,
            mention_type="none",
            competitor_mentions=None,
            cited_urls=["https://example.com"],
            raw_response="Here are some tools...",
            response_tokens=80,
            cost_usd=0.0008,
        ),
    ]
    db.add_all(snapshots)
    await db.commit()

    return {"project": project, "queries": [query1, query2], "tenant": tenant}


@pytest.mark.asyncio
async def test_llm_dashboard(client: AsyncClient, auth_headers, llm_project_with_data):
    pid = llm_project_with_data["project"].id
    resp = await client.get(f"/api/v1/llm/dashboard/{pid}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_queries"] == 2
    assert data["total_checks"] == 2
    assert 0 <= data["brand_mention_rate"] <= 1
    # 1 out of 2 mentioned => 0.5
    assert data["brand_mention_rate"] == 0.5
    assert "chatgpt" in data["mention_rate_by_provider"]
    assert data["total_cost_usd"] > 0
    assert isinstance(data["top_cited_urls"], list)


@pytest.mark.asyncio
async def test_llm_dashboard_nonexistent_project(client: AsyncClient, auth_headers):
    resp = await client.get("/api/v1/llm/dashboard/99999", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_llm_chart(client: AsyncClient, auth_headers, llm_project_with_data):
    pid = llm_project_with_data["project"].id
    resp = await client.get(f"/api/v1/llm/chart/{pid}?days=7", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "labels" in data
    assert "datasets" in data
    assert isinstance(data["labels"], list)
    assert len(data["labels"]) == 8  # 7 days + today


@pytest.mark.asyncio
async def test_llm_chart_with_provider_filter(client: AsyncClient, auth_headers, llm_project_with_data):
    pid = llm_project_with_data["project"].id
    resp = await client.get(f"/api/v1/llm/chart/{pid}?days=7&provider=chatgpt", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    for ds in data["datasets"]:
        assert ds["label"] == "chatgpt"


@pytest.mark.asyncio
async def test_llm_details(client: AsyncClient, auth_headers, llm_project_with_data):
    pid = llm_project_with_data["project"].id
    resp = await client.get(f"/api/v1/llm/details/{pid}?days=7", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


@pytest.mark.asyncio
async def test_llm_details_with_query_filter(client: AsyncClient, auth_headers, llm_project_with_data):
    pid = llm_project_with_data["project"].id
    qid = llm_project_with_data["queries"][0].id
    resp = await client.get(
        f"/api/v1/llm/details/{pid}?days=7&query_id={qid}",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


@pytest.mark.asyncio
async def test_llm_citations(client: AsyncClient, auth_headers, llm_project_with_data):
    pid = llm_project_with_data["project"].id
    resp = await client.get(f"/api/v1/llm/citations/{pid}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    assert len(data["citations"]) > 0
    # https://example.com is cited in both snapshots
    urls = [c["url"] for c in data["citations"]]
    assert "https://example.com" in urls
    # Check the most-cited URL
    top = data["citations"][0]
    assert top["count"] >= 1
    assert top["domain"] != ""
    assert "chatgpt" in top["providers"]
