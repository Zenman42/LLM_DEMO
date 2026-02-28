"""Tests for LLM Queries CRUD API."""

import pytest
from httpx import AsyncClient


@pytest.fixture
async def project_with_llm(client: AsyncClient, auth_headers):
    """Create a project with LLM tracking enabled."""
    resp = await client.post(
        "/api/v1/projects/",
        json={
            "name": "LLM Test Project",
            "domain": "example.com",
            "track_llm": True,
            "brand_name": "TestBrand",
            "llm_providers": ["chatgpt"],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    return resp.json()


@pytest.mark.asyncio
async def test_add_llm_query_single(client: AsyncClient, auth_headers, project_with_llm):
    pid = project_with_llm["id"]
    resp = await client.post(
        f"/api/v1/projects/{pid}/llm-queries/single",
        json={
            "query_text": "What is the best SEO tool?",
            "query_type": "recommendation",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["query_text"] == "What is the best SEO tool?"
    assert data["query_type"] == "recommendation"
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_add_llm_queries_bulk(client: AsyncClient, auth_headers, project_with_llm):
    pid = project_with_llm["id"]
    resp = await client.post(
        f"/api/v1/projects/{pid}/llm-queries/",
        json={
            "queries": [
                {"query_text": "What is TestBrand?", "query_type": "brand_check"},
                {"query_text": "Compare TestBrand vs Competitor", "query_type": "comparison"},
            ]
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["created"] == 2
    assert data["skipped"] == 0


@pytest.mark.asyncio
async def test_add_duplicate_query_skipped(client: AsyncClient, auth_headers, project_with_llm):
    pid = project_with_llm["id"]
    body = {"queries": [{"query_text": "Unique query text"}]}
    await client.post(f"/api/v1/projects/{pid}/llm-queries/", json=body, headers=auth_headers)
    resp2 = await client.post(f"/api/v1/projects/{pid}/llm-queries/", json=body, headers=auth_headers)
    data = resp2.json()
    assert data["created"] == 0
    assert data["skipped"] == 1


@pytest.mark.asyncio
async def test_list_llm_queries(client: AsyncClient, auth_headers, project_with_llm):
    pid = project_with_llm["id"]
    await client.post(
        f"/api/v1/projects/{pid}/llm-queries/",
        json={"queries": [{"query_text": "Query A"}, {"query_text": "Query B"}]},
        headers=auth_headers,
    )
    resp = await client.get(f"/api/v1/projects/{pid}/llm-queries/", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


@pytest.mark.asyncio
async def test_update_llm_query(client: AsyncClient, auth_headers, project_with_llm):
    pid = project_with_llm["id"]
    create_resp = await client.post(
        f"/api/v1/projects/{pid}/llm-queries/single",
        json={"query_text": "Old query"},
        headers=auth_headers,
    )
    query_id = create_resp.json()["id"]

    resp = await client.put(
        f"/api/v1/projects/{pid}/llm-queries/{query_id}",
        json={"is_active": False},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False


@pytest.mark.asyncio
async def test_delete_llm_query(client: AsyncClient, auth_headers, project_with_llm):
    pid = project_with_llm["id"]
    create_resp = await client.post(
        f"/api/v1/projects/{pid}/llm-queries/single",
        json={"query_text": "To be deleted"},
        headers=auth_headers,
    )
    query_id = create_resp.json()["id"]

    resp = await client.delete(
        f"/api/v1/projects/{pid}/llm-queries/{query_id}",
        headers=auth_headers,
    )
    assert resp.status_code == 204

    # Verify deleted
    list_resp = await client.get(f"/api/v1/projects/{pid}/llm-queries/", headers=auth_headers)
    assert list_resp.json()["total"] == 0


@pytest.mark.asyncio
async def test_delete_nonexistent_query_404(client: AsyncClient, auth_headers, project_with_llm):
    pid = project_with_llm["id"]
    resp = await client.delete(
        f"/api/v1/projects/{pid}/llm-queries/99999",
        headers=auth_headers,
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_query_with_competitors(client: AsyncClient, auth_headers, project_with_llm):
    pid = project_with_llm["id"]
    resp = await client.post(
        f"/api/v1/projects/{pid}/llm-queries/single",
        json={
            "query_text": "What is the best SEO tool?",
            "query_type": "comparison",
            "competitors": ["SEMrush", "Moz"],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["competitors"] == ["SEMrush", "Moz"]
