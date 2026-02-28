"""Tests for keywords API with pagination."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_add_and_list_keywords_paginated(client: AsyncClient, auth_headers):
    """Keywords list returns paginated response."""
    # Create project
    proj = await client.post("/api/v1/projects/", json={"name": "KW Test"}, headers=auth_headers)
    project_id = proj.json()["id"]

    # Add keywords in bulk
    keywords = [{"keyword": f"keyword {i}"} for i in range(15)]
    resp = await client.post(
        f"/api/v1/projects/{project_id}/keywords/",
        json={"keywords": keywords},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    assert resp.json()["created"] == 15

    # List page 1 (limit=10)
    resp = await client.get(
        f"/api/v1/projects/{project_id}/keywords/",
        params={"limit": 10, "offset": 0},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 15
    assert len(data["items"]) == 10
    assert data["offset"] == 0
    assert data["limit"] == 10

    # List page 2 (offset=10, limit=10)
    resp = await client.get(
        f"/api/v1/projects/{project_id}/keywords/",
        params={"limit": 10, "offset": 10},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 15
    assert len(data["items"]) == 5  # remaining 5


@pytest.mark.asyncio
async def test_add_single_keyword(client: AsyncClient, auth_headers):
    """Single keyword add works."""
    proj = await client.post("/api/v1/projects/", json={"name": "Single KW"}, headers=auth_headers)
    project_id = proj.json()["id"]

    resp = await client.post(
        f"/api/v1/projects/{project_id}/keywords/single",
        json={"keyword": "test keyword", "target_url": "https://example.com"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    assert resp.json()["keyword"] == "test keyword"
    assert resp.json()["target_url"] == "https://example.com"


@pytest.mark.asyncio
async def test_update_keyword(client: AsyncClient, auth_headers):
    """Keyword update works."""
    proj = await client.post("/api/v1/projects/", json={"name": "Upd KW"}, headers=auth_headers)
    project_id = proj.json()["id"]

    add_resp = await client.post(
        f"/api/v1/projects/{project_id}/keywords/single",
        json={"keyword": "my keyword"},
        headers=auth_headers,
    )
    keyword_id = add_resp.json()["id"]

    resp = await client.put(
        f"/api/v1/projects/{project_id}/keywords/{keyword_id}",
        json={"target_url": "https://new-url.com"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["target_url"] == "https://new-url.com"


@pytest.mark.asyncio
async def test_delete_keyword(client: AsyncClient, auth_headers):
    """Keyword delete works."""
    proj = await client.post("/api/v1/projects/", json={"name": "Del KW"}, headers=auth_headers)
    project_id = proj.json()["id"]

    add_resp = await client.post(
        f"/api/v1/projects/{project_id}/keywords/single",
        json={"keyword": "to delete"},
        headers=auth_headers,
    )
    keyword_id = add_resp.json()["id"]

    resp = await client.delete(
        f"/api/v1/projects/{project_id}/keywords/{keyword_id}",
        headers=auth_headers,
    )
    assert resp.status_code == 204

    # Verify it's gone
    list_resp = await client.get(
        f"/api/v1/projects/{project_id}/keywords/",
        headers=auth_headers,
    )
    assert list_resp.json()["total"] == 0
