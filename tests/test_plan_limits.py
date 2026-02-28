"""Tests for plan limit enforcement."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_create_project_within_limit(client: AsyncClient, auth_headers):
    """Project creation should succeed within plan limits."""
    response = await client.post(
        "/api/v1/projects/",
        json={"name": "Site 1", "domain": "example.com"},
        headers=auth_headers,
    )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_create_project_exceeds_limit(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Project creation should fail when plan limit is reached."""
    tenant, user = tenant_and_user

    # Set max_projects to 1
    tenant.max_projects = 1
    await db.commit()

    # First project should succeed
    resp1 = await client.post(
        "/api/v1/projects/",
        json={"name": "Site 1"},
        headers=auth_headers,
    )
    assert resp1.status_code == 201

    # Second project should fail with 402
    resp2 = await client.post(
        "/api/v1/projects/",
        json={"name": "Site 2"},
        headers=auth_headers,
    )
    assert resp2.status_code == 402
    assert "max" in resp2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_add_keywords_within_limit(client: AsyncClient, auth_headers):
    """Adding keywords within limit should succeed."""
    # Create project first
    proj = await client.post(
        "/api/v1/projects/",
        json={"name": "Site 1", "domain": "example.com"},
        headers=auth_headers,
    )
    project_id = proj.json()["id"]

    # Add keyword
    response = await client.post(
        f"/api/v1/projects/{project_id}/keywords/single",
        json={"keyword": "test keyword"},
        headers=auth_headers,
    )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_add_keywords_exceeds_limit(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Adding keywords should fail when tenant keyword limit is reached."""
    tenant, user = tenant_and_user

    # Set max_keywords to 2
    tenant.max_keywords = 2
    await db.commit()

    # Create project
    proj = await client.post(
        "/api/v1/projects/",
        json={"name": "Site 1"},
        headers=auth_headers,
    )
    project_id = proj.json()["id"]

    # Add 2 keywords (should succeed)
    resp1 = await client.post(
        f"/api/v1/projects/{project_id}/keywords/",
        json={"keywords": [{"keyword": "kw1"}, {"keyword": "kw2"}]},
        headers=auth_headers,
    )
    assert resp1.status_code == 201

    # Add 1 more (should fail with 402)
    resp2 = await client.post(
        f"/api/v1/projects/{project_id}/keywords/single",
        json={"keyword": "kw3"},
        headers=auth_headers,
    )
    assert resp2.status_code == 402
    assert "max" in resp2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_bulk_add_keywords_exceeds_limit(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Bulk keyword add should fail if total would exceed limit."""
    tenant, user = tenant_and_user
    tenant.max_keywords = 3
    await db.commit()

    proj = await client.post(
        "/api/v1/projects/",
        json={"name": "Site 1"},
        headers=auth_headers,
    )
    project_id = proj.json()["id"]

    # Try to add 4 keywords at once (limit is 3)
    resp = await client.post(
        f"/api/v1/projects/{project_id}/keywords/",
        json={"keywords": [{"keyword": f"kw{i}"} for i in range(4)]},
        headers=auth_headers,
    )
    assert resp.status_code == 402
