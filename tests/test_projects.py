import pytest
from httpx import AsyncClient

from app.core.security import create_access_token
from app.models.tenant import Tenant
from app.models.user import User


@pytest.mark.asyncio
async def test_create_project(client: AsyncClient, auth_headers):
    response = await client.post(
        "/api/v1/projects/",
        json={"name": "My Site", "domain": "example.com"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "My Site"
    assert data["domain"] == "example.com"
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_list_projects(client: AsyncClient, auth_headers):
    # Create 2 projects
    await client.post("/api/v1/projects/", json={"name": "Site A"}, headers=auth_headers)
    await client.post("/api/v1/projects/", json={"name": "Site B"}, headers=auth_headers)

    response = await client.get("/api/v1/projects/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


@pytest.mark.asyncio
async def test_get_project(client: AsyncClient, auth_headers):
    create_resp = await client.post("/api/v1/projects/", json={"name": "Test"}, headers=auth_headers)
    project_id = create_resp.json()["id"]

    response = await client.get(f"/api/v1/projects/{project_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["name"] == "Test"


@pytest.mark.asyncio
async def test_update_project(client: AsyncClient, auth_headers):
    create_resp = await client.post("/api/v1/projects/", json={"name": "Old Name"}, headers=auth_headers)
    project_id = create_resp.json()["id"]

    response = await client.put(
        f"/api/v1/projects/{project_id}",
        json={"name": "New Name"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["name"] == "New Name"


@pytest.mark.asyncio
async def test_delete_project(client: AsyncClient, auth_headers):
    create_resp = await client.post("/api/v1/projects/", json={"name": "To Delete"}, headers=auth_headers)
    project_id = create_resp.json()["id"]

    response = await client.delete(f"/api/v1/projects/{project_id}", headers=auth_headers)
    assert response.status_code == 204

    get_resp = await client.get(f"/api/v1/projects/{project_id}", headers=auth_headers)
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_tenant_isolation(client: AsyncClient, db, auth_headers):
    """Tenant A cannot see Tenant B's projects."""
    from app.core.security import hash_password

    # Create project as Tenant A
    await client.post("/api/v1/projects/", json={"name": "Tenant A Project"}, headers=auth_headers)

    # Create Tenant B
    tenant_b = Tenant(name="Other Corp", slug="other-corp")
    db.add(tenant_b)
    await db.flush()
    user_b = User(
        tenant_id=tenant_b.id,
        email="other@example.com",
        password_hash=hash_password("otherpass123"),
        role="owner",
    )
    db.add(user_b)
    await db.commit()

    token_b = create_access_token(user_b.id, tenant_b.id)
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # Tenant B should see 0 projects
    response = await client.get("/api/v1/projects/", headers=headers_b)
    assert response.status_code == 200
    assert len(response.json()) == 0
