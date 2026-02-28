"""Tests for API key management."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password
from app.models.tenant import Tenant
from app.models.user import User


@pytest.mark.asyncio
async def test_create_api_key(client: AsyncClient, auth_headers):
    """Admin/owner should be able to create an API key."""
    response = await client.post(
        "/api/v1/api-keys/",
        json={"name": "My Integration"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "My Integration"
    assert data["key"].startswith("pt_")
    assert len(data["key"]) > 20
    assert "id" in data


@pytest.mark.asyncio
async def test_list_api_keys(client: AsyncClient, auth_headers):
    """Should return masked keys."""
    # Create a key first
    await client.post(
        "/api/v1/api-keys/",
        json={"name": "Key 1"},
        headers=auth_headers,
    )

    response = await client.get("/api/v1/api-keys/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1
    item = data["items"][0]
    assert item["name"] == "Key 1"
    assert item["key_prefix"].endswith("...")
    # Full key should NOT be in list response
    assert "key" not in item or item.get("key") is None


@pytest.mark.asyncio
async def test_create_api_key_as_member_forbidden(client: AsyncClient, db: AsyncSession, tenant_and_user):
    """Member role should not be able to create API keys."""
    tenant, owner = tenant_and_user

    member = User(
        tenant_id=tenant.id,
        email="member@example.com",
        password_hash=hash_password("memberpass123"),
        role="member",
    )
    db.add(member)
    await db.commit()

    token = create_access_token(member.id, tenant.id)
    headers = {"Authorization": f"Bearer {token}"}

    response = await client.post(
        "/api/v1/api-keys/",
        json={"name": "Should Fail"},
        headers=headers,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_delete_api_key(client: AsyncClient, auth_headers):
    """Admin/owner should be able to delete an API key."""
    # Create
    create_resp = await client.post(
        "/api/v1/api-keys/",
        json={"name": "To Delete"},
        headers=auth_headers,
    )
    key_id = create_resp.json()["id"]

    # Delete
    response = await client.delete(f"/api/v1/api-keys/{key_id}", headers=auth_headers)
    assert response.status_code == 200
    assert "deleted" in response.json()["message"].lower()

    # Verify gone
    list_resp = await client.get("/api/v1/api-keys/", headers=auth_headers)
    assert list_resp.json()["total"] == 0


@pytest.mark.asyncio
async def test_delete_nonexistent_key(client: AsyncClient, auth_headers):
    """Deleting a non-existent key should return 404."""
    import uuid

    fake_id = str(uuid.uuid4())
    response = await client.delete(f"/api/v1/api-keys/{fake_id}", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_key_tenant_isolation(client: AsyncClient, db: AsyncSession, auth_headers, tenant_and_user):
    """Tenant should not see another tenant's API keys."""
    # Create key for tenant A
    await client.post(
        "/api/v1/api-keys/",
        json={"name": "Tenant A Key"},
        headers=auth_headers,
    )

    # Create tenant B
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

    # Tenant B should see 0 keys
    response = await client.get("/api/v1/api-keys/", headers=headers_b)
    assert response.status_code == 200
    assert response.json()["total"] == 0
