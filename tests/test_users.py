"""Tests for user management API."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password
from app.models.user import User


@pytest.mark.asyncio
async def test_list_users(client: AsyncClient, auth_headers):
    """Admin/owner should see all users in tenant."""
    response = await client.get("/api/v1/users/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert len(data["items"]) >= 1
    assert data["items"][0]["email"] == "test@example.com"


@pytest.mark.asyncio
async def test_invite_user(client: AsyncClient, auth_headers):
    """Admin should be able to invite a new user."""
    response = await client.post(
        "/api/v1/users/invite",
        json={"email": "newuser@example.com", "role": "member"},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert data["role"] == "member"
    assert "temporary_password" in data
    assert len(data["temporary_password"]) > 10


@pytest.mark.asyncio
async def test_invite_duplicate_email(client: AsyncClient, auth_headers):
    """Inviting with existing email should return 409."""
    # First invite
    await client.post(
        "/api/v1/users/invite",
        json={"email": "dup@example.com", "role": "member"},
        headers=auth_headers,
    )

    # Duplicate
    response = await client.post(
        "/api/v1/users/invite",
        json={"email": "dup@example.com", "role": "viewer"},
        headers=auth_headers,
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_invite_as_member_forbidden(client: AsyncClient, db: AsyncSession, tenant_and_user):
    """Member role should not be able to invite users."""
    tenant, owner = tenant_and_user

    # Create a member user
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
        "/api/v1/users/invite",
        json={"email": "another@example.com", "role": "member"},
        headers=headers,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_change_role(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Owner should be able to change user roles."""
    tenant, owner = tenant_and_user

    # Create a member
    member = User(
        tenant_id=tenant.id,
        email="changeme@example.com",
        password_hash=hash_password("pass123"),
        role="member",
    )
    db.add(member)
    await db.commit()

    response = await client.patch(
        f"/api/v1/users/{member.id}/role",
        json={"role": "admin"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["role"] == "admin"


@pytest.mark.asyncio
async def test_change_role_to_owner_rejected(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Setting role to 'owner' should be rejected by validation."""
    tenant, owner = tenant_and_user

    member = User(
        tenant_id=tenant.id,
        email="noowner@example.com",
        password_hash=hash_password("pass123"),
        role="member",
    )
    db.add(member)
    await db.commit()

    response = await client.patch(
        f"/api/v1/users/{member.id}/role",
        json={"role": "owner"},
        headers=auth_headers,
    )
    assert response.status_code == 422  # pydantic validation error (pattern mismatch)


@pytest.mark.asyncio
async def test_deactivate_user(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Admin/owner should be able to deactivate a user."""
    tenant, owner = tenant_and_user

    member = User(
        tenant_id=tenant.id,
        email="deactivate@example.com",
        password_hash=hash_password("pass123"),
        role="member",
    )
    db.add(member)
    await db.commit()

    response = await client.delete(
        f"/api/v1/users/{member.id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "deactivated" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_cannot_deactivate_owner(client: AsyncClient, auth_headers, db: AsyncSession, tenant_and_user):
    """Cannot deactivate the owner user."""
    tenant, owner = tenant_and_user

    # Create admin who tries to deactivate owner
    admin = User(
        tenant_id=tenant.id,
        email="admin@example.com",
        password_hash=hash_password("pass123"),
        role="admin",
    )
    db.add(admin)
    await db.commit()

    admin_token = create_access_token(admin.id, tenant.id)
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    response = await client.delete(
        f"/api/v1/users/{owner.id}",
        headers=admin_headers,
    )
    assert response.status_code == 400
    assert "owner" in response.json()["detail"].lower()
