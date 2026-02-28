"""Tests for account API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_account(client: AsyncClient, auth_headers):
    """Should return profile with tenant info."""
    response = await client.get("/api/v1/account/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["role"] == "owner"
    assert data["tenant_name"] == "Test Corp"
    assert data["tenant_slug"] == "test-corp"
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_change_password_success(client: AsyncClient, auth_headers):
    """Should change password with correct current password."""
    response = await client.post(
        "/api/v1/account/change-password",
        json={
            "current_password": "testpassword123",
            "new_password": "newstrongpass456",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert "changed" in response.json()["message"].lower()


@pytest.mark.asyncio
async def test_change_password_wrong_current(client: AsyncClient, auth_headers):
    """Should reject if current password is wrong."""
    response = await client.post(
        "/api/v1/account/change-password",
        json={
            "current_password": "wrongpassword",
            "new_password": "newstrongpass456",
        },
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "incorrect" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_change_password_same(client: AsyncClient, auth_headers):
    """Should reject if new password equals current."""
    response = await client.post(
        "/api/v1/account/change-password",
        json={
            "current_password": "testpassword123",
            "new_password": "testpassword123",
        },
        headers=auth_headers,
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_change_password_too_short(client: AsyncClient, auth_headers):
    """Should reject password shorter than 8 chars."""
    response = await client.post(
        "/api/v1/account/change-password",
        json={
            "current_password": "testpassword123",
            "new_password": "short",
        },
        headers=auth_headers,
    )
    assert response.status_code == 422  # Pydantic validation


@pytest.mark.asyncio
async def test_update_profile_email(client: AsyncClient, auth_headers):
    """Should update email."""
    response = await client.patch(
        "/api/v1/account/me",
        json={"email": "newemail@example.com"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["email"] == "newemail@example.com"
