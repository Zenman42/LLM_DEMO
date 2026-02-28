"""Tests for web UI routes (HTML template serving)."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_login_page(client: AsyncClient):
    """Login page renders without auth."""
    resp = await client.get("/login")
    assert resp.status_code == 200
    assert "LLM Tracker" in resp.text
    assert "Sign in to continue" in resp.text


@pytest.mark.asyncio
async def test_dashboard_page(client: AsyncClient):
    """Dashboard page renders (auth check happens client-side)."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


@pytest.mark.asyncio
async def test_project_page(client: AsyncClient):
    """Project page renders with project_id."""
    resp = await client.get("/project/1")
    assert resp.status_code == 200
    assert "PROJECT_ID = 1" in resp.text


@pytest.mark.asyncio
async def test_settings_page(client: AsyncClient):
    """Settings page renders."""
    resp = await client.get("/settings")
    assert resp.status_code == 200
    assert "Settings" in resp.text
    assert "Add Project" in resp.text


@pytest.mark.asyncio
async def test_users_page(client: AsyncClient):
    """Users page renders."""
    resp = await client.get("/users")
    assert resp.status_code == 200
    assert "Users" in resp.text
    assert "Invite User" in resp.text


@pytest.mark.asyncio
async def test_api_keys_page(client: AsyncClient):
    """API Keys page renders."""
    resp = await client.get("/api-keys")
    assert resp.status_code == 200
    assert "API Keys" in resp.text
    assert "Create API Key" in resp.text


@pytest.mark.asyncio
async def test_export_page(client: AsyncClient):
    """Export page renders."""
    resp = await client.get("/export")
    assert resp.status_code == 200
    assert "Export" in resp.text


@pytest.mark.asyncio
async def test_account_page(client: AsyncClient):
    """Account page renders."""
    resp = await client.get("/account")
    assert resp.status_code == 200
    assert "Account" in resp.text
    assert "Change Password" in resp.text


@pytest.mark.asyncio
async def test_static_css(client: AsyncClient):
    """Static CSS file is served."""
    resp = await client.get("/static/style.css")
    assert resp.status_code == 200
    assert "var(--accent)" in resp.text


@pytest.mark.asyncio
async def test_static_js(client: AsyncClient):
    """Static JS file is served."""
    resp = await client.get("/static/charts.js")
    assert resp.status_code == 200
    assert "loadChart" in resp.text
