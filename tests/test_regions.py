"""Tests for regions API."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_regions(client: AsyncClient):
    """Regions endpoint returns both yandex and google lists."""
    response = await client.get("/api/v1/regions/")
    assert response.status_code == 200
    data = response.json()
    assert "yandex" in data
    assert "google" in data
    assert len(data["yandex"]) > 10
    assert len(data["google"]) > 10

    # Check structure of yandex entry
    yx = data["yandex"][0]
    assert "code" in yx
    assert "name" in yx
    assert "country" in yx
    assert isinstance(yx["code"], int)

    # Check structure of google entry
    gl = data["google"][0]
    assert "code" in gl
    assert "name" in gl
    assert isinstance(gl["code"], str)


@pytest.mark.asyncio
async def test_regions_contains_moscow(client: AsyncClient):
    """Moscow should be present in both region lists."""
    response = await client.get("/api/v1/regions/")
    data = response.json()

    yandex_codes = [r["code"] for r in data["yandex"]]
    assert 213 in yandex_codes  # Moscow Yandex code

    google_names = [r["name"] for r in data["google"]]
    assert any("Москва" in n for n in google_names)
