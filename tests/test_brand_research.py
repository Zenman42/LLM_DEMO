"""Tests for the brand research service."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.services.brand_research import (
    _parse_json_list,
    research_brand,
    suggest_categories,
    suggest_competitors,
)

# Reusable mock request for httpx.Response (required for raise_for_status())
_MOCK_REQUEST_OPENAI = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
_MOCK_REQUEST_PPX = httpx.Request("POST", "https://api.perplexity.ai/chat/completions")


def _mock_httpx_client(side_effect=None, return_value=None):
    """Create a mock httpx.AsyncClient context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if side_effect:
        mock_client.post = AsyncMock(side_effect=side_effect)
    elif return_value:
        mock_client.post = AsyncMock(return_value=return_value)
    return mock_client


# ---------------------------------------------------------------------------
# _parse_json_list tests
# ---------------------------------------------------------------------------


def test_parse_json_list_clean_json():
    """_parse_json_list parses clean JSON object with key."""
    text = '{"competitors": ["Ruukki", "Grand Line", "Lindab"]}'
    result = _parse_json_list(text, "competitors")
    assert result == ["Ruukki", "Grand Line", "Lindab"]


def test_parse_json_list_with_markdown_fences():
    """_parse_json_list handles markdown code fences around JSON."""
    text = '```json\n{"competitors": ["A", "B", "C"]}\n```'
    result = _parse_json_list(text, "competitors")
    assert result == ["A", "B", "C"]


def test_parse_json_list_plain_array():
    """_parse_json_list handles plain JSON array without key."""
    text = '["Alpha", "Beta", "Gamma"]'
    result = _parse_json_list(text, "anything")
    assert result == ["Alpha", "Beta", "Gamma"]


def test_parse_json_list_embedded_json():
    """_parse_json_list finds JSON object embedded in text."""
    text = 'Here are the results:\n{"categories": ["Cat1", "Cat2"]}\nHope this helps!'
    result = _parse_json_list(text, "categories")
    assert result == ["Cat1", "Cat2"]


def test_parse_json_list_fallback_to_lines():
    """_parse_json_list falls back to line-based parsing when no valid JSON."""
    text = "- Ruukki\n- Grand Line\n- Lindab"
    result = _parse_json_list(text, "competitors")
    assert "Ruukki" in result
    assert "Grand Line" in result
    assert "Lindab" in result


def test_parse_json_list_filters_empty_strings():
    """_parse_json_list filters out empty strings."""
    text = '{"items": ["A", "", "B", "  ", "C"]}'
    result = _parse_json_list(text, "items")
    assert result == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# research_brand — OpenAI fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_brand_returns_description():
    """research_brand returns a description from OpenAI when no Perplexity key."""
    mock_response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "Металлпрофиль — крупный российский производитель кровельных и фасадных систем из тонколистовой стали."
                    }
                }
            ]
        },
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await research_brand(
            brand_name="Металлпрофиль",
            api_key="test-key",
            domain="metallprofil.ru",
            market="ru",
        )

    assert "Металлпрофиль" in result
    assert "кровельных" in result
    assert len(result) > 20


@pytest.mark.asyncio
async def test_research_brand_uses_gpt4o_by_default():
    """research_brand uses gpt-4o model by default (not gpt-4o-mini)."""
    captured_payload = {}

    async def capture_post(url, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Test description."}}]},
            request=_MOCK_REQUEST_OPENAI,
        )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(side_effect=capture_post)

        await research_brand(
            brand_name="TestBrand",
            api_key="test-key",
            market="ru",
        )

    assert captured_payload.get("model") == "gpt-4o"


@pytest.mark.asyncio
async def test_research_brand_en_market():
    """research_brand works with English market."""
    mock_response = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": "MetallProfil is a major Russian manufacturer of roofing and facade systems."}}
            ]
        },
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await research_brand(
            brand_name="MetallProfil",
            api_key="test-key",
            market="en",
        )

    assert "MetallProfil" in result
    assert "roofing" in result


@pytest.mark.asyncio
async def test_research_brand_empty_response_raises():
    """research_brand raises ValueError on empty response."""
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": ""}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        with pytest.raises(ValueError, match="Empty response"):
            await research_brand(
                brand_name="Unknown",
                api_key="test-key",
                market="ru",
            )


@pytest.mark.asyncio
async def test_research_brand_truncates_long_response():
    """research_brand truncates responses over 15000 chars."""
    long_text = "A" * 20000

    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": long_text}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await research_brand(
            brand_name="Test",
            api_key="test-key",
            market="en",
        )

    assert len(result) <= 15000


@pytest.mark.asyncio
async def test_research_brand_no_truncation_under_limit():
    """research_brand does not truncate responses under 10000 chars."""
    text = "B" * 5000

    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await research_brand(
            brand_name="Test",
            api_key="test-key",
            market="en",
        )

    assert len(result) == 5000
    assert not result.endswith("...")


@pytest.mark.asyncio
async def test_research_brand_includes_domain_hint():
    """research_brand includes domain in the prompt when provided."""
    captured_payload = {}

    async def capture_post(url, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Test description for company."}}]},
            request=_MOCK_REQUEST_OPENAI,
        )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(side_effect=capture_post)

        await research_brand(
            brand_name="TestBrand",
            api_key="test-key",
            domain="testbrand.com",
            market="ru",
        )

    # Check that the domain was mentioned in the prompt
    messages = captured_payload.get("messages", [])
    user_content = messages[0]["content"] if messages else ""
    assert "testbrand.com" in user_content


# ---------------------------------------------------------------------------
# research_brand — Perplexity priority tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_brand_prefers_perplexity():
    """research_brand uses Perplexity when key is provided."""
    ppx_response = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": "Fonbet — one of the largest Russian bookmakers, operating since 1994."}}
            ]
        },
        request=_MOCK_REQUEST_PPX,
    )

    captured_urls = []

    async def capture_post(url, **kwargs):
        captured_urls.append(url)
        return ppx_response

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(side_effect=capture_post)

        result = await research_brand(
            brand_name="Fonbet",
            api_key="openai-key",
            market="en",
            perplexity_api_key="ppx-key",
        )

    assert "Fonbet" in result
    # Should call Perplexity, not OpenAI
    assert len(captured_urls) == 1
    assert "perplexity" in captured_urls[0]


@pytest.mark.asyncio
async def test_research_brand_falls_back_to_openai_on_perplexity_error():
    """research_brand falls back to OpenAI if Perplexity fails."""
    call_count = {"n": 0}

    openai_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Fonbet is a Russian sports betting company."}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    async def route_post(url, **kwargs):
        call_count["n"] += 1
        if "perplexity" in url:
            raise httpx.HTTPStatusError(
                "Rate limited",
                request=_MOCK_REQUEST_PPX,
                response=httpx.Response(429, request=_MOCK_REQUEST_PPX),
            )
        return openai_response

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(side_effect=route_post)

        result = await research_brand(
            brand_name="Fonbet",
            api_key="openai-key",
            market="en",
            perplexity_api_key="ppx-key",
        )

    assert "Fonbet" in result
    # Should have called both: first Perplexity (failed), then OpenAI
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# suggest_competitors tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suggest_competitors_returns_list():
    """suggest_competitors returns a list of competitor names from JSON."""
    mock_response = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": '{"competitors": ["Ruukki", "Grand Line", "Lindab", "Knauf", "Технониколь"]}'}}
            ]
        },
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_competitors(
            brand_name="Металлпрофиль",
            brand_description="Крупный производитель кровли.",
            api_key="test-key",
            market="ru",
        )

    assert isinstance(result, list)
    assert len(result) == 5
    assert "Ruukki" in result
    assert "Grand Line" in result


@pytest.mark.asyncio
async def test_suggest_competitors_excludes_existing():
    """suggest_competitors excludes competitors already in the list."""
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"competitors": ["Ruukki", "Grand Line", "Lindab"]}'}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_competitors(
            brand_name="Металлпрофиль",
            brand_description="Крупный производитель кровли.",
            api_key="test-key",
            market="ru",
            existing=["Grand Line"],
        )

    # Grand Line should be filtered out
    assert "Grand Line" not in result
    assert "Ruukki" in result
    assert "Lindab" in result


@pytest.mark.asyncio
async def test_suggest_competitors_excludes_brand_itself():
    """suggest_competitors excludes the brand itself from results."""
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"competitors": ["Металлпрофиль", "Ruukki", "Lindab"]}'}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_competitors(
            brand_name="Металлпрофиль",
            brand_description="Крупный производитель кровли.",
            api_key="test-key",
            market="ru",
        )

    assert "Металлпрофиль" not in result
    assert "Ruukki" in result


@pytest.mark.asyncio
async def test_suggest_competitors_max_10():
    """suggest_competitors returns at most 10 results."""
    many_competitors = [f"Competitor_{i}" for i in range(20)]
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps({"competitors": many_competitors})}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_competitors(
            brand_name="Test",
            brand_description="Test company.",
            api_key="test-key",
            market="en",
        )

    assert len(result) <= 10


@pytest.mark.asyncio
async def test_suggest_competitors_empty_response():
    """suggest_competitors returns empty list on empty LLM response."""
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": ""}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_competitors(
            brand_name="Test",
            brand_description="Test.",
            api_key="test-key",
            market="en",
        )

    assert result == []


@pytest.mark.asyncio
async def test_suggest_competitors_prefers_perplexity():
    """suggest_competitors uses Perplexity when key is available."""
    captured_urls = []

    ppx_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"competitors": ["A", "B"]}'}}]},
        request=_MOCK_REQUEST_PPX,
    )

    async def capture_post(url, **kwargs):
        captured_urls.append(url)
        return ppx_response

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(side_effect=capture_post)

        result = await suggest_competitors(
            brand_name="Test",
            brand_description="Test.",
            api_key="openai-key",
            market="en",
            perplexity_api_key="ppx-key",
        )

    assert len(captured_urls) == 1
    assert "perplexity" in captured_urls[0]
    assert result == ["A", "B"]


# ---------------------------------------------------------------------------
# suggest_categories tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suggest_categories_returns_list():
    """suggest_categories returns a list of category strings from JSON."""
    mock_response = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": '{"categories": ["Строительство дома", "Замена кровли", "Утепление фасада"]}'}}
            ]
        },
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_categories(
            brand_name="Металлпрофиль",
            brand_description="Крупный производитель кровли.",
            api_key="test-key",
            market="ru",
        )

    assert isinstance(result, list)
    assert len(result) == 3
    assert "Строительство дома" in result
    assert "Замена кровли" in result


@pytest.mark.asyncio
async def test_suggest_categories_excludes_existing():
    """suggest_categories excludes categories already added."""
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"categories": ["Cat A", "Cat B", "Cat C"]}'}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_categories(
            brand_name="Test",
            brand_description="Test company.",
            api_key="test-key",
            market="en",
            existing=["Cat B"],
        )

    assert "Cat B" not in result
    assert "Cat A" in result
    assert "Cat C" in result


@pytest.mark.asyncio
async def test_suggest_categories_max_15():
    """suggest_categories returns at most 15 results."""
    many_cats = [f"Category_{i}" for i in range(25)]
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps({"categories": many_cats})}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_categories(
            brand_name="Test",
            brand_description="Test.",
            api_key="test-key",
            market="en",
        )

    assert len(result) <= 15


@pytest.mark.asyncio
async def test_suggest_categories_handles_markdown_fences():
    """suggest_categories handles LLM responses with markdown code fences."""
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '```json\n{"categories": ["Home building", "Roof repair"]}\n```'}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_categories(
            brand_name="Test",
            brand_description="Test.",
            api_key="test-key",
            market="en",
        )

    assert result == ["Home building", "Roof repair"]


@pytest.mark.asyncio
async def test_suggest_categories_empty_response():
    """suggest_categories returns empty list on empty response."""
    mock_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": ""}}]},
        request=_MOCK_REQUEST_OPENAI,
    )

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(return_value=mock_response)

        result = await suggest_categories(
            brand_name="Test",
            brand_description="Test.",
            api_key="test-key",
            market="en",
        )

    assert result == []


@pytest.mark.asyncio
async def test_suggest_categories_prefers_perplexity():
    """suggest_categories uses Perplexity when key is available."""
    captured_urls = []

    ppx_response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": '{"categories": ["Cat X", "Cat Y"]}'}}]},
        request=_MOCK_REQUEST_PPX,
    )

    async def capture_post(url, **kwargs):
        captured_urls.append(url)
        return ppx_response

    with patch("app.services.brand_research.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client(side_effect=capture_post)

        result = await suggest_categories(
            brand_name="Test",
            brand_description="Test.",
            api_key="openai-key",
            market="en",
            perplexity_api_key="ppx-key",
        )

    assert len(captured_urls) == 1
    assert "perplexity" in captured_urls[0]
    assert result == ["Cat X", "Cat Y"]
