"""Tests for Perplexity collector."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.collectors.llm_perplexity import PerplexityCollector


@pytest.fixture
def collector():
    return PerplexityCollector(
        api_key="pplx-test-fake-key",
        tenant_id=uuid.uuid4(),
        model="sonar",
    )


class TestCalculateCost:
    def test_cost_sonar(self, collector):
        # sonar: input=1.00/1M, output=1.00/1M
        cost = collector._calculate_cost(input_tokens=1000, output_tokens=500)
        expected = (1000 * 1.00 + 500 * 1.00) / 1_000_000
        assert cost == round(expected, 6)

    def test_cost_zero_tokens(self, collector):
        assert collector._calculate_cost(0, 0) == 0.0

    def test_cost_sonar_pro(self):
        c = PerplexityCollector(api_key="test", tenant_id=uuid.uuid4(), model="sonar-pro")
        cost = c._calculate_cost(input_tokens=1000, output_tokens=500)
        expected = (1000 * 3.00 + 500 * 15.00) / 1_000_000
        assert cost == round(expected, 6)


class TestQueryLlm:
    @pytest.mark.asyncio
    async def test_query_llm_success(self, collector):
        mock_response_data = {
            "choices": [
                {
                    "message": {"content": "Ahrefs is an SEO tool [1]."},
                    "finish_reason": "stop",
                }
            ],
            "model": "sonar",
            "citations": [
                "https://ahrefs.com",
                "https://example.com/seo-tools",
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("app.collectors.llm_perplexity.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await collector.query_llm("What is the best SEO tool?")

        assert result.text == "Ahrefs is an SEO tool [1]."
        assert result.model == "sonar"
        assert result.tokens == 150
        assert result.cost_usd > 0
        # Perplexity returns native citations
        assert len(result.cited_urls) == 2
        assert "https://ahrefs.com" in result.cited_urls

    @pytest.mark.asyncio
    async def test_query_llm_no_citations(self, collector):
        """When API returns no citations field, cited_urls should be empty."""
        mock_response_data = {
            "choices": [{"message": {"content": "Some text."}, "finish_reason": "stop"}],
            "model": "sonar",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("app.collectors.llm_perplexity.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await collector.query_llm("test")

        assert result.cited_urls == []


class TestProvider:
    def test_provider_name(self, collector):
        assert collector.provider == "perplexity"

    def test_default_model(self):
        c = PerplexityCollector(api_key="test", tenant_id=uuid.uuid4())
        assert c.model == "sonar"

    def test_api_url(self):
        from app.collectors.llm_perplexity import API_URL

        assert "api.perplexity.ai" in API_URL
