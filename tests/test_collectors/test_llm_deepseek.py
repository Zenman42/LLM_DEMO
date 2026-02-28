"""Tests for DeepSeek collector."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.collectors.llm_deepseek import DeepSeekCollector


@pytest.fixture
def collector():
    return DeepSeekCollector(
        api_key="sk-test-fake-key",
        tenant_id=uuid.uuid4(),
        model="deepseek-chat",
    )


class TestCalculateCost:
    def test_cost_deepseek_chat(self, collector):
        # deepseek-chat: input=0.28/1M, output=0.42/1M (cache miss)
        cost = collector._calculate_cost(input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.28 + 500 * 0.42) / 1_000_000
        assert cost == round(expected, 6)

    def test_cost_zero_tokens(self, collector):
        assert collector._calculate_cost(0, 0) == 0.0

    def test_cost_large_request(self, collector):
        cost = collector._calculate_cost(input_tokens=100_000, output_tokens=50_000)
        assert cost > 0
        assert isinstance(cost, float)


class TestQueryLlm:
    @pytest.mark.asyncio
    async def test_query_llm_success(self, collector):
        mock_response_data = {
            "choices": [
                {
                    "message": {"content": "Ahrefs is a popular SEO tool."},
                    "finish_reason": "stop",
                }
            ],
            "model": "deepseek-chat",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("app.collectors.llm_deepseek.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await collector.query_llm("What is the best SEO tool?")

        assert result.text == "Ahrefs is a popular SEO tool."
        assert result.model == "deepseek-chat"
        assert result.tokens == 150
        assert result.cost_usd > 0


class TestProvider:
    def test_provider_name(self, collector):
        assert collector.provider == "deepseek"

    def test_default_model(self):
        c = DeepSeekCollector(api_key="test", tenant_id=uuid.uuid4())
        assert c.model == "deepseek-chat"

    def test_api_url(self):
        from app.collectors.llm_deepseek import API_URL

        assert "api.deepseek.com" in API_URL
