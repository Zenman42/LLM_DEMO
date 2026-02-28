"""Tests for Google Gemini collector."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.collectors.llm_gemini import GeminiCollector


@pytest.fixture
def collector():
    return GeminiCollector(
        api_key="AIzaSyTest-fake-key",
        tenant_id=uuid.uuid4(),
        model="gemini-3-flash-preview",
    )


class TestCalculateCost:
    def test_cost_gemini_3_flash(self, collector):
        # gemini-3-flash-preview: input=0.50/1M, output=3.00/1M
        cost = collector._calculate_cost(input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.50 + 500 * 3.00) / 1_000_000
        assert cost == round(expected, 6)

    def test_cost_zero_tokens(self, collector):
        assert collector._calculate_cost(0, 0) == 0.0

    def test_cost_large_request(self, collector):
        cost = collector._calculate_cost(input_tokens=100_000, output_tokens=50_000)
        assert cost > 0
        assert isinstance(cost, float)

    def test_cost_flash_lite(self):
        c = GeminiCollector(
            api_key="test",
            tenant_id=uuid.uuid4(),
            model="gemini-2.5-flash-lite",
        )
        # gemini-2.5-flash-lite: input=0.02/1M, output=0.10/1M
        cost = c._calculate_cost(input_tokens=10_000, output_tokens=5_000)
        expected = (10_000 * 0.02 + 5_000 * 0.10) / 1_000_000
        assert cost == round(expected, 6)


class TestQueryLlm:
    @pytest.mark.asyncio
    async def test_query_llm_success(self, collector):
        mock_response_data = {
            "choices": [
                {
                    "message": {"content": "Google is a popular search engine."},
                    "finish_reason": "stop",
                }
            ],
            "model": "gemini-3-flash-preview",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("app.collectors.llm_gemini.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await collector.query_llm("What is the best search engine?")

        assert result.text == "Google is a popular search engine."
        assert result.model == "gemini-3-flash-preview"
        assert result.tokens == 150
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_query_llm_sends_correct_headers(self, collector):
        mock_response_data = {
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}],
            "model": "gemini-3-flash-preview",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("app.collectors.llm_gemini.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            await collector.query_llm("test prompt")

            # Verify the API URL and auth header
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            assert call_args[1]["headers"]["Authorization"] == "Bearer AIzaSyTest-fake-key"


class TestProvider:
    def test_provider_name(self, collector):
        assert collector.provider == "gemini"

    def test_default_model(self):
        c = GeminiCollector(api_key="test", tenant_id=uuid.uuid4())
        assert c.model == "gemini-3-flash-preview"

    def test_custom_model(self):
        c = GeminiCollector(
            api_key="test",
            tenant_id=uuid.uuid4(),
            model="gemini-2.5-pro",
        )
        assert c.model == "gemini-2.5-pro"
