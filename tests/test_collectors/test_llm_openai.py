"""Tests for OpenAI (ChatGPT) collector."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.collectors.llm_openai import OpenAiCollector


@pytest.fixture
def collector():
    return OpenAiCollector(
        api_key="sk-test-fake-key",
        tenant_id=uuid.uuid4(),
        model="gpt-5-mini",
    )


class TestCalculateCost:
    def test_cost_gpt5_mini_flex(self, collector):
        # gpt-5-mini: input=0.25/1M, output=2.00/1M, Flex 50% discount
        cost = collector._calculate_cost(input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.25 + 500 * 2.00) / 1_000_000 * 0.5
        assert cost == round(expected, 6)

    def test_cost_zero_tokens(self, collector):
        assert collector._calculate_cost(0, 0) == 0.0

    def test_cost_large_request(self, collector):
        cost = collector._calculate_cost(input_tokens=100_000, output_tokens=50_000)
        assert cost > 0
        assert isinstance(cost, float)

    def test_cost_without_flex(self):
        c = OpenAiCollector(
            api_key="test",
            tenant_id=uuid.uuid4(),
            model="gpt-5-mini",
            service_tier=None,
        )
        cost = c._calculate_cost(input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.25 + 500 * 2.00) / 1_000_000
        assert cost == round(expected, 6)

    def test_cost_flex_override(self, collector):
        """flex=False overrides instance service_tier (used on Flex 429 fallback)."""
        cost_flex = collector._calculate_cost(input_tokens=1000, output_tokens=500, flex=True)
        cost_std = collector._calculate_cost(input_tokens=1000, output_tokens=500, flex=False)
        assert cost_std == cost_flex * 2


class TestQueryLlm:
    @pytest.mark.asyncio
    async def test_query_llm_success(self, collector):
        mock_response_data = {
            "choices": [
                {
                    "message": {"content": "Ahrefs is a great SEO tool."},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-5-mini",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("app.collectors.llm_openai.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await collector.query_llm("What is the best SEO tool?")

        assert result.text == "Ahrefs is a great SEO tool."
        assert result.model == "gpt-5-mini"
        assert result.tokens == 150
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_query_llm_sends_flex_service_tier(self, collector):
        mock_response_data = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-5-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("app.collectors.llm_openai.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            await collector.query_llm("test")

            call_kwargs = mock_client.post.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
            assert payload["service_tier"] == "flex"
            assert payload["model"] == "gpt-5-mini"


class TestFlexFallback:
    @pytest.mark.asyncio
    async def test_flex_429_falls_back_to_standard(self, collector):
        """On Flex 429, retry with service_tier=auto and charge standard price."""
        ok_data = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-5-mini",
            "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
        }

        # First call: 429 (Flex unavailable), second call: 200 (standard OK)
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.text = '{"error":{"message":"Resource unavailable"}}'
        resp_429.json.return_value = {"error": {"message": "Resource unavailable"}}

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.json.return_value = ok_data
        resp_ok.raise_for_status = MagicMock()

        with patch("app.collectors.llm_openai.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [resp_429, resp_ok]
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await collector.query_llm("test")

        assert result.text == "ok"
        # Should have been called twice: flex, then auto
        assert mock_client.post.call_count == 2
        second_payload = mock_client.post.call_args_list[1].kwargs.get(
            "json",
            mock_client.post.call_args_list[1][1].get("json"),
        )
        assert second_payload["service_tier"] == "auto"
        # Cost should be standard (no Flex discount)
        expected_std = (100 * 0.25 + 100 * 2.00) / 1_000_000
        assert result.cost_usd == round(expected_std, 6)

    @pytest.mark.asyncio
    async def test_flex_429_then_standard_429_raises(self, collector):
        """If both Flex and standard return 429, error propagates to base retry."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.text = '{"error":{"message":"Rate limit"}}'
        resp_429.json.return_value = {"error": {"message": "Rate limit"}}
        resp_429.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=resp_429),
        )

        with patch("app.collectors.llm_openai.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = resp_429
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await collector.query_llm("test")

        # Two calls: flex attempt + standard fallback, both 429
        assert mock_client.post.call_count == 2


class TestProvider:
    def test_provider_name(self, collector):
        assert collector.provider == "chatgpt"

    def test_default_model(self):
        c = OpenAiCollector(api_key="test", tenant_id=uuid.uuid4())
        assert c.model == "gpt-5-mini"

    def test_default_service_tier_is_flex(self):
        c = OpenAiCollector(api_key="test", tenant_id=uuid.uuid4())
        assert c.service_tier == "flex"
