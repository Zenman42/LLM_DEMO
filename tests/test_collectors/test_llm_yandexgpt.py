"""Tests for YandexGPT collector."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.collectors.llm_yandexgpt import YandexGptCollector


@pytest.fixture
def collector():
    return YandexGptCollector(
        api_key="AQV-test-fake-key",
        tenant_id=uuid.uuid4(),
        folder_id="b1g-test-folder",
        model="aliceai-llm",
    )


class TestModelUri:
    def test_model_uri(self, collector):
        uri = collector._model_uri()
        assert uri == "gpt://b1g-test-folder/aliceai-llm"

    def test_model_uri_pro(self):
        c = YandexGptCollector(
            api_key="test",
            tenant_id=uuid.uuid4(),
            folder_id="my-folder",
            model="yandexgpt/latest",
        )
        assert c._model_uri() == "gpt://my-folder/yandexgpt/latest"


class TestCalculateCost:
    def test_cost_lite(self, collector):
        # yandexgpt-lite: ~2.2 USD per 1M tokens (input and output same)
        cost = collector._calculate_cost(input_tokens=1000, output_tokens=500)
        assert cost > 0
        assert isinstance(cost, float)

    def test_cost_zero_tokens(self, collector):
        assert collector._calculate_cost(0, 0) == 0.0

    def test_cost_pro(self):
        c = YandexGptCollector(
            api_key="test",
            tenant_id=uuid.uuid4(),
            folder_id="f",
            model="yandexgpt/latest",
        )
        cost_pro = c._calculate_cost(input_tokens=1000, output_tokens=500)
        lite = YandexGptCollector(
            api_key="test",
            tenant_id=uuid.uuid4(),
            folder_id="f",
            model="yandexgpt-lite/latest",
        )
        cost_lite = lite._calculate_cost(input_tokens=1000, output_tokens=500)
        # Pro should be more expensive than Lite
        assert cost_pro > cost_lite


class TestQueryLlm:
    @pytest.mark.asyncio
    async def test_query_llm_success(self, collector):
        mock_response_data = {
            "result": {
                "alternatives": [
                    {
                        "message": {
                            "role": "assistant",
                            "text": "Ahrefs is a popular SEO tool.",
                        },
                        "status": "ALTERNATIVE_STATUS_FINAL",
                    }
                ],
                "usage": {
                    "inputTextTokens": "50",
                    "completionTokens": "100",
                    "totalTokens": "150",
                },
                "modelVersion": "23.10.2024",
            }
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("app.collectors.llm_yandexgpt.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await collector.query_llm("What is the best SEO tool?")

        assert result.text == "Ahrefs is a popular SEO tool."
        assert "yandexgpt" in result.model
        assert result.tokens == 150
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_query_llm_uses_api_key_header(self, collector):
        """Verify that Api-Key auth header is used (not Bearer)."""
        mock_response_data = {
            "result": {
                "alternatives": [{"message": {"role": "assistant", "text": "ok"}}],
                "usage": {"inputTextTokens": "10", "completionTokens": "5", "totalTokens": "15"},
                "modelVersion": "1.0",
            }
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_response_data
        mock_resp.raise_for_status = MagicMock()

        with patch("app.collectors.llm_yandexgpt.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            await collector.query_llm("test")

            # Check that post was called with Api-Key auth
            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers", {}) if call_kwargs.kwargs else call_kwargs[1].get("headers", {})
            assert headers["Authorization"].startswith("Api-Key ")


class TestProvider:
    def test_provider_name(self, collector):
        assert collector.provider == "yandexgpt"

    def test_default_model(self):
        c = YandexGptCollector(api_key="test", tenant_id=uuid.uuid4())
        assert c.model == "aliceai-llm"

    def test_api_url(self):
        from app.collectors.llm_yandexgpt import API_URL

        assert "llm.api.cloud.yandex.net" in API_URL
