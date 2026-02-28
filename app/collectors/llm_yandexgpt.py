"""YandexGPT LLM collector (Yandex Foundation Models API).

Uses the proprietary REST API format since it provides more control
over model URIs and folder IDs than the OpenAI-compatible wrapper.
"""

import logging
import uuid

import httpx

from app.collectors.llm_base import BaseLlmCollector, LlmResponse

logger = logging.getLogger(__name__)

# Pricing per 1K tokens in USD (approximate, based on RUB prices, 1 USD ~ 90 RUB)
# YandexGPT prices are per-token (input + output combined) in synchronous mode
MODEL_PRICING = {
    "aliceai-llm": {"input": 0.0044, "output": 0.0044},  # ~0.40 RUB/1K tokens (flagship)
    "yandexgpt/latest": {"input": 0.0044, "output": 0.0044},  # ~0.40 RUB/1K tokens
    "yandexgpt/rc": {"input": 0.0044, "output": 0.0044},
    "yandexgpt-lite/latest": {"input": 0.0022, "output": 0.0022},  # ~0.20 RUB/1K tokens
}

# Convert to per-1M-token pricing to match other collectors
MODEL_PRICING_1M = {k: {"input": v["input"] * 1000, "output": v["output"] * 1000} for k, v in MODEL_PRICING.items()}

DEFAULT_MODEL = "aliceai-llm"
API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


class YandexGptCollector(BaseLlmCollector):
    """Collect LLM visibility data from YandexGPT API.

    Requires:
        - api_key: Yandex Cloud API key (Api-Key authentication)
        - folder_id: Yandex Cloud folder ID (used in model URI)
    """

    provider = "yandexgpt"

    def __init__(
        self,
        api_key: str,
        tenant_id: uuid.UUID,
        folder_id: str = "",
        model: str = DEFAULT_MODEL,
    ):
        super().__init__(api_key=api_key, tenant_id=tenant_id)
        self.folder_id = folder_id
        self.model = model

    def _model_uri(self) -> str:
        """Build model URI from folder_id and model name."""
        return f"gpt://{self.folder_id}/{self.model}"

    async def query_llm(self, prompt: str) -> LlmResponse:
        """Send a prompt to YandexGPT Foundation Models API."""
        payload = {
            "modelUri": self._model_uri(),
            "completionOptions": {
                "stream": False,
                "temperature": 0.0,
                "maxTokens": "2048",
            },
            "messages": [
                {
                    "role": "system",
                    "text": "You are a helpful assistant. Answer the following question thoroughly.",
                },
                {"role": "user", "text": prompt},
            ],
        }

        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                API_URL,
                json=payload,
                headers={
                    "Authorization": f"Api-Key {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        result = data.get("result", data)
        alternatives = result.get("alternatives", [])
        text = alternatives[0]["message"]["text"] if alternatives else ""

        usage = result.get("usage", {})
        input_tokens = int(usage.get("inputTextTokens", 0))
        output_tokens = int(usage.get("completionTokens", 0))
        total_tokens = int(usage.get("totalTokens", input_tokens + output_tokens))

        model_version = result.get("modelVersion", self.model)
        cost = self._calculate_cost(input_tokens, output_tokens)

        return LlmResponse(
            text=text,
            model=f"yandexgpt/{model_version}",
            tokens=total_tokens,
            cost_usd=cost,
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD based on model pricing."""
        pricing = MODEL_PRICING_1M.get(self.model, MODEL_PRICING_1M[DEFAULT_MODEL])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        return round(cost, 6)
