"""DeepSeek LLM collector (OpenAI-compatible API)."""

import logging
import uuid

import httpx

from app.collectors.llm_base import BaseLlmCollector, LlmResponse

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (cache miss rates, as of Feb 2026)
# deepseek-chat now serves DeepSeek-V3.2 behind the same model ID
MODEL_PRICING = {
    "deepseek-chat": {"input": 0.28, "output": 0.42},
    "deepseek-reasoner": {"input": 0.28, "output": 0.42},
}

DEFAULT_MODEL = "deepseek-chat"
API_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekCollector(BaseLlmCollector):
    """Collect LLM visibility data from DeepSeek API."""

    provider = "deepseek"

    def __init__(self, api_key: str, tenant_id: uuid.UUID, model: str = DEFAULT_MODEL):
        super().__init__(api_key=api_key, tenant_id=tenant_id)
        self.model = model

    async def query_llm(self, prompt: str) -> LlmResponse:
        """Send a prompt to DeepSeek Chat Completions API."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the following question thoroughly.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        text = choice["message"]["content"] or ""

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = input_tokens + output_tokens

        cost = self._calculate_cost(input_tokens, output_tokens)

        return LlmResponse(
            text=text,
            model=data.get("model", self.model),
            tokens=total_tokens,
            cost_usd=cost,
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD based on model pricing (cache miss rates)."""
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING[DEFAULT_MODEL])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        return round(cost, 6)
