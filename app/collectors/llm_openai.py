"""OpenAI (ChatGPT) LLM collector."""

import logging
import uuid

import httpx

from app.collectors.llm_base import BaseLlmCollector, LlmResponse

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (as of Feb 2026)
MODEL_PRICING = {
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5": {"input": 1.00, "output": 8.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

DEFAULT_MODEL = "gpt-5-mini"

# Flex processing: 50% discount on token costs, higher latency.
# Set service_tier="flex" in API requests to enable.
FLEX_DISCOUNT = 0.5
API_URL = "https://api.openai.com/v1/chat/completions"

# GPT-5 series are reasoning models that do NOT support temperature,
# top_p, frequency_penalty, presence_penalty, or max_tokens.
# They require max_completion_tokens instead.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    """Check if a model is a reasoning model (GPT-5 / o-series)."""
    return any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES)


class OpenAiCollector(BaseLlmCollector):
    """Collect LLM visibility data from OpenAI ChatGPT API."""

    provider = "chatgpt"

    def __init__(
        self,
        api_key: str,
        tenant_id: uuid.UUID,
        model: str = DEFAULT_MODEL,
        service_tier: str | None = "flex",
    ):
        super().__init__(api_key=api_key, tenant_id=tenant_id)
        self.model = model
        self.service_tier = service_tier

    async def query_llm(self, prompt: str) -> LlmResponse:
        """Send a prompt to OpenAI Chat Completions API."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the following question thoroughly.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        if _is_reasoning_model(self.model):
            # Reasoning models (GPT-5, o-series): no temperature, use max_completion_tokens
            payload["max_completion_tokens"] = 2048
        else:
            # Classic models (GPT-4.1 and older): temperature + max_tokens
            payload["temperature"] = 0.0
            payload["max_tokens"] = 2048

        if self.service_tier:
            payload["service_tier"] = self.service_tier

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Flex processing needs longer timeout (OpenAI recommends >=15 min)
        timeout = 900 if self.service_tier == "flex" else 60
        used_flex = self.service_tier == "flex"

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(API_URL, json=payload, headers=headers)

            # Flex 429 "resource unavailable" — fallback to standard tier.
            # Regular rate-limit 429s will propagate to base-class retry loop.
            if resp.status_code == 429 and self.service_tier == "flex":
                logger.warning(
                    "Flex processing unavailable for model=%s, retrying with standard tier",
                    self.model,
                )
                fallback_payload = {**payload, "service_tier": "auto"}
                resp = await client.post(API_URL, json=fallback_payload, headers=headers)
                used_flex = False

            if resp.status_code >= 400:
                try:
                    error_body = resp.json()
                    error_msg = error_body.get("error", {}).get("message", resp.text[:500])
                except Exception:
                    error_msg = resp.text[:500]
                logger.error(
                    "OpenAI API %d for model=%s: %s",
                    resp.status_code,
                    self.model,
                    error_msg,
                )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        text = choice["message"]["content"] or ""

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = input_tokens + output_tokens

        cost = self._calculate_cost(input_tokens, output_tokens, flex=used_flex)

        return LlmResponse(
            text=text,
            model=data.get("model", self.model),
            tokens=total_tokens,
            cost_usd=cost,
        )

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        flex: bool | None = None,
    ) -> float:
        """Calculate cost in USD based on model pricing.

        *flex* overrides the discount flag — used when Flex was requested but
        the request fell back to standard tier on a 429.
        """
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING[DEFAULT_MODEL])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        apply_flex = flex if flex is not None else (self.service_tier == "flex")
        if apply_flex:
            cost *= FLEX_DISCOUNT
        return round(cost, 6)
