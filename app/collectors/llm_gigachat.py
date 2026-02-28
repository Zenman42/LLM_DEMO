"""GigaChat LLM collector (Sber GigaChat REST API).

Uses OAuth 2.0 authentication: a base64-encoded credentials string (Client ID + Secret)
is exchanged for a 30-minute bearer token before making chat completion requests.

API docs: https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api
"""

import logging
import time
import uuid as _uuid

import httpx

from app.collectors.llm_base import BaseLlmCollector, LlmResponse

logger = logging.getLogger(__name__)

# Auth endpoint (OAuth 2.0 token exchange)
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

# Chat completions endpoint
API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

# Pricing per 1M tokens in USD (approximate, based on RUB tariffs, 1 USD ~ 90 RUB)
MODEL_PRICING = {
    "GigaChat": {"input": 0.50, "output": 0.50},
    "GigaChat:latest": {"input": 0.50, "output": 0.50},
    "GigaChat-Pro": {"input": 2.00, "output": 2.00},
    "GigaChat-Max": {"input": 4.00, "output": 4.00},
    "GigaChat-2": {"input": 0.50, "output": 0.50},
    "GigaChat-2-Pro": {"input": 2.00, "output": 2.00},
    "GigaChat-2-Max": {"input": 4.00, "output": 4.00},
}

DEFAULT_MODEL = "GigaChat-2"


class GigaChatCollector(BaseLlmCollector):
    """Collect LLM visibility data from Sber GigaChat API.

    Requires:
        - api_key: Base64-encoded authorization credentials string
          (from Sber developer portal: base64(client_id:client_secret))
        - scope: API scope (GIGACHAT_API_PERS / GIGACHAT_API_B2B / GIGACHAT_API_CORP)
    """

    provider = "gigachat"

    def __init__(
        self,
        api_key: str,
        tenant_id: _uuid.UUID,
        model: str = DEFAULT_MODEL,
        scope: str = "GIGACHAT_API_PERS",
    ):
        super().__init__(api_key=api_key, tenant_id=tenant_id)
        self.model = model
        self.scope = scope
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0

    async def _ensure_token(self, client: httpx.AsyncClient) -> str:
        """Obtain or refresh an OAuth 2.0 bearer token.

        Tokens are valid for 30 minutes. We refresh 60 seconds early to avoid
        edge-case expiry during a request.
        """
        if self._access_token and time.time() < (self._token_expires_at - 60):
            return self._access_token

        rq_uid = str(_uuid.uuid4())
        resp = await client.post(
            AUTH_URL,
            headers={
                "Authorization": f"Basic {self.api_key}",
                "RqUID": rq_uid,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"scope": self.scope},
        )
        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        # expires_at is in milliseconds
        self._token_expires_at = data.get("expires_at", 0) / 1000.0
        if self._token_expires_at == 0:
            # Fallback: 30 min from now
            self._token_expires_at = time.time() + 1800

        return self._access_token

    async def query_llm(self, prompt: str) -> LlmResponse:
        """Send a prompt to GigaChat chat/completions endpoint."""
        # GigaChat uses Russian Trusted Root CA certificates.
        # We disable SSL verification as a practical workaround for the
        # non-standard certificate chain (same approach as official SDK).
        async with httpx.AsyncClient(timeout=90, verify=False) as client:
            token = await self._ensure_token(client)

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
                "stream": False,
            }

            resp = await client.post(
                API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        text = choices[0]["message"]["content"] if choices else ""

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        model_used = data.get("model", self.model)
        cost = self._calculate_cost(input_tokens, output_tokens)

        return LlmResponse(
            text=text,
            model=model_used,
            tokens=total_tokens,
            cost_usd=cost,
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD based on model pricing."""
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING[DEFAULT_MODEL])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        return round(cost, 6)
