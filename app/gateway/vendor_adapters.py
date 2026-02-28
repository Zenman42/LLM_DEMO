"""Vendor-Specific Adapters — protocol-level handling for each LLM vendor.

Each adapter translates a GatewayRequest into the vendor's HTTP protocol,
sends it, and returns a GatewayResponse with normalized fields.

Vendor-specific behaviors:
  - ChatGPT: Standard OpenAI chat completions
  - DeepSeek: OpenAI-compatible, 120s timeout, "Server Busy" → slow retry queue
  - Gemini: Google AI generateContent, finish_reason: SAFETY → [CENSORED_BY_VENDOR]
  - YandexGPT: Proprietary async Operation Long Polling, IAM token refresh
  - Perplexity: OpenAI-compatible with native citations
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx

from app.gateway.types import (
    GatewayRequest,
    GatewayResponse,
    GatewayVendor,
    RequestStatus,
)

logger = logging.getLogger(__name__)


class VendorAdapterError(Exception):
    """Raised when a vendor adapter encounters a non-retryable error."""

    def __init__(self, message: str, status_code: int = 0, error_code: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class BaseVendorAdapter(ABC):
    """Base class for all vendor adapters."""

    vendor: GatewayVendor

    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key

    @abstractmethod
    async def send(self, request: GatewayRequest, timeout: float = 60.0) -> GatewayResponse:
        """Send a request to the vendor and return a normalized response."""
        ...

    def _base_response(self, request: GatewayRequest) -> GatewayResponse:
        """Create a base response with context from the request."""
        return GatewayResponse(
            request_id=request.request_id,
            vendor=request.vendor,
            session_id=request.session_id,
            run_index=request.run_index,
            tenant_id=request.tenant_id,
            project_id=request.project_id,
            llm_query_id=request.llm_query_id,
            prompt_id=request.prompt_id,
            started_at=datetime.now(timezone.utc),
        )


# ---------------------------------------------------------------------------
# ChatGPT Adapter (OpenAI)
# ---------------------------------------------------------------------------

# Pricing per 1M tokens
_CHATGPT_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


class ChatGPTAdapter(BaseVendorAdapter):
    """OpenAI Chat Completions adapter."""

    vendor = GatewayVendor.CHATGPT
    default_model = "gpt-4o-mini"
    api_url = "https://api.openai.com/v1/chat/completions"

    async def send(self, request: GatewayRequest, timeout: float = 60.0) -> GatewayResponse:
        response = self._base_response(request)
        model = request.model or self.default_model
        start = time.monotonic()

        payload = {
            "model": model,
            "messages": [],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.system_prompt:
            payload["messages"].append({"role": "system", "content": request.system_prompt})
        payload["messages"].append({"role": "user", "content": request.user_prompt})

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            response.latency_ms = elapsed_ms

            if resp.status_code == 429:
                response.status = RequestStatus.RATE_LIMITED
                response.error_code = "429"
                response.error_message = "Rate limited by OpenAI"
                return response

            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            response.raw_response_text = choice["message"]["content"] or ""
            response.model_version = data.get("model", model)

            usage = data.get("usage", {})
            response.input_tokens = usage.get("prompt_tokens", 0)
            response.output_tokens = usage.get("completion_tokens", 0)
            response.total_tokens = response.input_tokens + response.output_tokens
            response.cost_usd = self._calc_cost(model, response.input_tokens, response.output_tokens)

            response.status = RequestStatus.SUCCESS
            response.completed_at = datetime.now(timezone.utc)
            response.vendor_raw = data

        except httpx.TimeoutException:
            response.status = RequestStatus.TIMEOUT
            response.error_message = f"Timeout after {timeout}s"
            response.latency_ms = int((time.monotonic() - start) * 1000)
        except httpx.HTTPStatusError as e:
            response.status = RequestStatus.VENDOR_ERROR
            response.error_code = str(e.response.status_code)
            response.error_message = str(e)
            response.latency_ms = int((time.monotonic() - start) * 1000)

        return response

    @staticmethod
    def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = _CHATGPT_PRICING.get(model, _CHATGPT_PRICING["gpt-4o-mini"])
        return round((input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000, 6)


# ---------------------------------------------------------------------------
# DeepSeek Adapter
# ---------------------------------------------------------------------------

_DEEPSEEK_PRICING = {
    "deepseek-chat": {"input": 0.28, "output": 0.42},
    "deepseek-reasoner": {"input": 0.28, "output": 0.42},
}


class DeepSeekAdapter(BaseVendorAdapter):
    """DeepSeek adapter with extended timeout and Server Busy handling."""

    vendor = GatewayVendor.DEEPSEEK
    default_model = "deepseek-chat"
    api_url = "https://api.deepseek.com/chat/completions"

    async def send(self, request: GatewayRequest, timeout: float = 120.0) -> GatewayResponse:
        response = self._base_response(request)
        model = request.model or self.default_model
        start = time.monotonic()

        payload = {
            "model": model,
            "messages": [],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.system_prompt:
            payload["messages"].append({"role": "system", "content": request.system_prompt})
        payload["messages"].append({"role": "user", "content": request.user_prompt})

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            response.latency_ms = elapsed_ms

            if resp.status_code == 429:
                response.status = RequestStatus.RATE_LIMITED
                response.error_code = "429"
                response.error_message = "Rate limited by DeepSeek"
                return response

            # DeepSeek "Server Busy" → slow retry queue
            if resp.status_code == 503:
                body = resp.text
                if "busy" in body.lower() or "server" in body.lower():
                    response.status = RequestStatus.RATE_LIMITED  # Treat as rate limit for slow retry
                    response.error_code = "503_BUSY"
                    response.error_message = "DeepSeek server busy — slow retry queue"
                    return response

            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            response.raw_response_text = choice["message"]["content"] or ""
            response.model_version = data.get("model", model)

            usage = data.get("usage", {})
            response.input_tokens = usage.get("prompt_tokens", 0)
            response.output_tokens = usage.get("completion_tokens", 0)
            response.total_tokens = response.input_tokens + response.output_tokens
            response.cost_usd = self._calc_cost(model, response.input_tokens, response.output_tokens)

            response.status = RequestStatus.SUCCESS
            response.completed_at = datetime.now(timezone.utc)
            response.vendor_raw = data

        except httpx.TimeoutException:
            response.status = RequestStatus.TIMEOUT
            response.error_message = f"DeepSeek timeout after {timeout}s"
            response.latency_ms = int((time.monotonic() - start) * 1000)
        except httpx.HTTPStatusError as e:
            response.status = RequestStatus.VENDOR_ERROR
            response.error_code = str(e.response.status_code)
            response.error_message = str(e)
            response.latency_ms = int((time.monotonic() - start) * 1000)

        return response

    @staticmethod
    def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = _DEEPSEEK_PRICING.get(model, _DEEPSEEK_PRICING["deepseek-chat"])
        return round((input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000, 6)


# ---------------------------------------------------------------------------
# Gemini Adapter (Google AI)
# ---------------------------------------------------------------------------

_GEMINI_PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-3.1-pro": {"input": 2.00, "output": 12.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
}


class GeminiAdapter(BaseVendorAdapter):
    """Google Gemini adapter with SAFETY filter detection."""

    vendor = GatewayVendor.GEMINI
    default_model = "gemini-2.0-flash"
    api_url_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def send(self, request: GatewayRequest, timeout: float = 60.0) -> GatewayResponse:
        response = self._base_response(request)
        model = request.model or self.default_model
        start = time.monotonic()

        url = self.api_url_template.format(model=model)

        # Build Gemini request format
        contents = []
        if request.system_prompt:
            # Gemini uses systemInstruction for system prompts
            pass  # Handled below

        contents.append(
            {
                "role": "user",
                "parts": [{"text": request.user_prompt}],
            }
        )

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }

        # System instruction (separate from contents in Gemini API)
        if request.system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": request.system_prompt}],
            }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            response.latency_ms = elapsed_ms

            if resp.status_code == 429:
                response.status = RequestStatus.RATE_LIMITED
                response.error_code = "429"
                response.error_message = "Rate limited by Google AI"
                return response

            resp.raise_for_status()
            data = resp.json()

            # Check for SAFETY block
            candidates = data.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                finish_reason = candidate.get("finishReason", "")

                if finish_reason == "SAFETY":
                    response.status = RequestStatus.CENSORED
                    response.error_code = "SAFETY"
                    response.error_message = "[CENSORED_BY_VENDOR] Gemini safety filter triggered"
                    response.raw_response_text = "[CENSORED_BY_VENDOR]"

                    # Extract safety ratings if available
                    safety_ratings = candidate.get("safetyRatings", [])
                    response.vendor_raw = {"safety_ratings": safety_ratings, "finish_reason": "SAFETY"}
                    response.completed_at = datetime.now(timezone.utc)
                    return response

                # Extract text
                parts = candidate.get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in parts if "text" in p]
                response.raw_response_text = "".join(text_parts)
            else:
                # No candidates — check prompt feedback
                prompt_feedback = data.get("promptFeedback", {})
                block_reason = prompt_feedback.get("blockReason", "")
                if block_reason:
                    response.status = RequestStatus.CENSORED
                    response.error_code = f"BLOCKED_{block_reason}"
                    response.error_message = f"[CENSORED_BY_VENDOR] Prompt blocked: {block_reason}"
                    response.raw_response_text = "[CENSORED_BY_VENDOR]"
                    response.vendor_raw = {"prompt_feedback": prompt_feedback}
                    response.completed_at = datetime.now(timezone.utc)
                    return response

            response.model_version = model

            # Token usage
            usage = data.get("usageMetadata", {})
            response.input_tokens = usage.get("promptTokenCount", 0)
            response.output_tokens = usage.get("candidatesTokenCount", 0)
            response.total_tokens = usage.get("totalTokenCount", 0)
            response.cost_usd = self._calc_cost(model, response.input_tokens, response.output_tokens)

            response.status = RequestStatus.SUCCESS
            response.completed_at = datetime.now(timezone.utc)
            response.vendor_raw = data

        except httpx.TimeoutException:
            response.status = RequestStatus.TIMEOUT
            response.error_message = f"Gemini timeout after {timeout}s"
            response.latency_ms = int((time.monotonic() - start) * 1000)
        except httpx.HTTPStatusError as e:
            response.status = RequestStatus.VENDOR_ERROR
            response.error_code = str(e.response.status_code)
            response.error_message = str(e)
            response.latency_ms = int((time.monotonic() - start) * 1000)

        return response

    @staticmethod
    def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = _GEMINI_PRICING.get(model, _GEMINI_PRICING["gemini-2.0-flash"])
        return round((input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000, 6)


# ---------------------------------------------------------------------------
# YandexGPT Adapter (async Operation Long Polling)
# ---------------------------------------------------------------------------

_YANDEX_PRICING_1M = {
    "yandexgpt-lite/latest": {"input": 2.2, "output": 2.2},
    "yandexgpt/latest": {"input": 4.4, "output": 4.4},
    "yandexgpt/rc": {"input": 4.4, "output": 4.4},
}


class YandexGPTAdapter(BaseVendorAdapter):
    """YandexGPT adapter with async Operation polling and IAM token support.

    Supports two auth modes:
      - API Key: Authorization: Api-Key <key>
      - IAM Token: Authorization: Bearer <iam_token>

    Uses async mode: sends request, gets Operation ID, polls for completion.
    Falls back to sync mode if async endpoint returns result directly.
    """

    vendor = GatewayVendor.YANDEXGPT
    default_model = "yandexgpt-lite/latest"
    sync_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    async_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync"
    operation_url = "https://operation.api.cloud.yandex.net/operations/{operation_id}"

    def __init__(self, api_key: str, folder_id: str = "", use_iam: bool = False, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.folder_id = folder_id
        self.use_iam = use_iam

    def _model_uri(self, model: str) -> str:
        return f"gpt://{self.folder_id}/{model}"

    def _auth_header(self) -> dict[str, str]:
        if self.use_iam:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {"Authorization": f"Api-Key {self.api_key}"}

    async def send(self, request: GatewayRequest, timeout: float = 90.0) -> GatewayResponse:
        response = self._base_response(request)
        model = request.model or self.default_model
        start = time.monotonic()

        payload = {
            "modelUri": self._model_uri(model),
            "completionOptions": {
                "stream": False,
                "temperature": request.temperature,
                "maxTokens": str(request.max_tokens),
            },
            "messages": [],
        }
        if request.system_prompt:
            payload["messages"].append({"role": "system", "text": request.system_prompt})
        payload["messages"].append({"role": "user", "text": request.user_prompt})

        headers = {**self._auth_header(), "Content-Type": "application/json"}

        try:
            # Try async mode first
            data = await self._send_async(payload, headers, timeout)
            if data is None:
                # Fallback to sync mode
                data = await self._send_sync(payload, headers, timeout)

            elapsed_ms = int((time.monotonic() - start) * 1000)
            response.latency_ms = elapsed_ms

            if data is None:
                response.status = RequestStatus.VENDOR_ERROR
                response.error_message = "YandexGPT returned no data"
                return response

            # Parse response
            result = data.get("result", data) if "result" in data else data
            # Handle async operation wrapper
            if "response" in result:
                result = result["response"]

            alternatives = result.get("alternatives", [])
            if alternatives:
                response.raw_response_text = alternatives[0].get("message", {}).get("text", "")
            else:
                response.raw_response_text = ""

            usage = result.get("usage", {})
            response.input_tokens = int(usage.get("inputTextTokens", 0))
            response.output_tokens = int(usage.get("completionTokens", 0))
            response.total_tokens = int(usage.get("totalTokens", response.input_tokens + response.output_tokens))
            response.model_version = f"yandexgpt/{result.get('modelVersion', model)}"
            response.cost_usd = self._calc_cost(model, response.input_tokens, response.output_tokens)

            response.status = RequestStatus.SUCCESS
            response.completed_at = datetime.now(timezone.utc)
            response.vendor_raw = data

        except httpx.TimeoutException:
            response.status = RequestStatus.TIMEOUT
            response.error_message = f"YandexGPT timeout after {timeout}s"
            response.latency_ms = int((time.monotonic() - start) * 1000)
        except httpx.HTTPStatusError as e:
            response.latency_ms = int((time.monotonic() - start) * 1000)
            if e.response.status_code == 429:
                response.status = RequestStatus.RATE_LIMITED
                response.error_code = "429"
                response.error_message = "Rate limited by YandexGPT"
            else:
                response.status = RequestStatus.VENDOR_ERROR
                response.error_code = str(e.response.status_code)
                response.error_message = str(e)

        return response

    async def _send_async(self, payload: dict, headers: dict, timeout: float) -> dict | None:
        """Send via async endpoint, poll for completion."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(self.async_url, json=payload, headers=headers)

                if resp.status_code != 200:
                    return None

                data = resp.json()

                # If we got a direct result (not an operation), return it
                if "alternatives" in data.get("result", data):
                    return data

                # Got an operation — poll for it
                operation_id = data.get("id")
                if not operation_id:
                    return None

                return await self._poll_operation(client, operation_id, headers, timeout)

        except Exception as e:
            logger.debug("YandexGPT async mode failed, falling back to sync: %s", e)
            return None

    async def _poll_operation(
        self,
        client: httpx.AsyncClient,
        operation_id: str,
        headers: dict,
        timeout: float,
    ) -> dict | None:
        """Poll an async operation until done or timeout."""
        url = self.operation_url.format(operation_id=operation_id)
        deadline = time.monotonic() + timeout
        poll_interval = 0.5

        while time.monotonic() < deadline:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return None

            data = resp.json()
            if data.get("done"):
                return data

            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 5.0)

        return None

    async def _send_sync(self, payload: dict, headers: dict, timeout: float) -> dict | None:
        """Send via synchronous endpoint (fallback)."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.sync_url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = _YANDEX_PRICING_1M.get(model, _YANDEX_PRICING_1M["yandexgpt-lite/latest"])
        return round((input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000, 6)


# ---------------------------------------------------------------------------
# Perplexity Adapter
# ---------------------------------------------------------------------------

_PERPLEXITY_PRICING = {
    "sonar": {"input": 1.00, "output": 1.00},
    "sonar-pro": {"input": 3.00, "output": 15.00},
    "sonar-reasoning": {"input": 1.00, "output": 5.00},
    "sonar-reasoning-pro": {"input": 2.00, "output": 8.00},
}


class PerplexityAdapter(BaseVendorAdapter):
    """Perplexity adapter with native citation extraction."""

    vendor = GatewayVendor.PERPLEXITY
    default_model = "sonar"
    api_url = "https://api.perplexity.ai/chat/completions"

    async def send(self, request: GatewayRequest, timeout: float = 90.0) -> GatewayResponse:
        response = self._base_response(request)
        model = request.model or self.default_model
        start = time.monotonic()

        payload = {
            "model": model,
            "messages": [],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.system_prompt:
            payload["messages"].append({"role": "system", "content": request.system_prompt})
        payload["messages"].append({"role": "user", "content": request.user_prompt})

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            response.latency_ms = elapsed_ms

            if resp.status_code == 429:
                response.status = RequestStatus.RATE_LIMITED
                response.error_code = "429"
                response.error_message = "Rate limited by Perplexity"
                return response

            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            response.raw_response_text = choice["message"]["content"] or ""
            response.model_version = data.get("model", model)

            usage = data.get("usage", {})
            response.input_tokens = usage.get("prompt_tokens", 0)
            response.output_tokens = usage.get("completion_tokens", 0)
            response.total_tokens = response.input_tokens + response.output_tokens
            response.cost_usd = self._calc_cost(model, response.input_tokens, response.output_tokens)

            # Perplexity native citations
            response.cited_urls = data.get("citations", [])

            response.status = RequestStatus.SUCCESS
            response.completed_at = datetime.now(timezone.utc)
            response.vendor_raw = data

        except httpx.TimeoutException:
            response.status = RequestStatus.TIMEOUT
            response.error_message = f"Perplexity timeout after {timeout}s"
            response.latency_ms = int((time.monotonic() - start) * 1000)
        except httpx.HTTPStatusError as e:
            response.status = RequestStatus.VENDOR_ERROR
            response.error_code = str(e.response.status_code)
            response.error_message = str(e)
            response.latency_ms = int((time.monotonic() - start) * 1000)

        return response

    @staticmethod
    def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = _PERPLEXITY_PRICING.get(model, _PERPLEXITY_PRICING["sonar"])
        return round((input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000, 6)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: dict[GatewayVendor, type[BaseVendorAdapter]] = {
    GatewayVendor.CHATGPT: ChatGPTAdapter,
    GatewayVendor.DEEPSEEK: DeepSeekAdapter,
    GatewayVendor.GEMINI: GeminiAdapter,
    GatewayVendor.YANDEXGPT: YandexGPTAdapter,
    GatewayVendor.PERPLEXITY: PerplexityAdapter,
}


def get_adapter(vendor: GatewayVendor, api_key: str, **kwargs) -> BaseVendorAdapter:
    """Factory: get the appropriate adapter for a vendor."""
    cls = ADAPTER_REGISTRY.get(vendor)
    if cls is None:
        raise ValueError(f"No adapter registered for vendor: {vendor}")
    return cls(api_key=api_key, **kwargs)
