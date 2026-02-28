"""Tests for Module 2: LLM API Gateway Layer.

Covers all 5 components:
  - Gateway types and DTOs
  - Adaptive Rate Limiter
  - Priority Queue Manager
  - Circuit Breaker
  - Vendor Adapters (mocked HTTP)
  - Response Normalizer
  - Gateway Orchestrator
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.gateway.circuit_breaker import (
    FAILURE_THRESHOLD,
    CircuitBreaker,
)
from app.gateway.gateway import LlmGateway
from app.gateway.normalizer import (
    aggregate_resilience_responses,
    normalize_response,
)
from app.gateway.queue_manager import QueueManager
from app.gateway.rate_limiter import AdaptiveRateLimiter
from app.gateway.types import (
    DEFAULT_VENDOR_CONFIGS,
    GatewayRequest,
    GatewayResponse,
    GatewayVendor,
    RequestPriority,
    RequestStatus,
    VendorConfig,
)
from app.gateway.vendor_adapters import (
    ADAPTER_REGISTRY,
    ChatGPTAdapter,
    DeepSeekAdapter,
    GeminiAdapter,
    PerplexityAdapter,
    YandexGPTAdapter,
    get_adapter,
)


# ==========================================================================
# Test: Gateway Types
# ==========================================================================


class TestGatewayTypes:
    """Test core types and DTOs."""

    def test_gateway_request_defaults(self):
        req = GatewayRequest()
        assert req.vendor == GatewayVendor.CHATGPT
        assert req.priority == RequestPriority.NORMAL
        assert req.run_index == 0
        assert req.total_runs == 1
        assert len(req.request_id) == 16

    def test_gateway_request_queue_key(self):
        req = GatewayRequest(vendor=GatewayVendor.DEEPSEEK)
        assert req.queue_key.startswith("deepseek:")

    def test_gateway_response_to_dict(self):
        resp = GatewayResponse(
            request_id="abc123",
            vendor=GatewayVendor.CHATGPT,
            model_version="gpt-4o-mini",
            status=RequestStatus.SUCCESS,
            raw_response_text="Hello world",
            latency_ms=150,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
        )
        d = resp.to_dict()
        assert d["request_id"] == "abc123"
        assert d["vendor"] == "chatgpt"
        assert d["status"] == "success"
        assert d["latency_ms"] == 150
        assert d["total_tokens"] == 30

    def test_vendor_config_defaults(self):
        config = VendorConfig(vendor=GatewayVendor.CHATGPT)
        assert config.rpm_limit == 60
        assert config.max_retries == 5
        assert config.base_retry_delay == 1.0

    def test_default_vendor_configs(self):
        assert GatewayVendor.CHATGPT in DEFAULT_VENDOR_CONFIGS
        assert GatewayVendor.DEEPSEEK in DEFAULT_VENDOR_CONFIGS
        assert DEFAULT_VENDOR_CONFIGS[GatewayVendor.DEEPSEEK].timeout_seconds == 120

    def test_request_status_enum(self):
        assert RequestStatus.SUCCESS.value == "success"
        assert RequestStatus.DEAD_LETTER.value == "dead_letter"
        assert RequestStatus.CENSORED.value == "censored"

    def test_request_priority_ordering(self):
        assert RequestPriority.CRITICAL.value < RequestPriority.HIGH.value
        assert RequestPriority.HIGH.value < RequestPriority.NORMAL.value
        assert RequestPriority.NORMAL.value < RequestPriority.LOW.value


# ==========================================================================
# Test: Adaptive Rate Limiter
# ==========================================================================


class TestRateLimiter:
    """Test per-vendor RPM/TPM rate limiting."""

    @pytest.fixture
    def limiter(self):
        configs = {
            GatewayVendor.CHATGPT: VendorConfig(
                vendor=GatewayVendor.CHATGPT,
                rpm_limit=5,
                tpm_limit=1000,
                max_concurrent=10,  # High enough to not interfere with RPM tests
            ),
        }
        return AdaptiveRateLimiter(configs)

    @pytest.mark.asyncio
    async def test_acquire_under_limit(self, limiter):
        wait = await limiter.acquire(GatewayVendor.CHATGPT)
        assert wait == 0.0

    @pytest.mark.asyncio
    async def test_acquire_at_rpm_limit(self, limiter):
        # Fill up to RPM limit, releasing each to avoid concurrent limit
        for _ in range(5):
            wait = await limiter.acquire(GatewayVendor.CHATGPT)
            assert wait == 0.0
            limiter.release(GatewayVendor.CHATGPT, total_tokens=10)

        # Next request should be delayed (RPM=5 hit)
        wait = await limiter.acquire(GatewayVendor.CHATGPT)
        assert wait > 0

    @pytest.mark.asyncio
    async def test_release_decrements_active(self, limiter):
        await limiter.acquire(GatewayVendor.CHATGPT)
        stats_before = limiter.get_stats(GatewayVendor.CHATGPT)
        assert stats_before["active_requests"] == 1

        limiter.release(GatewayVendor.CHATGPT, total_tokens=100)
        stats_after = limiter.get_stats(GatewayVendor.CHATGPT)
        assert stats_after["active_requests"] == 0

    @pytest.mark.asyncio
    async def test_acquire_concurrent_limit(self):
        configs = {
            GatewayVendor.CHATGPT: VendorConfig(
                vendor=GatewayVendor.CHATGPT,
                rpm_limit=100,
                tpm_limit=100_000,
                max_concurrent=3,
            ),
        }
        limiter = AdaptiveRateLimiter(configs)

        # Fill concurrent slots
        for _ in range(3):
            await limiter.acquire(GatewayVendor.CHATGPT)

        # Next should wait (concurrent limit hit)
        wait = await limiter.acquire(GatewayVendor.CHATGPT)
        assert wait > 0

    def test_get_stats(self, limiter):
        stats = limiter.get_stats(GatewayVendor.CHATGPT)
        assert stats["vendor"] == "chatgpt"
        assert stats["rpm_limit"] == 5
        assert stats["current_rpm"] == 0

    def test_get_all_stats(self, limiter):
        stats = limiter.get_all_stats()
        assert len(stats) >= 1

    @pytest.mark.asyncio
    async def test_acquire_blocking_success(self, limiter):
        result = await limiter.acquire_blocking(GatewayVendor.CHATGPT, timeout=5.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_unknown_vendor_creates_default(self, limiter):
        wait = await limiter.acquire(GatewayVendor.PERPLEXITY)
        assert wait == 0.0


# ==========================================================================
# Test: Circuit Breaker
# ==========================================================================


class TestCircuitBreaker:
    """Test circuit breaker with exponential backoff."""

    @pytest.fixture
    def cb(self):
        return CircuitBreaker()

    @pytest.fixture
    def config(self):
        return VendorConfig(
            vendor=GatewayVendor.CHATGPT,
            max_retries=3,
            base_retry_delay=1.0,
            max_retry_delay=30.0,
        )

    def test_initial_state_closed(self, cb):
        assert cb.allow_request(GatewayVendor.CHATGPT) is True
        state = cb.get_circuit_state(GatewayVendor.CHATGPT)
        assert state["state"] == "closed"

    def test_record_success_resets_failures(self, cb, config):
        cb.record_failure(GatewayVendor.CHATGPT, attempt=0, config=config)
        cb.record_failure(GatewayVendor.CHATGPT, attempt=1, config=config)
        cb.record_success(GatewayVendor.CHATGPT)
        state = cb.get_circuit_state(GatewayVendor.CHATGPT)
        assert state["consecutive_failures"] == 0

    def test_circuit_opens_after_threshold(self, cb, config):
        for i in range(FAILURE_THRESHOLD):
            cb.record_failure(GatewayVendor.CHATGPT, attempt=0, config=config)

        assert cb.allow_request(GatewayVendor.CHATGPT) is False
        state = cb.get_circuit_state(GatewayVendor.CHATGPT)
        assert state["state"] == "open"

    def test_record_failure_returns_delay(self, cb, config):
        delay = cb.record_failure(GatewayVendor.CHATGPT, attempt=0, config=config)
        assert delay is not None
        assert delay > 0

    def test_max_retries_returns_none(self, cb, config):
        # attempt >= max_retries → None (dead letter)
        delay = cb.record_failure(GatewayVendor.CHATGPT, attempt=3, config=config)
        assert delay is None

    def test_exponential_backoff(self, cb):
        delay_0 = cb.calculate_backoff(attempt=0, base_delay=1.0)
        delay_1 = cb.calculate_backoff(attempt=1, base_delay=1.0)
        delay_2 = cb.calculate_backoff(attempt=2, base_delay=1.0)
        # Each delay roughly doubles (with jitter)
        assert delay_1 > delay_0
        assert delay_2 > delay_1

    def test_backoff_capped_at_max(self, cb):
        delay = cb.calculate_backoff(attempt=20, base_delay=1.0, max_delay=60.0)
        assert delay <= 60.0

    def test_dead_letter_queue(self, cb):
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT)
        resp = GatewayResponse(request_id=req.request_id, error_message="fail")
        cb.add_to_dead_letter(req, resp, attempts=5)

        letters = cb.get_dead_letters()
        assert len(letters) == 1
        assert letters[0].attempts == 5
        assert letters[0].last_error == "fail"

    def test_clear_dead_letters(self, cb):
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT)
        resp = GatewayResponse(request_id=req.request_id)
        cb.add_to_dead_letter(req, resp, attempts=3)

        count = cb.clear_dead_letters()
        assert count == 1
        assert len(cb.get_dead_letters()) == 0

    def test_reset_circuit(self, cb, config):
        for i in range(FAILURE_THRESHOLD):
            cb.record_failure(GatewayVendor.CHATGPT, attempt=0, config=config)

        assert cb.allow_request(GatewayVendor.CHATGPT) is False
        cb.reset(GatewayVendor.CHATGPT)
        assert cb.allow_request(GatewayVendor.CHATGPT) is True

    def test_get_all_states(self, cb):
        states = cb.get_all_states()
        assert len(states) == len(GatewayVendor)


# ==========================================================================
# Test: Priority Queue Manager
# ==========================================================================


class TestQueueManager:
    """Test priority queue with batching."""

    @pytest.fixture
    def qm(self):
        return QueueManager()

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, qm):
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="Hello")
        await qm.enqueue(req)
        assert qm.queue_size(GatewayVendor.CHATGPT) == 1

        result = await qm.dequeue(GatewayVendor.CHATGPT)
        assert result is not None
        assert result.user_prompt == "Hello"
        assert qm.queue_size(GatewayVendor.CHATGPT) == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self, qm):
        low = GatewayRequest(vendor=GatewayVendor.CHATGPT, priority=RequestPriority.LOW, user_prompt="low")
        high = GatewayRequest(vendor=GatewayVendor.CHATGPT, priority=RequestPriority.HIGH, user_prompt="high")
        critical = GatewayRequest(
            vendor=GatewayVendor.CHATGPT, priority=RequestPriority.CRITICAL, user_prompt="critical"
        )

        # Enqueue in reverse priority order
        await qm.enqueue(low)
        await qm.enqueue(high)
        await qm.enqueue(critical)

        # Should dequeue in priority order
        first = await qm.dequeue(GatewayVendor.CHATGPT)
        assert first.user_prompt == "critical"

        second = await qm.dequeue(GatewayVendor.CHATGPT)
        assert second.user_prompt == "high"

        third = await qm.dequeue(GatewayVendor.CHATGPT)
        assert third.user_prompt == "low"

    @pytest.mark.asyncio
    async def test_enqueue_batch(self, qm):
        requests = [GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt=f"req_{i}") for i in range(5)]
        count = await qm.enqueue_batch(requests)
        assert count == 5
        assert qm.total_size() == 5

    @pytest.mark.asyncio
    async def test_dequeue_empty_returns_none(self, qm):
        result = await qm.dequeue(GatewayVendor.CHATGPT)
        assert result is None

    @pytest.mark.asyncio
    async def test_peek_does_not_remove(self, qm):
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="peek test")
        await qm.enqueue(req)

        peeked = await qm.peek(GatewayVendor.CHATGPT)
        assert peeked is not None
        assert qm.queue_size(GatewayVendor.CHATGPT) == 1

    @pytest.mark.asyncio
    async def test_multi_vendor_queues(self, qm):
        chatgpt_req = GatewayRequest(vendor=GatewayVendor.CHATGPT)
        deepseek_req = GatewayRequest(vendor=GatewayVendor.DEEPSEEK)

        await qm.enqueue(chatgpt_req)
        await qm.enqueue(deepseek_req)

        assert qm.queue_size(GatewayVendor.CHATGPT) == 1
        assert qm.queue_size(GatewayVendor.DEEPSEEK) == 1
        assert qm.total_size() == 2

    def test_get_stats(self, qm):
        stats = qm.get_stats()
        assert "total" in stats
        assert "by_vendor" in stats
        assert stats["total"] == 0

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self, qm):
        first = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="first")
        second = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="second")

        await qm.enqueue(first)
        await qm.enqueue(second)

        result1 = await qm.dequeue(GatewayVendor.CHATGPT)
        result2 = await qm.dequeue(GatewayVendor.CHATGPT)
        assert result1.user_prompt == "first"
        assert result2.user_prompt == "second"


class TestQueueManagerBuildFromPrompts:
    """Test building GatewayRequests from Module 1 FinalPrompt objects."""

    def _make_prompt(self, target_llm="chatgpt", intent="comparative", resilience=1):
        from app.prompt_engine.types import FinalPrompt, Intent, Persona, TargetLLM

        return FinalPrompt(
            prompt_id="test_prompt_001",
            target_llm=TargetLLM(target_llm),
            system_prompt="You are an expert.",
            user_prompt="Compare SEO tools",
            temperature=0.7,
            resilience_runs=resilience,
            max_tokens=2048,
            persona=Persona.CTO,
            intent=Intent(intent),
            category="SEO",
            brand="Ahrefs",
            competitors=["SEMrush", "Moz"],
            market="ru",
        )

    def test_build_single_prompt(self):
        prompts = [self._make_prompt()]
        requests = QueueManager.build_from_prompts(prompts, tenant_id="t1", project_id=1)
        assert len(requests) == 1
        assert requests[0].vendor == GatewayVendor.CHATGPT
        assert requests[0].tenant_id == "t1"
        assert requests[0].project_id == 1

    def test_build_with_resilience_runs(self):
        prompts = [self._make_prompt(resilience=3)]
        requests = QueueManager.build_from_prompts(prompts, tenant_id="t1", project_id=1)
        assert len(requests) == 3
        session_ids = {r.session_id for r in requests}
        # All runs share the same session base
        assert all("_run" in sid for sid in session_ids)
        # Each run has unique session_id
        assert len(session_ids) == 3

    def test_build_comparative_gets_high_priority(self):
        prompts = [self._make_prompt(intent="comparative")]
        requests = QueueManager.build_from_prompts(prompts, tenant_id="t1", project_id=1)
        assert requests[0].priority == RequestPriority.HIGH

    def test_build_informational_gets_normal_priority(self):
        prompts = [self._make_prompt(intent="informational")]
        requests = QueueManager.build_from_prompts(prompts, tenant_id="t1", project_id=1)
        assert requests[0].priority == RequestPriority.NORMAL

    def test_build_multi_vendor(self):
        prompts = [
            self._make_prompt(target_llm="chatgpt"),
            self._make_prompt(target_llm="deepseek"),
            self._make_prompt(target_llm="yandexgpt"),
        ]
        requests = QueueManager.build_from_prompts(prompts, tenant_id="t1", project_id=1)
        vendors = {r.vendor for r in requests}
        assert GatewayVendor.CHATGPT in vendors
        assert GatewayVendor.DEEPSEEK in vendors
        assert GatewayVendor.YANDEXGPT in vendors


# ==========================================================================
# Test: Vendor Adapters (mocked HTTP)
# ==========================================================================


def _make_httpx_response(status_code: int, json_data: dict | None = None, text: str = "") -> httpx.Response:
    """Create a proper httpx.Response with request set (needed for raise_for_status)."""
    request = httpx.Request("POST", "https://example.com")
    if json_data is not None:
        resp = httpx.Response(status_code, json=json_data, request=request)
    else:
        resp = httpx.Response(status_code, text=text, request=request)
    return resp


def _mock_openai_response(text="Hello world", model="gpt-4o-mini", input_tokens=10, output_tokens=20):
    return _make_httpx_response(
        200,
        json_data={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "model": model,
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
        },
    )


def _mock_gemini_response(text="Hello world", finish_reason="STOP"):
    return _make_httpx_response(
        200,
        json_data={
            "candidates": [
                {
                    "content": {"parts": [{"text": text}]},
                    "finishReason": finish_reason,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
            },
        },
    )


def _mock_gemini_safety_response():
    return _make_httpx_response(
        200,
        json_data={
            "candidates": [
                {
                    "finishReason": "SAFETY",
                    "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH"}],
                }
            ],
        },
    )


def _mock_yandex_response(text="Привет мир"):
    return _make_httpx_response(
        200,
        json_data={
            "result": {
                "alternatives": [{"message": {"role": "assistant", "text": text}}],
                "usage": {"inputTextTokens": "10", "completionTokens": "20", "totalTokens": "30"},
                "modelVersion": "07.03.2024",
            }
        },
    )


def _mock_perplexity_response(text="Hello", citations=None):
    return _make_httpx_response(
        200,
        json_data={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "model": "sonar",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "citations": citations or ["https://example.com"],
        },
    )


class TestChatGPTAdapter:
    @pytest.mark.asyncio
    async def test_success(self):
        adapter = ChatGPTAdapter(api_key="test-key")
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="Hello")

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_openai_response()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req)

        assert resp.status == RequestStatus.SUCCESS
        assert resp.raw_response_text == "Hello world"
        assert resp.total_tokens == 30
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_rate_limited(self):
        adapter = ChatGPTAdapter(api_key="test-key")
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="Hello")

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _make_httpx_response(429, text="rate limited")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req)

        assert resp.status == RequestStatus.RATE_LIMITED
        assert resp.error_code == "429"

    @pytest.mark.asyncio
    async def test_timeout(self):
        adapter = ChatGPTAdapter(api_key="test-key")
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="Hello")

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("timeout")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req, timeout=5.0)

        assert resp.status == RequestStatus.TIMEOUT

    def test_cost_calculation(self):
        cost = ChatGPTAdapter._calc_cost("gpt-4o-mini", 1000, 500)
        assert cost > 0


class TestDeepSeekAdapter:
    @pytest.mark.asyncio
    async def test_success(self):
        adapter = DeepSeekAdapter(api_key="test-key")
        req = GatewayRequest(vendor=GatewayVendor.DEEPSEEK, user_prompt="Hello")

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_openai_response(model="deepseek-chat")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req)

        assert resp.status == RequestStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_server_busy_503(self):
        adapter = DeepSeekAdapter(api_key="test-key")
        req = GatewayRequest(vendor=GatewayVendor.DEEPSEEK, user_prompt="Hello")

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_response = _make_httpx_response(503, text="Server is busy, please retry later")
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req)

        assert resp.status == RequestStatus.RATE_LIMITED
        assert resp.error_code == "503_BUSY"


class TestGeminiAdapter:
    @pytest.mark.asyncio
    async def test_success(self):
        adapter = GeminiAdapter(api_key="test-key")
        req = GatewayRequest(vendor=GatewayVendor.GEMINI, user_prompt="Hello")

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_gemini_response()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req)

        assert resp.status == RequestStatus.SUCCESS
        assert resp.raw_response_text == "Hello world"
        assert resp.total_tokens == 30

    @pytest.mark.asyncio
    async def test_safety_filter(self):
        adapter = GeminiAdapter(api_key="test-key")
        req = GatewayRequest(vendor=GatewayVendor.GEMINI, user_prompt="Test")

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_gemini_safety_response()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req)

        assert resp.status == RequestStatus.CENSORED
        assert resp.error_code == "SAFETY"
        assert "[CENSORED_BY_VENDOR]" in resp.raw_response_text

    @pytest.mark.asyncio
    async def test_prompt_blocked(self):
        adapter = GeminiAdapter(api_key="test-key")
        req = GatewayRequest(vendor=GatewayVendor.GEMINI, user_prompt="Test")

        blocked_response = _make_httpx_response(
            200,
            json_data={
                "candidates": [],
                "promptFeedback": {"blockReason": "SAFETY"},
            },
        )

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = blocked_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req)

        assert resp.status == RequestStatus.CENSORED
        assert "BLOCKED_SAFETY" in resp.error_code


class TestYandexGPTAdapter:
    @pytest.mark.asyncio
    async def test_sync_fallback(self):
        adapter = YandexGPTAdapter(api_key="test-key", folder_id="b1g123")
        req = GatewayRequest(vendor=GatewayVendor.YANDEXGPT, user_prompt="Привет")

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()

            # Async mode fails → fallback to sync
            async_fail = _make_httpx_response(500, text="error")
            sync_ok = _mock_yandex_response()

            mock_client.post.side_effect = [async_fail, sync_ok]
            mock_client.get = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req)

        assert resp.status == RequestStatus.SUCCESS
        assert "Привет мир" in resp.raw_response_text

    def test_model_uri(self):
        adapter = YandexGPTAdapter(api_key="test", folder_id="b1gfoo")
        assert adapter._model_uri("yandexgpt/latest") == "gpt://b1gfoo/yandexgpt/latest"

    def test_auth_header_api_key(self):
        adapter = YandexGPTAdapter(api_key="my-key", use_iam=False)
        assert adapter._auth_header() == {"Authorization": "Api-Key my-key"}

    def test_auth_header_iam(self):
        adapter = YandexGPTAdapter(api_key="iam-token", use_iam=True)
        assert adapter._auth_header() == {"Authorization": "Bearer iam-token"}


class TestPerplexityAdapter:
    @pytest.mark.asyncio
    async def test_success_with_citations(self):
        adapter = PerplexityAdapter(api_key="test-key")
        req = GatewayRequest(vendor=GatewayVendor.PERPLEXITY, user_prompt="Hello")

        citations = ["https://example.com/1", "https://example.com/2"]

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_perplexity_response(citations=citations)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await adapter.send(req)

        assert resp.status == RequestStatus.SUCCESS
        assert resp.cited_urls == citations


class TestAdapterRegistry:
    def test_all_vendors_registered(self):
        for vendor in GatewayVendor:
            assert vendor in ADAPTER_REGISTRY

    def test_get_adapter(self):
        adapter = get_adapter(GatewayVendor.CHATGPT, "test-key")
        assert isinstance(adapter, ChatGPTAdapter)

    def test_get_adapter_yandex_with_kwargs(self):
        adapter = get_adapter(GatewayVendor.YANDEXGPT, "test-key", folder_id="b1g123")
        assert isinstance(adapter, YandexGPTAdapter)
        assert adapter.folder_id == "b1g123"

    def test_get_adapter_unknown_vendor_raises(self):
        # This won't happen in practice since GatewayVendor is an enum
        # but test the error message
        with pytest.raises(ValueError, match="No adapter"):
            get_adapter("unknown_vendor", "key")


# ==========================================================================
# Test: Response Normalizer
# ==========================================================================


class TestNormalizer:
    def test_normalize_sets_completed_at(self):
        resp = GatewayResponse(
            status=RequestStatus.SUCCESS,
            completed_at=None,
        )
        normalized = normalize_response(resp)
        assert normalized.completed_at is not None

    def test_normalize_calculates_total_tokens(self):
        resp = GatewayResponse(
            status=RequestStatus.SUCCESS,
            input_tokens=100,
            output_tokens=200,
            total_tokens=0,
        )
        normalized = normalize_response(resp)
        assert normalized.total_tokens == 300

    def test_normalize_extracts_urls(self):
        resp = GatewayResponse(
            status=RequestStatus.SUCCESS,
            raw_response_text="Check https://example.com and https://test.org/page for details.",
        )
        normalized = normalize_response(resp)
        assert "https://example.com" in normalized.cited_urls
        assert "https://test.org/page" in normalized.cited_urls

    def test_normalize_preserves_native_citations(self):
        resp = GatewayResponse(
            status=RequestStatus.SUCCESS,
            cited_urls=["https://native.com"],
            raw_response_text="Some text with https://extracted.com",
        )
        normalized = normalize_response(resp)
        # Native citations preserved, text URLs not added
        assert "https://native.com" in normalized.cited_urls

    def test_normalize_censored_response(self):
        resp = GatewayResponse(
            status=RequestStatus.CENSORED,
            raw_response_text="Some partial text",
        )
        normalized = normalize_response(resp)
        assert normalized.raw_response_text == "[CENSORED_BY_VENDOR]"

    def test_normalize_idempotent(self):
        resp = GatewayResponse(
            status=RequestStatus.SUCCESS,
            raw_response_text="Hello",
            total_tokens=100,
        )
        first = normalize_response(resp)
        second = normalize_response(first)
        assert first.total_tokens == second.total_tokens

    def test_clean_urls_deduplicates(self):
        resp = GatewayResponse(
            status=RequestStatus.SUCCESS,
            raw_response_text="Visit https://example.com and https://example.com again",
        )
        normalized = normalize_response(resp)
        assert len(normalized.cited_urls) == 1

    def test_aggregate_resilience_responses(self):
        responses = [
            GatewayResponse(
                status=RequestStatus.SUCCESS,
                session_id="s1_run0",
                prompt_id="p1",
                latency_ms=100,
                cost_usd=0.001,
            ),
            GatewayResponse(
                status=RequestStatus.SUCCESS,
                session_id="s1_run1",
                prompt_id="p1",
                latency_ms=200,
                cost_usd=0.002,
            ),
            GatewayResponse(
                status=RequestStatus.CENSORED,
                session_id="s1_run2",
                prompt_id="p1",
                latency_ms=50,
                cost_usd=0.0,
            ),
        ]
        summary = aggregate_resilience_responses(responses)
        assert summary["total_runs"] == 3
        assert summary["successful_runs"] == 2
        assert summary["censored_runs"] == 1
        assert summary["total_cost_usd"] == 0.003
        assert summary["avg_latency_ms"] > 0

    def test_aggregate_empty_responses(self):
        summary = aggregate_resilience_responses([])
        assert summary["total_runs"] == 0


# ==========================================================================
# Test: Gateway Orchestrator
# ==========================================================================


class TestLlmGateway:
    """Test the main gateway orchestrator."""

    @pytest.fixture
    def gateway(self):
        return LlmGateway(
            api_keys={"chatgpt": "test-key", "deepseek": "test-key"},
            vendor_configs={
                GatewayVendor.CHATGPT: VendorConfig(
                    vendor=GatewayVendor.CHATGPT,
                    rpm_limit=100,
                    max_concurrent=10,
                    max_retries=2,
                    timeout_seconds=5,
                ),
                GatewayVendor.DEEPSEEK: VendorConfig(
                    vendor=GatewayVendor.DEEPSEEK,
                    rpm_limit=100,
                    max_concurrent=5,
                    max_retries=2,
                    timeout_seconds=5,
                ),
            },
        )

    @pytest.mark.asyncio
    async def test_execute_single_success(self, gateway):
        req = GatewayRequest(
            vendor=GatewayVendor.CHATGPT,
            user_prompt="Hello",
            tenant_id="t1",
            project_id=1,
        )

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_openai_response()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            resp = await gateway.execute_single(req)

        assert resp.status == RequestStatus.SUCCESS
        assert resp.raw_response_text == "Hello world"
        assert resp.tenant_id == "t1"
        assert resp.project_id == 1

    @pytest.mark.asyncio
    async def test_execute_single_no_api_key(self):
        gw = LlmGateway(api_keys={})
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="Hello")
        resp = await gw.execute_single(req)
        assert resp.status == RequestStatus.VENDOR_ERROR
        assert "No API key" in resp.error_message

    @pytest.mark.asyncio
    async def test_execute_single_circuit_open(self, gateway):
        # Trip the circuit breaker
        config = gateway.configs[GatewayVendor.CHATGPT]
        for _ in range(FAILURE_THRESHOLD):
            gateway.circuit_breaker.record_failure(GatewayVendor.CHATGPT, attempt=0, config=config)

        req = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="Hello")
        resp = await gateway.execute_single(req)
        assert resp.status == RequestStatus.CIRCUIT_OPEN

    @pytest.mark.asyncio
    async def test_execute_batch(self, gateway):
        requests = [GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt=f"req_{i}") for i in range(3)]

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_openai_response()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            responses = await gateway.execute_batch(requests)

        assert len(responses) == 3
        assert all(r.status == RequestStatus.SUCCESS for r in responses)

    @pytest.mark.asyncio
    async def test_execute_batch_empty(self, gateway):
        responses = await gateway.execute_batch([])
        assert responses == []

    def test_get_status(self, gateway):
        status = gateway.get_status()
        assert "queue" in status
        assert "rate_limits" in status
        assert "circuits" in status
        assert "dead_letters" in status

    @pytest.mark.asyncio
    async def test_process_queue(self, gateway):
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="Hello")
        await gateway.queue.enqueue(req)

        with patch("app.gateway.vendor_adapters.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_openai_response()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            responses = await gateway.process_queue()

        assert len(responses) == 1
        assert responses[0].status == RequestStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_censored_not_retried(self, gateway):
        """Censored responses should be returned immediately, not retried."""
        req = GatewayRequest(vendor=GatewayVendor.CHATGPT, user_prompt="Hello")

        # Create a mock adapter that returns CENSORED
        mock_adapter = AsyncMock(spec=ChatGPTAdapter)
        mock_adapter.send.return_value = GatewayResponse(
            request_id=req.request_id,
            vendor=GatewayVendor.CHATGPT,
            status=RequestStatus.CENSORED,
            error_code="SAFETY",
            error_message="[CENSORED_BY_VENDOR]",
            raw_response_text="[CENSORED_BY_VENDOR]",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        gateway._adapters[GatewayVendor.CHATGPT] = mock_adapter

        resp = await gateway.execute_single(req)
        assert resp.status == RequestStatus.CENSORED
        # Should only be called once (no retries)
        assert mock_adapter.send.call_count == 1
