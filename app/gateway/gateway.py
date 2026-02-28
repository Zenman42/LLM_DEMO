"""LLM API Gateway — orchestrator integrating all gateway components.

Main entry point for dispatching prompts to LLM vendors:
  1. Accepts GatewayRequests (from Module 1 via QueueManager.build_from_prompts)
  2. Routes through Priority Queue
  3. Checks Rate Limits & Circuit Breaker
  4. Dispatches via Vendor Adapters
  5. Normalizes responses
  6. Handles retries with exponential backoff
  7. Routes exhausted requests to Dead Letter Queue

Usage:
    gateway = LlmGateway(api_keys={"chatgpt": "sk-...", "deepseek": "..."})

    # Submit batch
    responses = await gateway.execute_batch(requests)

    # Or process queue
    await gateway.process_queue()
"""

from __future__ import annotations

import asyncio
import logging

from app.gateway.circuit_breaker import CircuitBreaker
from app.gateway.normalizer import normalize_response
from app.gateway.queue_manager import QueueManager
from app.gateway.rate_limiter import AdaptiveRateLimiter
from app.gateway.types import (
    DEFAULT_VENDOR_CONFIGS,
    GatewayRequest,
    GatewayResponse,
    GatewayVendor,
    RequestStatus,
    VendorConfig,
)
from app.gateway.vendor_adapters import BaseVendorAdapter, get_adapter

logger = logging.getLogger(__name__)


class LlmGateway:
    """Main gateway orchestrator.

    Integrates:
      - QueueManager: priority-based request scheduling
      - AdaptiveRateLimiter: per-vendor RPM/TPM enforcement
      - CircuitBreaker: failure detection and recovery
      - VendorAdapters: protocol-specific HTTP calls
      - Normalizer: response post-processing
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        vendor_configs: dict[GatewayVendor, VendorConfig] | None = None,
        adapter_kwargs: dict[str, dict] | None = None,
    ):
        """
        Args:
            api_keys: Mapping of vendor name → API key
            vendor_configs: Override default vendor configurations
            adapter_kwargs: Extra kwargs per vendor (e.g. folder_id for YandexGPT)
        """
        self.api_keys = api_keys or {}
        self.configs = vendor_configs or DEFAULT_VENDOR_CONFIGS

        self.queue = QueueManager()
        self.rate_limiter = AdaptiveRateLimiter(self.configs)
        self.circuit_breaker = CircuitBreaker()

        # Pre-create adapters for configured vendors
        self._adapters: dict[GatewayVendor, BaseVendorAdapter] = {}
        self._adapter_kwargs = adapter_kwargs or {}

    def _get_adapter(self, vendor: GatewayVendor) -> BaseVendorAdapter | None:
        """Get or create adapter for a vendor."""
        if vendor not in self._adapters:
            api_key = self.api_keys.get(vendor.value, "")
            if not api_key:
                return None
            kwargs = self._adapter_kwargs.get(vendor.value, {})
            self._adapters[vendor] = get_adapter(vendor, api_key, **kwargs)
        return self._adapters[vendor]

    async def execute_single(
        self,
        request: GatewayRequest,
    ) -> GatewayResponse:
        """Execute a single request through the full gateway pipeline.

        Handles rate limiting, circuit breaker, retries, and normalization.
        """
        vendor = request.vendor
        config = self.configs.get(vendor, VendorConfig(vendor=vendor))

        # Check circuit breaker
        if not self.circuit_breaker.allow_request(vendor):
            response = GatewayResponse(
                request_id=request.request_id,
                vendor=vendor,
                status=RequestStatus.CIRCUIT_OPEN,
                error_message=f"Circuit breaker open for {vendor.value}",
                session_id=request.session_id,
                run_index=request.run_index,
                tenant_id=request.tenant_id,
                project_id=request.project_id,
                llm_query_id=request.llm_query_id,
                prompt_id=request.prompt_id,
            )
            return response

        # Get adapter
        adapter = self._get_adapter(vendor)
        if adapter is None:
            return GatewayResponse(
                request_id=request.request_id,
                vendor=vendor,
                status=RequestStatus.VENDOR_ERROR,
                error_message=f"No API key configured for {vendor.value}",
                session_id=request.session_id,
                run_index=request.run_index,
                tenant_id=request.tenant_id,
                project_id=request.project_id,
                llm_query_id=request.llm_query_id,
                prompt_id=request.prompt_id,
            )

        # Retry loop
        last_response: GatewayResponse | None = None

        for attempt in range(config.max_retries + 1):
            # Acquire rate limit
            acquired = await self.rate_limiter.acquire_blocking(vendor, timeout=config.timeout_seconds)
            if not acquired:
                last_response = GatewayResponse(
                    request_id=request.request_id,
                    vendor=vendor,
                    status=RequestStatus.RATE_LIMITED,
                    error_message="Rate limit acquisition timeout",
                    retry_count=attempt,
                    session_id=request.session_id,
                    run_index=request.run_index,
                    tenant_id=request.tenant_id,
                    project_id=request.project_id,
                    llm_query_id=request.llm_query_id,
                    prompt_id=request.prompt_id,
                )
                break

            try:
                # Send request via vendor adapter
                response = await adapter.send(request, timeout=config.timeout_seconds)
                response.retry_count = attempt

                # Release rate limiter slot
                self.rate_limiter.release(vendor, total_tokens=response.total_tokens)

                # Check result
                if response.status == RequestStatus.SUCCESS:
                    self.circuit_breaker.record_success(vendor)
                    return normalize_response(response)

                if response.status == RequestStatus.CENSORED:
                    # Censored responses are not retried — it's the answer
                    self.circuit_breaker.record_success(vendor)
                    return normalize_response(response)

                # Retryable failure
                last_response = response
                retry_delay = self.circuit_breaker.record_failure(vendor, attempt, config)

                if retry_delay is None:
                    # Max retries exceeded → dead letter
                    response.status = RequestStatus.DEAD_LETTER
                    self.circuit_breaker.add_to_dead_letter(request, response, attempt + 1)
                    return normalize_response(response)

                # Wait before retry (with DeepSeek "Server Busy" getting extra delay)
                if response.error_code == "503_BUSY":
                    retry_delay = max(retry_delay, 5.0)  # Minimum 5s for busy servers

                logger.info(
                    "Retrying %s request %s (attempt %d/%d) in %.1fs",
                    vendor.value,
                    request.request_id,
                    attempt + 1,
                    config.max_retries,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)

            except Exception as e:
                self.rate_limiter.release(vendor)
                last_response = GatewayResponse(
                    request_id=request.request_id,
                    vendor=vendor,
                    status=RequestStatus.VENDOR_ERROR,
                    error_message=str(e),
                    retry_count=attempt,
                    session_id=request.session_id,
                    run_index=request.run_index,
                    tenant_id=request.tenant_id,
                    project_id=request.project_id,
                    llm_query_id=request.llm_query_id,
                    prompt_id=request.prompt_id,
                )

                retry_delay = self.circuit_breaker.record_failure(vendor, attempt, config)
                if retry_delay is None:
                    last_response.status = RequestStatus.DEAD_LETTER
                    self.circuit_breaker.add_to_dead_letter(request, last_response, attempt + 1)
                    return normalize_response(last_response)

                await asyncio.sleep(retry_delay)

        # If we exhausted retries without a success
        if last_response:
            last_response.status = RequestStatus.DEAD_LETTER
            self.circuit_breaker.add_to_dead_letter(request, last_response, config.max_retries + 1)
            return normalize_response(last_response)

        # Should not reach here
        return GatewayResponse(
            request_id=request.request_id,
            vendor=vendor,
            status=RequestStatus.VENDOR_ERROR,
            error_message="Unexpected gateway error",
        )

    async def execute_batch(
        self,
        requests: list[GatewayRequest],
        max_concurrent_per_vendor: int | None = None,
    ) -> list[GatewayResponse]:
        """Execute a batch of requests with per-vendor concurrency control.

        Requests are grouped by vendor and dispatched concurrently within
        each vendor's concurrency limit.
        """
        if not requests:
            return []

        # Group by vendor
        by_vendor: dict[GatewayVendor, list[GatewayRequest]] = {}
        for req in requests:
            by_vendor.setdefault(req.vendor, []).append(req)

        # Create per-vendor semaphores
        semaphores: dict[GatewayVendor, asyncio.Semaphore] = {}
        for vendor in by_vendor:
            config = self.configs.get(vendor, VendorConfig(vendor=vendor))
            limit = max_concurrent_per_vendor or config.max_concurrent
            semaphores[vendor] = asyncio.Semaphore(limit)

        responses: list[GatewayResponse] = []
        response_lock = asyncio.Lock()

        async def _execute_with_semaphore(req: GatewayRequest) -> None:
            sem = semaphores[req.vendor]
            async with sem:
                resp = await self.execute_single(req)
                async with response_lock:
                    responses.append(resp)

        # Launch all requests concurrently (semaphores handle per-vendor limiting)
        tasks = [_execute_with_semaphore(req) for req in requests]
        await asyncio.gather(*tasks, return_exceptions=True)

        return responses

    async def process_queue(
        self,
        max_requests: int = 100,
    ) -> list[GatewayResponse]:
        """Process pending requests from the priority queue.

        Dequeues up to max_requests across all vendors and executes them.
        """
        requests: list[GatewayRequest] = []

        for vendor in GatewayVendor:
            while len(requests) < max_requests:
                req = await self.queue.dequeue(vendor)
                if req is None:
                    break
                requests.append(req)

        if not requests:
            return []

        logger.info("Processing %d queued requests", len(requests))
        return await self.execute_batch(requests)

    def get_status(self) -> dict:
        """Get comprehensive gateway status."""
        return {
            "queue": self.queue.get_stats(),
            "rate_limits": self.rate_limiter.get_all_stats(),
            "circuits": self.circuit_breaker.get_all_states(),
            "dead_letters": len(self.circuit_breaker.get_dead_letters()),
            "configured_vendors": list(self.api_keys.keys()),
        }
