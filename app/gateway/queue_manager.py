"""Priority Queue Manager — batching and scheduling of gateway requests.

Manages per-vendor priority queues with support for:
  - Priority-based ordering (CRITICAL > HIGH > NORMAL > LOW)
  - Session_ID grouping for Resilience Runs
  - Batch submission from Module 1's FinalPrompt output
  - Queue stats and monitoring
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from heapq import heappop, heappush

from app.gateway.types import (
    GatewayRequest,
    GatewayVendor,
    RequestPriority,
)

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _PriorityItem:
    """Wrapper for heap queue ordering."""

    priority: int
    sequence: int  # Tie-breaker for FIFO within same priority
    request: GatewayRequest = field(compare=False)


class QueueManager:
    """Per-vendor priority queue with session grouping.

    Usage:
        qm = QueueManager()

        # Submit requests
        qm.enqueue(request)
        qm.enqueue_batch(requests)

        # Get next request for a vendor
        request = qm.dequeue(vendor)

        # Build batch from Module 1 output
        requests = QueueManager.build_from_prompts(final_prompts, tenant_id, project_id, ...)
    """

    def __init__(self):
        self._queues: dict[GatewayVendor, list[_PriorityItem]] = {v: [] for v in GatewayVendor}
        self._sequence: int = 0
        self._lock = asyncio.Lock()

    async def enqueue(self, request: GatewayRequest) -> None:
        """Add a request to the appropriate vendor queue."""
        async with self._lock:
            self._sequence += 1
            item = _PriorityItem(
                priority=request.priority.value,
                sequence=self._sequence,
                request=request,
            )
            heappush(self._queues[request.vendor], item)

        logger.debug(
            "Enqueued request %s for %s (priority=%s)",
            request.request_id,
            request.vendor.value,
            request.priority.name,
        )

    async def enqueue_batch(self, requests: list[GatewayRequest]) -> int:
        """Add multiple requests to their respective vendor queues.

        Returns the number of requests enqueued.
        """
        async with self._lock:
            for req in requests:
                self._sequence += 1
                item = _PriorityItem(
                    priority=req.priority.value,
                    sequence=self._sequence,
                    request=req,
                )
                heappush(self._queues[req.vendor], item)

        logger.info("Enqueued batch of %d requests", len(requests))
        return len(requests)

    async def dequeue(self, vendor: GatewayVendor) -> GatewayRequest | None:
        """Get the next highest-priority request for a vendor.

        Returns None if the queue is empty.
        """
        async with self._lock:
            queue = self._queues.get(vendor, [])
            if not queue:
                return None
            item = heappop(queue)
            return item.request

    async def peek(self, vendor: GatewayVendor) -> GatewayRequest | None:
        """Peek at the next request without removing it."""
        async with self._lock:
            queue = self._queues.get(vendor, [])
            if not queue:
                return None
            return queue[0].request

    def queue_size(self, vendor: GatewayVendor) -> int:
        """Get the number of pending requests for a vendor."""
        return len(self._queues.get(vendor, []))

    def total_size(self) -> int:
        """Total requests across all vendors."""
        return sum(len(q) for q in self._queues.values())

    def get_stats(self) -> dict:
        """Get queue statistics."""
        return {
            "total": self.total_size(),
            "by_vendor": {v.value: len(self._queues.get(v, [])) for v in GatewayVendor},
        }

    @staticmethod
    def build_from_prompts(
        final_prompts: list,
        tenant_id: str,
        project_id: int,
        llm_query_id: int = 0,
        api_keys: dict[str, str] | None = None,
        vendor_models: dict[str, str] | None = None,
    ) -> list[GatewayRequest]:
        """Convert Module 1 FinalPrompt objects into GatewayRequests.

        Handles:
          - Vendor mapping (TargetLLM → GatewayVendor)
          - Session_ID generation for Resilience Runs
          - Priority assignment based on intent

        Args:
            final_prompts: List of FinalPrompt from Module 1
            tenant_id: Tenant UUID
            project_id: Project ID
            llm_query_id: Optional LLM query ID reference
            api_keys: Not used here (keys are resolved at dispatch time)
            vendor_models: Optional {vendor: model} overrides

        Returns:
            List of GatewayRequest ready for enqueue
        """
        from app.prompt_engine.types import Intent, TargetLLM

        # Map TargetLLM → GatewayVendor
        _llm_to_vendor = {
            TargetLLM.CHATGPT: GatewayVendor.CHATGPT,
            TargetLLM.GEMINI: GatewayVendor.GEMINI,
            TargetLLM.DEEPSEEK: GatewayVendor.DEEPSEEK,
            TargetLLM.YANDEXGPT: GatewayVendor.YANDEXGPT,
        }

        # Intent → Priority
        _intent_priority = {
            Intent.COMPARATIVE: RequestPriority.HIGH,
            Intent.INFORMATIONAL: RequestPriority.NORMAL,
            Intent.NAVIGATIONAL: RequestPriority.NORMAL,
            Intent.TRANSACTIONAL: RequestPriority.NORMAL,
        }

        vendor_models = vendor_models or {}
        requests: list[GatewayRequest] = []

        for prompt in final_prompts:
            vendor = _llm_to_vendor.get(prompt.target_llm)
            if vendor is None:
                logger.warning("No vendor mapping for %s, skipping", prompt.target_llm)
                continue

            model = vendor_models.get(vendor.value, "")
            priority = _intent_priority.get(prompt.intent, RequestPriority.NORMAL)

            # Generate session_id for resilience run grouping
            session_base = uuid.uuid4().hex[:12]

            # Create one request per resilience run
            for run_idx in range(prompt.resilience_runs):
                req = GatewayRequest(
                    vendor=vendor,
                    model=model,
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    temperature=prompt.temperature,
                    max_tokens=prompt.max_tokens,
                    session_id=f"{session_base}_run{run_idx}",
                    run_index=run_idx,
                    total_runs=prompt.resilience_runs,
                    priority=priority,
                    tenant_id=tenant_id,
                    project_id=project_id,
                    llm_query_id=llm_query_id,
                    prompt_id=prompt.prompt_id,
                )
                requests.append(req)

        logger.info(
            "Built %d gateway requests from %d prompts",
            len(requests),
            len(final_prompts),
        )
        return requests
