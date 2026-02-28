"""Core types and DTOs for the LLM API Gateway Layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GatewayVendor(str, Enum):
    """Supported LLM vendors."""

    CHATGPT = "chatgpt"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    YANDEXGPT = "yandexgpt"
    PERPLEXITY = "perplexity"
    GIGACHAT = "gigachat"


class RequestStatus(str, Enum):
    """Status of a gateway request through its lifecycle."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    VENDOR_ERROR = "vendor_error"
    TIMEOUT = "timeout"
    CENSORED = "censored"  # Gemini SAFETY filter or similar
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker tripped
    DEAD_LETTER = "dead_letter"  # All retries exhausted
    CANCELLED = "cancelled"


class RequestPriority(int, Enum):
    """Priority levels for the queue (lower = higher priority)."""

    CRITICAL = 0  # Manual trigger from UI
    HIGH = 1  # Comparative intent (key metric)
    NORMAL = 2  # Standard collection
    LOW = 3  # Background / bulk expansion


# ---------------------------------------------------------------------------
# Gateway Request — input to the gateway
# ---------------------------------------------------------------------------


@dataclass
class GatewayRequest:
    """A single prompt request to dispatch to an LLM vendor.

    This is the input to the Gateway, typically constructed from
    Module 1's FinalPrompt + execution context.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    vendor: GatewayVendor = GatewayVendor.CHATGPT
    model: str = ""  # e.g. "gpt-4o-mini", "deepseek-chat"
    system_prompt: str = ""
    user_prompt: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048

    # Session management for Resilience Runs
    session_id: str = ""  # Unique per resilience group
    run_index: int = 0  # Which run within the group (0, 1, 2...)
    total_runs: int = 1  # Total resilience runs for this prompt

    # Scheduling
    priority: RequestPriority = RequestPriority.NORMAL

    # Context metadata (passed through to response)
    tenant_id: str = ""
    project_id: int = 0
    llm_query_id: int = 0
    prompt_id: str = ""  # Reference back to Module 1's FinalPrompt

    # Vendor-specific overrides
    vendor_params: dict[str, Any] = field(default_factory=dict)

    @property
    def queue_key(self) -> str:
        """Key for deduplication and routing."""
        return f"{self.vendor.value}:{self.request_id}"


# ---------------------------------------------------------------------------
# Gateway Response — unified DTO (output of the gateway)
# ---------------------------------------------------------------------------


@dataclass
class GatewayResponse:
    """Unified response DTO from any LLM vendor.

    This is the normalized output of the Gateway — same structure
    regardless of which vendor produced it.
    """

    request_id: str = ""
    vendor: GatewayVendor = GatewayVendor.CHATGPT
    model_version: str = ""  # Actual model reported by vendor
    status: RequestStatus = RequestStatus.SUCCESS

    # Content
    raw_response_text: str = ""
    cited_urls: list[str] = field(default_factory=list)

    # Performance
    latency_ms: int = 0  # Total round-trip time in milliseconds
    retry_count: int = 0  # How many retries were needed

    # Tokens & cost
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    # Timestamps
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Error details (if status != SUCCESS)
    error_code: str = ""  # e.g. "429", "503", "SAFETY"
    error_message: str = ""

    # Session context (passed through from request)
    session_id: str = ""
    run_index: int = 0
    tenant_id: str = ""
    project_id: int = 0
    llm_query_id: int = 0
    prompt_id: str = ""

    # Vendor-specific raw data
    vendor_raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for storage/API."""
        return {
            "request_id": self.request_id,
            "vendor": self.vendor.value,
            "model_version": self.model_version,
            "status": self.status.value,
            "raw_response_text": self.raw_response_text,
            "cited_urls": self.cited_urls,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "queued_at": self.queued_at.isoformat() if self.queued_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "session_id": self.session_id,
            "run_index": self.run_index,
            "prompt_id": self.prompt_id,
        }


# ---------------------------------------------------------------------------
# Dead Letter Entry
# ---------------------------------------------------------------------------


@dataclass
class DeadLetterEntry:
    """A request that exhausted all retries and ended up in the Dead Letter Queue."""

    request: GatewayRequest
    last_response: GatewayResponse
    attempts: int = 0
    last_error: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Vendor config
# ---------------------------------------------------------------------------


@dataclass
class VendorConfig:
    """Rate limit and connection configuration for a vendor."""

    vendor: GatewayVendor
    rpm_limit: int = 60  # Requests per minute
    tpm_limit: int = 100_000  # Tokens per minute
    max_concurrent: int = 10  # Max concurrent requests
    timeout_seconds: float = 60.0  # Request timeout
    max_retries: int = 5  # Max retries before dead letter
    base_retry_delay: float = 1.0  # Base delay for exponential backoff (seconds)
    max_retry_delay: float = 60.0  # Cap on retry delay


# Default configurations per vendor
DEFAULT_VENDOR_CONFIGS: dict[GatewayVendor, VendorConfig] = {
    GatewayVendor.CHATGPT: VendorConfig(
        vendor=GatewayVendor.CHATGPT,
        rpm_limit=60,
        tpm_limit=200_000,
        max_concurrent=10,
        timeout_seconds=60,
    ),
    GatewayVendor.GEMINI: VendorConfig(
        vendor=GatewayVendor.GEMINI,
        rpm_limit=60,
        tpm_limit=200_000,
        max_concurrent=10,
        timeout_seconds=60,
    ),
    GatewayVendor.DEEPSEEK: VendorConfig(
        vendor=GatewayVendor.DEEPSEEK,
        rpm_limit=30,
        tpm_limit=100_000,
        max_concurrent=5,
        timeout_seconds=120,  # DeepSeek: increased timeout
        max_retries=5,
        base_retry_delay=2.0,  # Slower retries
    ),
    GatewayVendor.YANDEXGPT: VendorConfig(
        vendor=GatewayVendor.YANDEXGPT,
        rpm_limit=10,
        tpm_limit=50_000,
        max_concurrent=3,
        timeout_seconds=90,
        max_retries=5,
    ),
    GatewayVendor.PERPLEXITY: VendorConfig(
        vendor=GatewayVendor.PERPLEXITY,
        rpm_limit=20,
        tpm_limit=100_000,
        max_concurrent=5,
        timeout_seconds=90,
    ),
    GatewayVendor.GIGACHAT: VendorConfig(
        vendor=GatewayVendor.GIGACHAT,
        rpm_limit=3,
        tpm_limit=50_000,
        max_concurrent=1,
        timeout_seconds=90,
    ),
}
