"""Adaptive Rate Limiter — per-vendor RPM/TPM tracking with sliding window.

Tracks requests-per-minute (RPM) and tokens-per-minute (TPM) for each vendor
using a sliding window approach. When a limit is hit, returns the wait time
until the next available slot.

Thread-safe via asyncio.Lock (one lock per vendor).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from app.gateway.types import GatewayVendor, VendorConfig

logger = logging.getLogger(__name__)


@dataclass
class _WindowEntry:
    """Single entry in the sliding window."""

    timestamp: float  # time.monotonic()
    tokens: int = 0  # Tokens consumed by this request


@dataclass
class _VendorBucket:
    """Sliding window bucket for a single vendor."""

    config: VendorConfig
    entries: deque[_WindowEntry] = field(default_factory=deque)
    active_count: int = 0  # Currently in-flight requests
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _prune(self, now: float) -> None:
        """Remove entries older than 60 seconds (1-minute window)."""
        cutoff = now - 60.0
        while self.entries and self.entries[0].timestamp < cutoff:
            self.entries.popleft()

    @property
    def current_rpm(self) -> int:
        """Requests in the current 1-minute window."""
        return len(self.entries)

    @property
    def current_tpm(self) -> int:
        """Tokens in the current 1-minute window."""
        return sum(e.tokens for e in self.entries)

    def wait_time(self, now: float) -> float:
        """Calculate how long to wait before the next request is allowed.

        Returns 0 if request can proceed immediately.
        """
        self._prune(now)

        # Check concurrent limit
        if self.active_count >= self.config.max_concurrent:
            return 0.5  # Brief wait for a slot to open

        # Check RPM
        if self.current_rpm >= self.config.rpm_limit:
            # Wait until the oldest entry expires from the window
            oldest_ts = self.entries[0].timestamp
            wait = (oldest_ts + 60.0) - now
            return max(wait, 0.1)

        # Check TPM
        if self.current_tpm >= self.config.tpm_limit:
            oldest_ts = self.entries[0].timestamp
            wait = (oldest_ts + 60.0) - now
            return max(wait, 0.1)

        return 0.0

    def record_request(self, now: float) -> None:
        """Record a new outgoing request (tokens added later via record_tokens)."""
        self.entries.append(_WindowEntry(timestamp=now, tokens=0))
        self.active_count += 1

    def record_completion(self, tokens: int = 0) -> None:
        """Record request completion and update token count."""
        self.active_count = max(0, self.active_count - 1)
        # Update the most recent entry with actual token count
        if self.entries and tokens > 0:
            self.entries[-1].tokens = tokens


class AdaptiveRateLimiter:
    """Per-vendor adaptive rate limiter with sliding window.

    Usage:
        limiter = AdaptiveRateLimiter(configs)

        # Before sending a request:
        wait = await limiter.acquire(vendor)
        if wait > 0:
            await asyncio.sleep(wait)

        # After response:
        limiter.release(vendor, total_tokens=1500)
    """

    def __init__(self, configs: dict[GatewayVendor, VendorConfig] | None = None):
        from app.gateway.types import DEFAULT_VENDOR_CONFIGS

        configs = configs or DEFAULT_VENDOR_CONFIGS
        self._buckets: dict[GatewayVendor, _VendorBucket] = {
            vendor: _VendorBucket(config=config) for vendor, config in configs.items()
        }

    def _get_bucket(self, vendor: GatewayVendor) -> _VendorBucket:
        """Get or create bucket for a vendor."""
        if vendor not in self._buckets:
            from app.gateway.types import DEFAULT_VENDOR_CONFIGS

            config = DEFAULT_VENDOR_CONFIGS.get(vendor, VendorConfig(vendor=vendor))
            self._buckets[vendor] = _VendorBucket(config=config)
        return self._buckets[vendor]

    async def acquire(self, vendor: GatewayVendor) -> float:
        """Try to acquire a rate limit slot for the vendor.

        Returns:
            Wait time in seconds. 0 means request can proceed immediately.
            If > 0, caller should sleep that long before retrying.
        """
        bucket = self._get_bucket(vendor)
        async with bucket.lock:
            now = time.monotonic()
            wait = bucket.wait_time(now)

            if wait <= 0:
                bucket.record_request(now)
                return 0.0

            return wait

    async def acquire_blocking(self, vendor: GatewayVendor, timeout: float = 120.0) -> bool:
        """Block until a rate limit slot is available.

        Returns True if acquired, False if timeout exceeded.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            wait = await self.acquire(vendor)
            if wait <= 0:
                return True
            # Sleep with jitter to avoid thundering herd
            sleep_time = min(wait, deadline - time.monotonic())
            if sleep_time <= 0:
                return False
            await asyncio.sleep(sleep_time)

        return False

    def release(self, vendor: GatewayVendor, total_tokens: int = 0) -> None:
        """Release a rate limit slot after request completion."""
        bucket = self._get_bucket(vendor)
        bucket.record_completion(tokens=total_tokens)

    def get_stats(self, vendor: GatewayVendor) -> dict:
        """Get current rate limit stats for a vendor."""
        bucket = self._get_bucket(vendor)
        now = time.monotonic()
        bucket._prune(now)
        return {
            "vendor": vendor.value,
            "current_rpm": bucket.current_rpm,
            "rpm_limit": bucket.config.rpm_limit,
            "current_tpm": bucket.current_tpm,
            "tpm_limit": bucket.config.tpm_limit,
            "active_requests": bucket.active_count,
            "max_concurrent": bucket.config.max_concurrent,
        }

    def get_all_stats(self) -> list[dict]:
        """Get stats for all configured vendors."""
        return [self.get_stats(v) for v in self._buckets]
