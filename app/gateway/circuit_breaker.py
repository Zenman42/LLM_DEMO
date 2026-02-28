"""Circuit Breaker with exponential backoff and jitter.

Implements the circuit breaker pattern per vendor:
  - CLOSED: normal operation, requests pass through
  - OPEN: too many failures, requests are rejected immediately
  - HALF_OPEN: testing recovery with a single probe request

Backoff strategy:
  delay = min(base * 2^attempt + jitter, max_delay)
  jitter = random(0, base * 0.5)

After max_retries failures → request goes to Dead Letter Queue.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum

from app.gateway.types import (
    DeadLetterEntry,
    GatewayRequest,
    GatewayResponse,
    GatewayVendor,
    VendorConfig,
)

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class _CircuitStats:
    """Failure tracking for a single vendor's circuit."""

    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: float = 0.0
    state: CircuitState = CircuitState.CLOSED
    opened_at: float = 0.0  # When circuit was opened


# Thresholds for opening the circuit
FAILURE_THRESHOLD = 5  # Consecutive failures to open circuit
RECOVERY_TIMEOUT = 60.0  # Seconds before trying half-open
HALF_OPEN_MAX_PROBES = 1  # Max concurrent probes in half-open state


class CircuitBreaker:
    """Per-vendor circuit breaker with exponential backoff.

    Usage:
        cb = CircuitBreaker()

        # Check if request is allowed
        if not cb.allow_request(vendor):
            # Circuit is open, reject
            ...

        # After success:
        cb.record_success(vendor)

        # After failure:
        retry_delay = cb.record_failure(vendor, attempt=2, config=vendor_config)
        if retry_delay is None:
            # Max retries exceeded, dead letter
            ...
        else:
            await asyncio.sleep(retry_delay)
    """

    def __init__(self):
        self._circuits: dict[GatewayVendor, _CircuitStats] = {}
        self._dead_letter: list[DeadLetterEntry] = []
        self._lock = asyncio.Lock()

    def _get_circuit(self, vendor: GatewayVendor) -> _CircuitStats:
        if vendor not in self._circuits:
            self._circuits[vendor] = _CircuitStats()
        return self._circuits[vendor]

    def allow_request(self, vendor: GatewayVendor) -> bool:
        """Check if a request to the vendor is allowed.

        Returns True if the circuit is closed or half-open (probe allowed).
        """
        circuit = self._get_circuit(vendor)
        now = time.monotonic()

        if circuit.state == CircuitState.CLOSED:
            return True

        if circuit.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if now - circuit.opened_at >= RECOVERY_TIMEOUT:
                circuit.state = CircuitState.HALF_OPEN
                logger.info("Circuit for %s transitioning to HALF_OPEN", vendor.value)
                return True
            return False

        if circuit.state == CircuitState.HALF_OPEN:
            # Allow one probe
            return True

        return False

    def record_success(self, vendor: GatewayVendor) -> None:
        """Record a successful request — resets failure counter, closes circuit."""
        circuit = self._get_circuit(vendor)
        circuit.consecutive_failures = 0
        circuit.total_successes += 1

        if circuit.state != CircuitState.CLOSED:
            logger.info("Circuit for %s CLOSED (recovered)", vendor.value)
            circuit.state = CircuitState.CLOSED

    def record_failure(
        self,
        vendor: GatewayVendor,
        attempt: int,
        config: VendorConfig,
    ) -> float | None:
        """Record a failure and calculate retry delay.

        Args:
            vendor: The vendor that failed
            attempt: Current attempt number (0-based)
            config: Vendor config with retry parameters

        Returns:
            Retry delay in seconds, or None if max retries exceeded (dead letter).
        """
        circuit = self._get_circuit(vendor)
        circuit.consecutive_failures += 1
        circuit.total_failures += 1
        circuit.last_failure_time = time.monotonic()

        # Check if circuit should open
        if circuit.consecutive_failures >= FAILURE_THRESHOLD:
            if circuit.state != CircuitState.OPEN:
                circuit.state = CircuitState.OPEN
                circuit.opened_at = time.monotonic()
                logger.warning(
                    "Circuit for %s OPENED after %d consecutive failures",
                    vendor.value,
                    circuit.consecutive_failures,
                )

        # Check if max retries exceeded → dead letter
        if attempt >= config.max_retries:
            return None

        # Calculate exponential backoff with jitter
        delay = self.calculate_backoff(
            attempt=attempt,
            base_delay=config.base_retry_delay,
            max_delay=config.max_retry_delay,
        )

        logger.info(
            "Retry %d/%d for %s in %.1fs",
            attempt + 1,
            config.max_retries,
            vendor.value,
            delay,
        )

        return delay

    @staticmethod
    def calculate_backoff(
        attempt: int,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> float:
        """Calculate exponential backoff with jitter.

        Formula: min(base * 2^attempt + jitter, max_delay)
        Jitter: random(0, base * 0.5)
        """
        exponential = base_delay * (2**attempt)
        jitter = random.uniform(0, base_delay * 0.5)
        delay = min(exponential + jitter, max_delay)
        return delay

    def add_to_dead_letter(
        self,
        request: GatewayRequest,
        response: GatewayResponse,
        attempts: int,
    ) -> None:
        """Add a request to the Dead Letter Queue."""
        entry = DeadLetterEntry(
            request=request,
            last_response=response,
            attempts=attempts,
            last_error=response.error_message,
        )
        self._dead_letter.append(entry)
        logger.warning(
            "Request %s sent to Dead Letter Queue after %d attempts: %s",
            request.request_id,
            attempts,
            response.error_message,
        )

    def get_dead_letters(self) -> list[DeadLetterEntry]:
        """Return all dead letter entries."""
        return list(self._dead_letter)

    def clear_dead_letters(self) -> int:
        """Clear the dead letter queue. Returns count of cleared entries."""
        count = len(self._dead_letter)
        self._dead_letter.clear()
        return count

    def get_circuit_state(self, vendor: GatewayVendor) -> dict:
        """Get the current state of a vendor's circuit."""
        circuit = self._get_circuit(vendor)
        return {
            "vendor": vendor.value,
            "state": circuit.state.value,
            "consecutive_failures": circuit.consecutive_failures,
            "total_failures": circuit.total_failures,
            "total_successes": circuit.total_successes,
        }

    def get_all_states(self) -> list[dict]:
        """Get circuit states for all vendors."""
        return [self.get_circuit_state(v) for v in GatewayVendor]

    def reset(self, vendor: GatewayVendor) -> None:
        """Manually reset a vendor's circuit to CLOSED."""
        circuit = self._get_circuit(vendor)
        circuit.state = CircuitState.CLOSED
        circuit.consecutive_failures = 0
        logger.info("Circuit for %s manually RESET", vendor.value)
