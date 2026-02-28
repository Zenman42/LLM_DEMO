"""Prometheus metrics for the application."""

import time

from prometheus_client import Counter, Histogram, Info, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# --- Metrics ---

APP_INFO = Info("app", "Position Tracker application info")
APP_INFO.info({"version": "2.0.0", "name": "position_tracker"})

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)

COLLECTION_RUNS = Counter(
    "collection_runs_total",
    "Total collection task runs",
    ["tenant_id", "status"],
)

ACTIVE_TENANTS = Counter(
    "tenants_registered_total",
    "Total tenants registered",
)


# --- Middleware ---

# Normalize dynamic path segments to reduce cardinality
_PATH_PREFIXES = ("/api/v1/projects/", "/api/v1/serp/", "/project/")


def _normalize_path(path: str) -> str:
    """Replace numeric IDs in paths with {id} to avoid high cardinality."""
    for prefix in _PATH_PREFIXES:
        if path.startswith(prefix):
            rest = path[len(prefix) :]
            parts = rest.split("/", 1)
            if parts[0].isdigit():
                tail = f"/{parts[1]}" if len(parts) > 1 else ""
                return f"{prefix}{{id}}{tail}"
    return path


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Collect HTTP request metrics for Prometheus."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        path = _normalize_path(request.url.path)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(method=method, path=path, status=response.status_code).inc()
        REQUEST_DURATION.labels(method=method, path=path).observe(duration)

        return response


def metrics_response() -> Response:
    """Generate Prometheus /metrics response."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
