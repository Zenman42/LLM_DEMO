import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import api_v1_router
from app.web.router import web_router
from app.core.config import settings, validate_settings_for_production
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.core.rate_limit import limiter
from app.db.clickhouse import ch_health_check, close_ch_client, init_ch_schema
from app.db.postgres import engine

# Configure logging before anything else
setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    validate_settings_for_production()
    logger.info("Starting Position Tracker SaaS...")

    # Init ClickHouse schema (non-blocking — CH is optional)
    try:
        init_ch_schema()
        logger.info("ClickHouse: connected and schema ready")
    except Exception as e:
        logger.warning("ClickHouse unavailable at startup: %s", e)

    yield

    # Shutdown
    close_ch_client()
    await engine.dispose()
    logger.info("Position Tracker SaaS shut down")


app = FastAPI(
    title="Position Tracker",
    description="SaaS SERP position tracker with PostgreSQL + ClickHouse",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.app_debug else None,
    redoc_url="/api/redoc" if settings.app_debug else None,
)


# Log unhandled exceptions so they appear in Railway logs
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    logger.error("Unhandled %s on %s %s:\n%s", type(exc).__name__, request.method, request.url.path, "".join(tb))
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# CORS — parse allowed_origins from settings (comma-separated)
_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo mode — bypass JWT auth, return demo user for all API requests
if settings.demo_mode and settings.demo_user_email:
    from app.core.dependencies import get_current_user, get_demo_user

    app.dependency_overrides[get_current_user] = get_demo_user
    logger.info("DEMO MODE enabled — auth bypassed for %s", settings.demo_user_email)

# Demo mode — force no-cache on all HTML responses to prevent CDN/Lovable caching
if settings.demo_mode:
    from starlette.middleware.base import BaseHTTPMiddleware

    class NoCacheMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            ct = response.headers.get("content-type", "")
            if "text/html" in ct:
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    app.add_middleware(NoCacheMiddleware)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# API routes
app.include_router(api_v1_router)

# Web UI routes
app.include_router(web_router)


@app.get("/api/v1/health")
async def health():
    ch_ok = ch_health_check()
    return {
        "status": "ok",
        "postgres": True,
        "clickhouse": ch_ok,
    }
