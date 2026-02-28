"""Sentry error tracking integration.

Initializes Sentry SDK if SENTRY_DSN env variable is set.
Does nothing otherwise — safe to call unconditionally.
"""

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def init_sentry() -> None:
    """Initialize Sentry if SENTRY_DSN is configured."""
    if not settings.sentry_dsn:
        logger.debug("Sentry DSN not configured — skipping")
        return

    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    from sentry_sdk.integrations.celery import CeleryIntegration

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.app_env,
        traces_sample_rate=0.1 if settings.app_env == "production" else 1.0,
        send_default_pii=False,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
            CeleryIntegration(),
        ],
    )
    logger.info("Sentry initialized (env=%s)", settings.app_env)
