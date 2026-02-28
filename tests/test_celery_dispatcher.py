"""Tests for Celery Beat dispatcher task."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasks.collection_tasks import _find_tenants_to_collect


@pytest.mark.asyncio
async def test_find_tenants_to_collect(db: AsyncSession, tenant_and_user):
    """Dispatcher finds tenants whose schedule matches current time."""
    tenant, user = tenant_and_user

    now = datetime.now(timezone.utc)

    # Set tenant collection to current hour:minute
    tenant.collection_hour = now.hour
    tenant.collection_minute = now.minute
    await db.commit()

    from tests.conftest import test_session_factory

    with patch(
        "app.tasks.collection_tasks._make_session_factory",
        return_value=test_session_factory,
    ):
        tenant_ids = await _find_tenants_to_collect()

    assert len(tenant_ids) == 1
    assert tenant_ids[0] == str(tenant.id)


@pytest.mark.asyncio
async def test_find_tenants_no_match(db: AsyncSession, tenant_and_user):
    """Dispatcher returns empty when no tenant matches current time."""
    tenant, user = tenant_and_user

    # Set tenant collection to a different hour
    now = datetime.now(timezone.utc)
    tenant.collection_hour = (now.hour + 5) % 24  # guaranteed different
    tenant.collection_minute = now.minute
    await db.commit()

    from tests.conftest import test_session_factory

    with patch(
        "app.tasks.collection_tasks._make_session_factory",
        return_value=test_session_factory,
    ):
        tenant_ids = await _find_tenants_to_collect()

    assert len(tenant_ids) == 0
