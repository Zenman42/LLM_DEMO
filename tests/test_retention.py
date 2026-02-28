"""Tests for data retention task.

Tests the task function logic at the unit level using mocks.
The PL/pgSQL function itself is tested via integration when running migrations.
"""

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_drop_old_partitions_task_calls_sql():
    """Task should call _run_sync_sql for both tables."""
    with patch("app.tasks.maintenance_tasks._run_sync_sql") as mock_sql:
        from app.tasks.maintenance_tasks import drop_old_partitions_task

        result = drop_old_partitions_task()

        assert result["status"] == "ok"
        assert result["retention_months"] == 24
        assert mock_sql.call_count == 4

        # Verify SQL calls contain correct function invocations
        calls = [str(c) for c in mock_sql.call_args_list]
        assert any("serp_snapshots" in c for c in calls)
        assert any("webmaster_data" in c for c in calls)
        assert any("llm_snapshots" in c for c in calls)
        assert any("ai_overview_snapshots" in c for c in calls)


@pytest.mark.asyncio
async def test_drop_old_partitions_task_handles_error():
    """Task should return error status on exception."""
    with patch(
        "app.tasks.maintenance_tasks._run_sync_sql",
        side_effect=Exception("DB connection failed"),
    ):
        from app.tasks.maintenance_tasks import drop_old_partitions_task

        result = drop_old_partitions_task()

        assert result["status"] == "error"
        assert "DB connection failed" in result["error"]
