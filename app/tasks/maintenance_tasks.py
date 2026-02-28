"""Celery tasks for database maintenance."""

import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_sync_sql(sql: str) -> None:
    """Execute raw SQL using a sync psycopg2 connection (for DDL operations)."""
    from app.core.config import settings

    import psycopg2

    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
    finally:
        conn.close()


@celery_app.task(name="create_partitions")
def create_partitions_task():
    """Create monthly partitions for the next 3 months.

    Runs on the 1st of each month at 03:00 UTC via Celery Beat.
    Uses the PG function created in the partitioning migration.
    """
    logger.info("Creating monthly partitions (3 months ahead)...")
    try:
        _run_sync_sql("SELECT create_monthly_partitions('serp_snapshots', 3)")
        _run_sync_sql("SELECT create_monthly_partitions('webmaster_data', 3)")
        _run_sync_sql("SELECT create_monthly_partitions('llm_snapshots', 3)")
        _run_sync_sql("SELECT create_monthly_partitions('ai_overview_snapshots', 3)")
        logger.info("Monthly partitions created successfully")
        return {"status": "ok"}
    except Exception as exc:
        logger.error("Failed to create partitions: %s", exc)
        return {"status": "error", "error": str(exc)}


@celery_app.task(name="drop_old_partitions")
def drop_old_partitions_task():
    """Drop partitions older than retention_months.

    Runs on the 2nd of each month at 04:00 UTC via Celery Beat,
    after partition creation on the 1st.
    Uses the PG function created in the retention migration.
    """
    from app.core.config import settings

    months = settings.retention_months
    logger.info("Dropping partitions older than %d months...", months)
    try:
        _run_sync_sql(f"SELECT drop_old_partitions('serp_snapshots', {months})")
        _run_sync_sql(f"SELECT drop_old_partitions('webmaster_data', {months})")
        _run_sync_sql(f"SELECT drop_old_partitions('llm_snapshots', {months})")
        _run_sync_sql(f"SELECT drop_old_partitions('ai_overview_snapshots', {months})")
        logger.info("Old partition cleanup completed successfully")
        return {"status": "ok", "retention_months": months}
    except Exception as exc:
        logger.error("Failed to drop old partitions: %s", exc)
        return {"status": "error", "error": str(exc)}
