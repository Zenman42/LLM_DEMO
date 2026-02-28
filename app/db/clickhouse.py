import logging

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_ch_client() -> Client:
    """Get or create a ClickHouse client (sync — clickhouse-connect is sync-native)."""
    global _client
    if _client is None:
        _client = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password,
            database=settings.clickhouse_db,
        )
    return _client


def ch_health_check() -> bool:
    """Check if ClickHouse is reachable."""
    try:
        client = get_ch_client()
        result = client.query("SELECT 1")
        return result.result_rows == [(1,)]
    except Exception as e:
        logger.warning("ClickHouse health check failed: %s", e)
        return False


def init_ch_schema() -> None:
    """Create ClickHouse tables if they don't exist."""
    client = get_ch_client()
    client.command("""
        CREATE TABLE IF NOT EXISTS serp_details (
            tenant_id       UUID,
            keyword_id      UInt32,
            date            Date,
            search_engine   LowCardinality(String),
            position        UInt8,
            url             String,
            domain          String,
            title           String,
            snippet         String,
            inserted_at     DateTime DEFAULT now()
        )
        ENGINE = ReplacingMergeTree(inserted_at)
        PARTITION BY toYYYYMM(date)
        ORDER BY (tenant_id, keyword_id, date, search_engine, position)
        TTL date + INTERVAL 2 YEAR DELETE
        SETTINGS index_granularity = 8192
    """)
    logger.info("ClickHouse schema initialized")


def close_ch_client() -> None:
    """Close the ClickHouse client connection."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
