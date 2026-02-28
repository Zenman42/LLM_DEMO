"""add partitioning and partition helper

Revision ID: c3a1f7d84e20
Revises: eb9ae91657c9
Create Date: 2026-02-15 22:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c3a1f7d84e20"
down_revision: Union[str, None] = "eb9ae91657c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # =========================================================
    # Convert serp_snapshots to a partitioned table.
    # This drops the old non-partitioned table and recreates it
    # with PARTITION BY RANGE (date).
    # Safe only on a fresh/empty DB — migration plan says "clean DB".
    # =========================================================

    op.execute("DROP TABLE IF EXISTS serp_snapshots CASCADE")
    op.execute("""
        CREATE TABLE serp_snapshots (
            id          BIGINT GENERATED ALWAYS AS IDENTITY,
            tenant_id   UUID NOT NULL,
            keyword_id  INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
            date        DATE NOT NULL,
            search_engine VARCHAR(10) NOT NULL,
            position    SMALLINT,
            found_url   VARCHAR(2048),
            previous_position SMALLINT,
            collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_serp_snapshot UNIQUE (keyword_id, date, search_engine)
        ) PARTITION BY RANGE (date)
    """)
    op.execute("CREATE INDEX ix_serp_snapshots_tenant_id ON serp_snapshots (tenant_id)")
    op.execute("CREATE INDEX ix_serp_snapshots_kw_date ON serp_snapshots (keyword_id, date, search_engine)")

    # =========================================================
    # Convert webmaster_data to a partitioned table.
    # =========================================================

    op.execute("DROP TABLE IF EXISTS webmaster_data CASCADE")
    op.execute("""
        CREATE TABLE webmaster_data (
            id          BIGINT GENERATED ALWAYS AS IDENTITY,
            tenant_id   UUID NOT NULL,
            keyword_id  INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
            date        DATE NOT NULL,
            search_engine VARCHAR(10) NOT NULL,
            impressions INTEGER NOT NULL DEFAULT 0,
            clicks      INTEGER NOT NULL DEFAULT 0,
            ctr         REAL,
            position    REAL,
            page_url    VARCHAR(2048),
            collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_webmaster_data UNIQUE (keyword_id, date, search_engine)
        ) PARTITION BY RANGE (date)
    """)
    op.execute("CREATE INDEX ix_webmaster_data_tenant_id ON webmaster_data (tenant_id)")
    op.execute("CREATE INDEX ix_webmaster_data_kw_date ON webmaster_data (keyword_id, date, search_engine)")

    # =========================================================
    # PG function: create_monthly_partitions(table_name, months_ahead)
    # Creates monthly partitions for the given table, skipping
    # partitions that already exist.
    # Run periodically via Celery Beat or a cron job.
    # =========================================================

    op.execute("""
        CREATE OR REPLACE FUNCTION create_monthly_partitions(
            p_table TEXT,
            p_months_ahead INT DEFAULT 3
        ) RETURNS void AS $$
        DECLARE
            v_start DATE;
            v_end   DATE;
            v_part  TEXT;
            v_i     INT;
        BEGIN
            FOR v_i IN 0..p_months_ahead LOOP
                v_start := date_trunc('month', CURRENT_DATE) + (v_i || ' months')::INTERVAL;
                v_end   := v_start + '1 month'::INTERVAL;
                v_part  := p_table || '_' || to_char(v_start, 'YYYY_MM');

                IF NOT EXISTS (
                    SELECT 1 FROM pg_class WHERE relname = v_part
                ) THEN
                    EXECUTE format(
                        'CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                        v_part, p_table, v_start, v_end
                    );
                    RAISE NOTICE 'Created partition %', v_part;
                END IF;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create partitions: current month + 3 months ahead
    op.execute("SELECT create_monthly_partitions('serp_snapshots', 3)")
    op.execute("SELECT create_monthly_partitions('webmaster_data', 3)")


def downgrade() -> None:
    # Drop partitioned tables and recreate as non-partitioned
    op.execute("DROP FUNCTION IF EXISTS create_monthly_partitions(TEXT, INT)")

    op.execute("DROP TABLE IF EXISTS serp_snapshots CASCADE")
    op.execute("DROP TABLE IF EXISTS webmaster_data CASCADE")

    # Recreate non-partitioned versions (matching initial migration)
    op.execute("""
        CREATE TABLE serp_snapshots (
            id          BIGSERIAL PRIMARY KEY,
            tenant_id   UUID NOT NULL,
            keyword_id  INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
            date        DATE NOT NULL,
            search_engine VARCHAR(10) NOT NULL,
            position    SMALLINT,
            found_url   VARCHAR(2048),
            previous_position SMALLINT,
            collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_serp_snapshot UNIQUE (keyword_id, date, search_engine)
        )
    """)
    op.execute("CREATE INDEX ix_serp_snapshots_tenant_id ON serp_snapshots (tenant_id)")

    op.execute("""
        CREATE TABLE webmaster_data (
            id          BIGSERIAL PRIMARY KEY,
            tenant_id   UUID NOT NULL,
            keyword_id  INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
            date        DATE NOT NULL,
            search_engine VARCHAR(10) NOT NULL,
            impressions INTEGER NOT NULL DEFAULT 0,
            clicks      INTEGER NOT NULL DEFAULT 0,
            ctr         REAL,
            position    REAL,
            page_url    VARCHAR(2048),
            collected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_webmaster_data UNIQUE (keyword_id, date, search_engine)
        )
    """)
    op.execute("CREATE INDEX ix_webmaster_data_tenant_id ON webmaster_data (tenant_id)")
