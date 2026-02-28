"""add LLM tracking tables and columns

Revision ID: a1b2c3d4e5f6
Revises: d1f2a3b4c5d6
Create Date: 2026-02-17 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "d1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # =========================================================
    # 1. Add LLM columns to tenants
    # =========================================================
    op.add_column("tenants", sa.Column("max_llm_queries", sa.Integer(), nullable=False, server_default="50"))
    op.add_column("tenants", sa.Column("openai_api_key", sa.LargeBinary(), nullable=True))
    op.add_column("tenants", sa.Column("google_ai_api_key", sa.LargeBinary(), nullable=True))
    op.add_column("tenants", sa.Column("anthropic_api_key", sa.LargeBinary(), nullable=True))
    op.add_column("tenants", sa.Column("perplexity_api_key", sa.LargeBinary(), nullable=True))

    # =========================================================
    # 2. Add LLM columns to projects
    # =========================================================
    op.add_column("projects", sa.Column("track_llm", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("projects", sa.Column("track_ai_overview", sa.Boolean(), nullable=False, server_default="false"))
    op.add_column("projects", sa.Column("llm_providers", JSONB(), nullable=True))
    op.add_column("projects", sa.Column("brand_name", sa.String(255), nullable=True))

    # =========================================================
    # 3. Create llm_queries table
    # =========================================================
    op.create_table(
        "llm_queries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "project_id", sa.Integer(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
        ),
        sa.Column("query_text", sa.String(2000), nullable=False),
        sa.Column("query_type", sa.String(20), nullable=False, server_default="custom"),
        sa.Column("target_brand", sa.String(255), nullable=True),
        sa.Column("competitors", JSONB(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("project_id", "query_text", name="uq_project_llm_query"),
    )

    # =========================================================
    # 4. Create llm_snapshots (partitioned by date)
    # =========================================================
    op.execute("""
        CREATE TABLE llm_snapshots (
            id              BIGINT GENERATED ALWAYS AS IDENTITY,
            tenant_id       UUID NOT NULL,
            llm_query_id    INTEGER NOT NULL REFERENCES llm_queries(id) ON DELETE CASCADE,
            date            DATE NOT NULL,
            llm_provider    VARCHAR(20) NOT NULL,
            llm_model       VARCHAR(50) NOT NULL,
            brand_mentioned BOOLEAN NOT NULL DEFAULT false,
            mention_type    VARCHAR(20) NOT NULL DEFAULT 'none',
            mention_context VARCHAR(1000),
            competitor_mentions JSONB,
            cited_urls      JSONB,
            raw_response    TEXT,
            response_tokens INTEGER,
            cost_usd        REAL,
            collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_llm_snapshot UNIQUE (llm_query_id, date, llm_provider)
        ) PARTITION BY RANGE (date)
    """)
    op.execute("CREATE INDEX ix_llm_snapshots_tenant_id ON llm_snapshots (tenant_id)")
    op.execute("CREATE INDEX ix_llm_snapshots_query_date ON llm_snapshots (llm_query_id, date, llm_provider)")

    # =========================================================
    # 5. Create ai_overview_snapshots (partitioned by date)
    # =========================================================
    op.execute("""
        CREATE TABLE ai_overview_snapshots (
            id              BIGINT GENERATED ALWAYS AS IDENTITY,
            tenant_id       UUID NOT NULL,
            keyword_id      INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
            date            DATE NOT NULL,
            has_ai_overview BOOLEAN NOT NULL DEFAULT false,
            brand_in_aio    BOOLEAN NOT NULL DEFAULT false,
            aio_position    SMALLINT,
            aio_sources     JSONB,
            aio_snippet     TEXT,
            collected_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_ai_overview_snapshot UNIQUE (keyword_id, date)
        ) PARTITION BY RANGE (date)
    """)
    op.execute("CREATE INDEX ix_ai_overview_snapshots_tenant_id ON ai_overview_snapshots (tenant_id)")
    op.execute("CREATE INDEX ix_ai_overview_snapshots_kw_date ON ai_overview_snapshots (keyword_id, date)")

    # =========================================================
    # 6. Create initial partitions for new tables
    # =========================================================
    op.execute("SELECT create_monthly_partitions('llm_snapshots', 3)")
    op.execute("SELECT create_monthly_partitions('ai_overview_snapshots', 3)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ai_overview_snapshots CASCADE")
    op.execute("DROP TABLE IF EXISTS llm_snapshots CASCADE")
    op.drop_table("llm_queries")
    op.drop_column("projects", "brand_name")
    op.drop_column("projects", "llm_providers")
    op.drop_column("projects", "track_ai_overview")
    op.drop_column("projects", "track_llm")
    op.drop_column("tenants", "perplexity_api_key")
    op.drop_column("tenants", "anthropic_api_key")
    op.drop_column("tenants", "google_ai_api_key")
    op.drop_column("tenants", "openai_api_key")
    op.drop_column("tenants", "max_llm_queries")
