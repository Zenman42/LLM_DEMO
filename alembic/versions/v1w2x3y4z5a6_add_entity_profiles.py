"""Add entity_profiles table and extend discovered_entities for ProLEA clustering.

Creates:
  - entity_profiles table (embedding stored as TEXT — pgvector not required;
    the clustering pipeline loads embeddings into numpy and computes similarity
    in Python via scipy, so native vector type is unnecessary)
  - entity_type and profile_id columns on discovered_entities

Revision ID: v1w2x3y4z5a6
Revises: u1v2w3x4y5z6
Create Date: 2026-02-25
"""

import sqlalchemy as sa
from alembic import op

revision = "v1w2x3y4z5a6"
down_revision = "u1v2w3x4y5z6"


def upgrade():
    conn = op.get_bind()

    # Create entity_profiles table.
    # Embedding column is TEXT: the pgvector Python adapter serializes vectors
    # to '[0.1,0.2,...]' strings and parses them back on read.  All similarity
    # computation is done in Python (numpy/scipy), so native pgvector is not
    # needed on the DB side.
    conn.execute(
        sa.text("""
        CREATE TABLE IF NOT EXISTS entity_profiles (
            id SERIAL PRIMARY KEY,
            tenant_id UUID NOT NULL,
            project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            entity_key VARCHAR(255) NOT NULL,
            canonical_name VARCHAR(255) NOT NULL,
            entity_type VARCHAR(50),
            industry VARCHAR(255),
            description TEXT,
            attributes JSONB,
            related_entities JSONB,
            embedding TEXT,
            embedding_text_hash VARCHAR(16),
            profile_source VARCHAR(20),
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now(),
            CONSTRAINT uq_entity_profile_key UNIQUE (project_id, entity_key)
        )
    """)
    )
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_entity_profiles_tenant_id ON entity_profiles (tenant_id)"))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_entity_profiles_project_id ON entity_profiles (project_id)"))

    # Add new columns to discovered_entities (IF NOT EXISTS guards for idempotency)
    res = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='discovered_entities' AND column_name='entity_type'"
        )
    )
    if not res.fetchone():
        op.add_column(
            "discovered_entities",
            sa.Column("entity_type", sa.String(50), nullable=True),
        )

    res = conn.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='discovered_entities' AND column_name='profile_id'"
        )
    )
    if not res.fetchone():
        op.add_column(
            "discovered_entities",
            sa.Column(
                "profile_id",
                sa.Integer,
                sa.ForeignKey("entity_profiles.id", ondelete="SET NULL"),
                nullable=True,
            ),
        )


def downgrade():
    op.drop_column("discovered_entities", "profile_id")
    op.drop_column("discovered_entities", "entity_type")
    op.drop_table("entity_profiles")
