"""Add query_tags and query_tag_links tables for many-to-many prompt tagging.

Migrates existing query_class and category data into the new tag system.

Revision ID: t1u2v3w4x5y6
Revises: s1t2u3v4w5x6
Create Date: 2025-05-22
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "t1u2v3w4x5y6"
down_revision = "s1t2u3v4w5x6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create query_tags table
    op.create_table(
        "query_tags",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "project_id",
            sa.Integer,
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tag_name", sa.String(255), nullable=False),
        sa.Column("tag_type", sa.String(20), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_query_tags_project_id", "query_tags", ["project_id"])
    op.create_index("ix_query_tags_tenant_id", "query_tags", ["tenant_id"])
    op.create_index("ix_query_tags_tag_type", "query_tags", ["tag_type"])
    op.create_unique_constraint(
        "uq_query_tag_per_project",
        "query_tags",
        ["project_id", "tag_name", "tag_type"],
    )

    # 2. Create query_tag_links junction table
    op.create_table(
        "query_tag_links",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "query_id",
            sa.Integer,
            sa.ForeignKey("llm_queries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tag_id",
            sa.Integer,
            sa.ForeignKey("query_tags.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    op.create_unique_constraint(
        "uq_query_tag_link",
        "query_tag_links",
        ["query_id", "tag_id"],
    )
    op.create_index("ix_query_tag_links_query_id", "query_tag_links", ["query_id"])
    op.create_index("ix_query_tag_links_tag_id", "query_tag_links", ["tag_id"])

    # 3. Data migration — populate tags from existing columns
    conn = op.get_bind()

    # 3a. Create 'class' tags from existing query_class values
    conn.execute(
        sa.text("""
            INSERT INTO query_tags (project_id, tenant_id, tag_name, tag_type)
            SELECT DISTINCT q.project_id, q.tenant_id, q.query_class, 'class'
            FROM llm_queries q
            WHERE q.query_class IS NOT NULL AND q.query_class != ''
            ON CONFLICT (project_id, tag_name, tag_type) DO NOTHING
        """)
    )

    # 3b. Create 'scenario' tags from existing category values
    conn.execute(
        sa.text("""
            INSERT INTO query_tags (project_id, tenant_id, tag_name, tag_type)
            SELECT DISTINCT q.project_id, q.tenant_id, q.category, 'scenario'
            FROM llm_queries q
            WHERE q.category IS NOT NULL AND q.category != ''
            ON CONFLICT (project_id, tag_name, tag_type) DO NOTHING
        """)
    )

    # 3c. Create scenario tags from project.categories JSONB
    conn.execute(
        sa.text("""
            INSERT INTO query_tags (project_id, tenant_id, tag_name, tag_type)
            SELECT p.id, p.tenant_id,
                   CASE
                       WHEN jsonb_typeof(elem) = 'string' THEN elem #>> '{}'
                       ELSE elem ->> 'name'
                   END,
                   'scenario'
            FROM projects p, jsonb_array_elements(p.categories) AS elem
            WHERE p.categories IS NOT NULL
              AND jsonb_typeof(p.categories) = 'array'
              AND jsonb_array_length(p.categories) > 0
            ON CONFLICT (project_id, tag_name, tag_type) DO NOTHING
        """)
    )

    # 3d. Link queries to their class tags
    conn.execute(
        sa.text("""
            INSERT INTO query_tag_links (query_id, tag_id)
            SELECT q.id, t.id
            FROM llm_queries q
            JOIN query_tags t
              ON t.project_id = q.project_id
             AND t.tag_name = q.query_class
             AND t.tag_type = 'class'
            WHERE q.query_class IS NOT NULL AND q.query_class != ''
            ON CONFLICT (query_id, tag_id) DO NOTHING
        """)
    )

    # 3e. Link queries to their scenario tags
    conn.execute(
        sa.text("""
            INSERT INTO query_tag_links (query_id, tag_id)
            SELECT q.id, t.id
            FROM llm_queries q
            JOIN query_tags t
              ON t.project_id = q.project_id
             AND t.tag_name = q.category
             AND t.tag_type = 'scenario'
            WHERE q.category IS NOT NULL AND q.category != ''
            ON CONFLICT (query_id, tag_id) DO NOTHING
        """)
    )


def downgrade() -> None:
    op.drop_table("query_tag_links")
    op.drop_table("query_tags")
