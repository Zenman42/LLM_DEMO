"""Add user_projects table for per-project user access control.

Creates user_projects join table linking users to specific projects they can access.
Admin/owner users bypass this check (they see all projects).
Data migration: assigns all existing viewer/member users to all projects in their tenant.

Revision ID: w1x2y3z4a5b6
Revises: v1w2x3y4z5a6
Create Date: 2026-02-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "w1x2y3z4a5b6"
down_revision = "v1w2x3y4z5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_projects",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
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
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_user_projects_user_id", "user_projects", ["user_id"])
    op.create_index("ix_user_projects_project_id", "user_projects", ["project_id"])
    op.create_index("ix_user_projects_tenant_id", "user_projects", ["tenant_id"])
    op.create_unique_constraint(
        "uq_user_project",
        "user_projects",
        ["user_id", "project_id"],
    )

    # Data migration: assign all existing viewer/member users to ALL projects in their tenant
    conn = op.get_bind()
    conn.execute(
        sa.text("""
            INSERT INTO user_projects (user_id, project_id, tenant_id)
            SELECT u.id, p.id, u.tenant_id
            FROM users u
            CROSS JOIN projects p
            WHERE u.tenant_id = p.tenant_id
              AND u.role IN ('viewer', 'member')
              AND u.is_active = true
            ON CONFLICT (user_id, project_id) DO NOTHING
        """)
    )


def downgrade() -> None:
    op.drop_table("user_projects")
