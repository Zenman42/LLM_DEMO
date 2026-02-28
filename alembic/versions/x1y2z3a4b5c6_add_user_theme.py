"""Add theme column to users table.

Per-user light/dark theme preference.

Revision ID: x1y2z3a4b5c6
Revises: w1x2y3z4a5b6
Create Date: 2026-02-27
"""

import sqlalchemy as sa
from alembic import op

revision = "x1y2z3a4b5c6"
down_revision = "w1x2y3z4a5b6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("theme", sa.String(10), server_default="dark", nullable=False))


def downgrade() -> None:
    op.drop_column("users", "theme")
