"""Add gigachat_api_key to tenants.

Revision ID: s1t2u3v4w5x6
Revises: r1s2t3u4v5w6
Create Date: 2025-02-22 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "s1t2u3v4w5x6"
down_revision = "r1s2t3u4v5w6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("gigachat_api_key", sa.LargeBinary(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tenants", "gigachat_api_key")
