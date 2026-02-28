"""Add llm_explanation to discovered_entities.

Revision ID: r1s2t3u4v5w6
Revises: q1r2s3t4u5v6
Create Date: 2025-01-21 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "r1s2t3u4v5w6"
down_revision = "q1r2s3t4u5v6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "discovered_entities",
        sa.Column("llm_explanation", sa.String(1000), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("discovered_entities", "llm_explanation")
