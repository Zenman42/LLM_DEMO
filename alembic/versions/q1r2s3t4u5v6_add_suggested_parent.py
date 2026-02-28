"""Add suggested_parent to discovered_entities.

Revision ID: q1r2s3t4u5v6
Revises: p1q2r3s4t5u6
Create Date: 2025-01-20 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "q1r2s3t4u5v6"
down_revision = "p1q2r3s4t5u6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "discovered_entities",
        sa.Column("suggested_parent", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("discovered_entities", "suggested_parent")
