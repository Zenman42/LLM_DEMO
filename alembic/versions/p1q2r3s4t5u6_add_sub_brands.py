"""Add brand_sub_brands and competitor_sub_brands to projects

Revision ID: p1q2r3s4t5u6
Revises: o1p2q3r4s5t6
Create Date: 2025-01-20 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "p1q2r3s4t5u6"
down_revision = "o1p2q3r4s5t6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("brand_sub_brands", JSONB(), nullable=True, server_default="{}"),
    )
    op.add_column(
        "projects",
        sa.Column("competitor_sub_brands", JSONB(), nullable=True, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("projects", "competitor_sub_brands")
    op.drop_column("projects", "brand_sub_brands")
