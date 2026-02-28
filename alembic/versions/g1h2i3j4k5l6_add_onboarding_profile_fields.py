"""add onboarding profile fields to projects

Revision ID: g1h2i3j4k5l6
Revises: f1a2b3c4d5e6
Create Date: 2026-02-18 16:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "g1h2i3j4k5l6"
down_revision: str | None = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Onboarding audit profile fields — all nullable for backward compatibility
    op.add_column("projects", sa.Column("brand_aliases", JSONB(), nullable=True))
    op.add_column("projects", sa.Column("competitors", JSONB(), nullable=True))
    op.add_column("projects", sa.Column("categories", JSONB(), nullable=True))
    op.add_column("projects", sa.Column("golden_facts", JSONB(), nullable=True))
    op.add_column("projects", sa.Column("personas", JSONB(), nullable=True))
    op.add_column("projects", sa.Column("market", sa.String(length=10), nullable=True))
    op.add_column("projects", sa.Column("geo", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "geo")
    op.drop_column("projects", "market")
    op.drop_column("projects", "personas")
    op.drop_column("projects", "golden_facts")
    op.drop_column("projects", "categories")
    op.drop_column("projects", "competitors")
    op.drop_column("projects", "brand_aliases")
