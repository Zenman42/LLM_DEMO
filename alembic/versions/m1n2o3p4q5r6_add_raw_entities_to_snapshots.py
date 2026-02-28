"""add raw_entities column to llm_snapshots

Revision ID: m1n2o3p4q5r6
Revises: l1m2n3o4p5q6
Create Date: 2026-02-20 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "m1n2o3p4q5r6"
down_revision = "l1m2n3o4p5q6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("llm_snapshots", sa.Column("raw_entities", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("llm_snapshots", "raw_entities")
