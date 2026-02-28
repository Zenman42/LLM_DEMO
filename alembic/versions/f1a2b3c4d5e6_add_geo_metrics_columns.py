"""add geo metrics columns to llm_snapshots

Revision ID: f1a2b3c4d5e6
Revises: b2c3d4e5f6a7
Create Date: 2026-02-18 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # GEO metrics columns for NSS, Top-1 Win Rate, Hallucination Rate
    # All nullable for backward compatibility with existing snapshots
    op.add_column("llm_snapshots", sa.Column("sentiment_label", sa.String(length=20), nullable=True))
    op.add_column("llm_snapshots", sa.Column("position_rank", sa.Integer(), nullable=True))
    op.add_column("llm_snapshots", sa.Column("structure_type", sa.String(length=30), nullable=True))
    op.add_column("llm_snapshots", sa.Column("is_hallucination", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_snapshots", "is_hallucination")
    op.drop_column("llm_snapshots", "structure_type")
    op.drop_column("llm_snapshots", "position_rank")
    op.drop_column("llm_snapshots", "sentiment_label")
