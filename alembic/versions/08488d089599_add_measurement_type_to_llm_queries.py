"""add_measurement_type_to_llm_queries

Revision ID: 08488d089599
Revises: g1h2i3j4k5l6
Create Date: 2026-02-18 18:16:47.589223

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "08488d089599"
down_revision: Union[str, None] = "g1h2i3j4k5l6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_queries", sa.Column("measurement_type", sa.String(length=30), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_queries", "measurement_type")
