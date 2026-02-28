"""add brand_description to projects

Revision ID: h1i2j3k4l5m6
Revises: 08488d089599
Create Date: 2026-02-18 18:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "h1i2j3k4l5m6"
down_revision: str | None = "08488d089599"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("brand_description", sa.String(length=2000), nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "brand_description")
