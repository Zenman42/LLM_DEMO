"""change brand_description from varchar(2000) to text

Revision ID: i1j2k3l4m5n6
Revises: h1i2j3k4l5m6
Create Date: 2026-02-18 22:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "i1j2k3l4m5n6"
down_revision: str | None = "h1i2j3k4l5m6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "projects",
        "brand_description",
        type_=sa.Text(),
        existing_type=sa.String(length=2000),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "projects",
        "brand_description",
        type_=sa.String(length=2000),
        existing_type=sa.Text(),
        existing_nullable=True,
    )
