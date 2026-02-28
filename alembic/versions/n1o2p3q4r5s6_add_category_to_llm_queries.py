"""add category column to llm_queries

Revision ID: n1o2p3q4r5s6
Revises: m1n2o3p4q5r6
Create Date: 2026-02-21 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "n1o2p3q4r5s6"
down_revision = "m1n2o3p4q5r6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("llm_queries", sa.Column("category", sa.String(255), nullable=True))
    op.create_index("ix_llm_queries_category", "llm_queries", ["category"])


def downgrade() -> None:
    op.drop_index("ix_llm_queries_category", table_name="llm_queries")
    op.drop_column("llm_queries", "category")
