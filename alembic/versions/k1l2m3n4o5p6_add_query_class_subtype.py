"""add query_class and query_subtype to llm_queries

Revision ID: k1l2m3n4o5p6
Revises: j1k2l3m4n5o6
Create Date: 2026-02-19 12:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "k1l2m3n4o5p6"
down_revision = "j1k2l3m4n5o6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("llm_queries", sa.Column("query_class", sa.String(20), nullable=True))
    op.add_column("llm_queries", sa.Column("query_subtype", sa.String(30), nullable=True))
    op.create_index("ix_llm_queries_query_class", "llm_queries", ["query_class"])
    op.create_index("ix_llm_queries_query_subtype", "llm_queries", ["query_subtype"])


def downgrade() -> None:
    op.drop_index("ix_llm_queries_query_subtype", table_name="llm_queries")
    op.drop_index("ix_llm_queries_query_class", table_name="llm_queries")
    op.drop_column("llm_queries", "query_subtype")
    op.drop_column("llm_queries", "query_class")
