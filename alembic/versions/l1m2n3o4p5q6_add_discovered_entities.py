"""add discovered_entities table

Revision ID: l1m2n3o4p5q6
Revises: k1l2m3n4o5p6
Create Date: 2026-02-19 18:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision = "l1m2n3o4p5q6"
down_revision = "k1l2m3n4o5p6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "discovered_entities",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("entity_name", sa.String(255), nullable=False),
        sa.Column("mention_count", sa.Integer(), server_default="1", nullable=False),
        sa.Column("first_seen", sa.Date(), nullable=False),
        sa.Column("last_seen", sa.Date(), nullable=False),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("verified_by", sa.String(20), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("project_id", "entity_name", name="uq_project_entity"),
    )
    op.create_index("ix_discovered_entity_project", "discovered_entities", ["project_id"])
    op.create_index("ix_discovered_entity_tenant", "discovered_entities", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("ix_discovered_entity_tenant", table_name="discovered_entities")
    op.drop_index("ix_discovered_entity_project", table_name="discovered_entities")
    op.drop_table("discovered_entities")
