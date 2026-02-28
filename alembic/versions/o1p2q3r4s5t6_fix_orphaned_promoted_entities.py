"""fix orphaned promoted entities — reset to confirmed

Entities that were removed from project.competitors but still have
status='promoted' in discovered_entities should be reset to 'confirmed'
so they reappear in the discovered list.

Revision ID: o1p2q3r4s5t6
Revises: n1o2p3q4r5s6
Create Date: 2026-02-21 14:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "o1p2q3r4s5t6"
down_revision = "n1o2p3q4r5s6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Reset orphaned promoted entities: status='promoted' but entity_name
    # is NOT in project.competitors JSONB array.
    op.execute("""
        UPDATE discovered_entities de
        SET status = 'confirmed'
        FROM projects p
        WHERE de.project_id = p.id
          AND de.status = 'promoted'
          AND NOT (p.competitors @> to_jsonb(de.entity_name))
    """)


def downgrade() -> None:
    # Cannot reliably reverse — the old orphaned state was a bug
    pass
