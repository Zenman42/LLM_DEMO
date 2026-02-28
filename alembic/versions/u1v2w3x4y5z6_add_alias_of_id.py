"""Add alias_of_id to discovered_entities for entity-to-entity alias relationships.

Migrates existing string-based aliases from brand_sub_brands / competitor_sub_brands
JSONB into entity records with alias_of_id set.

Revision ID: u1v2w3x4y5z6
Revises: t1u2v3w4x5y6
Create Date: 2025-06-01
"""

import json
import sqlalchemy as sa
from alembic import op
from datetime import date

revision = "u1v2w3x4y5z6"
down_revision = "t1u2v3w4x5y6"


def upgrade():
    # 1. Add alias_of_id column
    op.add_column(
        "discovered_entities",
        sa.Column(
            "alias_of_id",
            sa.Integer,
            sa.ForeignKey("discovered_entities.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_discovered_entity_alias_of", "discovered_entities", ["alias_of_id"])

    # 2. Data migration: convert JSONB alias strings to alias_of_id relationships
    conn = op.get_bind()

    projects = conn.execute(
        sa.text("SELECT id, tenant_id, brand_sub_brands, competitor_sub_brands FROM projects")
    ).fetchall()

    today = date.today()

    for proj in projects:
        project_id = proj[0]
        tenant_id = proj[1]
        brand_subs = proj[2] or {}  # {sub_name: [alias1, alias2]}
        comp_subs = proj[3] or {}  # {comp: {sub_name: [alias1, alias2]}}

        # Collect all (sub_brand_name, alias_string) pairs
        alias_pairs = []
        for sb_name, aliases in brand_subs.items():
            if isinstance(aliases, list):
                for a in aliases:
                    if a and isinstance(a, str):
                        alias_pairs.append((sb_name, a))
            elif isinstance(aliases, dict):
                for a in aliases.get("aliases", []):
                    if a and isinstance(a, str):
                        alias_pairs.append((sb_name, a))

        for comp_name, subs in comp_subs.items():
            if not isinstance(subs, dict):
                continue
            for sb_name, aliases in subs.items():
                if isinstance(aliases, list):
                    for a in aliases:
                        if a and isinstance(a, str):
                            alias_pairs.append((sb_name, a))
                elif isinstance(aliases, dict):
                    for a in aliases.get("aliases", []):
                        if a and isinstance(a, str):
                            alias_pairs.append((sb_name, a))

        for target_name, alias_name in alias_pairs:
            # Find or create the target (sub-brand) entity
            target = conn.execute(
                sa.text("SELECT id FROM discovered_entities WHERE project_id = :pid AND entity_name = :name"),
                {"pid": project_id, "name": target_name},
            ).fetchone()

            if not target:
                conn.execute(
                    sa.text(
                        "INSERT INTO discovered_entities "
                        "(tenant_id, project_id, entity_name, mention_count, first_seen, last_seen, status) "
                        "VALUES (:tid, :pid, :name, 0, :today, :today, 'promoted') "
                        "ON CONFLICT (project_id, entity_name) DO NOTHING"
                    ),
                    {"tid": str(tenant_id), "pid": project_id, "name": target_name, "today": today},
                )
                target = conn.execute(
                    sa.text("SELECT id FROM discovered_entities WHERE project_id = :pid AND entity_name = :name"),
                    {"pid": project_id, "name": target_name},
                ).fetchone()

            if not target:
                continue

            target_id = target[0]

            # Find or create the alias entity
            alias_ent = conn.execute(
                sa.text("SELECT id FROM discovered_entities WHERE project_id = :pid AND entity_name = :name"),
                {"pid": project_id, "name": alias_name},
            ).fetchone()

            if not alias_ent:
                conn.execute(
                    sa.text(
                        "INSERT INTO discovered_entities "
                        "(tenant_id, project_id, entity_name, mention_count, first_seen, last_seen, status, alias_of_id) "
                        "VALUES (:tid, :pid, :name, 0, :today, :today, 'alias', :aof) "
                        "ON CONFLICT (project_id, entity_name) DO UPDATE SET alias_of_id = :aof, status = 'alias'"
                    ),
                    {"tid": str(tenant_id), "pid": project_id, "name": alias_name, "today": today, "aof": target_id},
                )
            else:
                conn.execute(
                    sa.text("UPDATE discovered_entities SET alias_of_id = :aof, status = 'alias' WHERE id = :eid"),
                    {"aof": target_id, "eid": alias_ent[0]},
                )

        # 3. Clear alias lists in JSONB (keep keys for sub-brand relationship)
        if brand_subs:
            cleaned = {k: [] for k in brand_subs}
            conn.execute(
                sa.text("UPDATE projects SET brand_sub_brands = CAST(:val AS jsonb) WHERE id = :pid"),
                {"val": json.dumps(cleaned), "pid": project_id},
            )
        if comp_subs:
            cleaned_comp = {}
            for comp, subs in comp_subs.items():
                if isinstance(subs, dict):
                    cleaned_comp[comp] = {k: [] for k in subs}
                else:
                    cleaned_comp[comp] = {}
            conn.execute(
                sa.text("UPDATE projects SET competitor_sub_brands = CAST(:val AS jsonb) WHERE id = :pid"),
                {"val": json.dumps(cleaned_comp), "pid": project_id},
            )


def downgrade():
    # Drop index and column
    op.drop_index("ix_discovered_entity_alias_of", table_name="discovered_entities")
    op.drop_column("discovered_entities", "alias_of_id")
