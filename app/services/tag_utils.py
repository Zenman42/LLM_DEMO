"""Shared utilities for query tag management (dual-write helpers)."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.query_tag import QueryTag, QueryTagLink


async def ensure_tag(db: AsyncSession, project_id: int, tenant_id: UUID, tag_name: str, tag_type: str) -> int:
    """Get or create a tag and return its ID."""
    result = await db.execute(
        select(QueryTag.id).where(
            QueryTag.project_id == project_id,
            QueryTag.tag_name == tag_name,
            QueryTag.tag_type == tag_type,
        )
    )
    tag_id = result.scalar_one_or_none()
    if not tag_id:
        tag = QueryTag(project_id=project_id, tenant_id=tenant_id, tag_name=tag_name, tag_type=tag_type)
        db.add(tag)
        await db.flush()
        tag_id = tag.id
    return tag_id


async def link_tags_for_prompt(
    db: AsyncSession,
    query_id: int,
    project_id: int,
    tenant_id: UUID,
    query_class: str | None,
    category: str | None,
) -> None:
    """Create tag links for a saved prompt (dual write from legacy columns)."""
    tag_ids: list[int] = []
    if query_class:
        tag_ids.append(await ensure_tag(db, project_id, tenant_id, query_class, "class"))
    if category:
        tag_ids.append(await ensure_tag(db, project_id, tenant_id, category, "scenario"))
    for tid in tag_ids:
        await db.execute(
            pg_insert(QueryTagLink)
            .values(query_id=query_id, tag_id=tid)
            .on_conflict_do_nothing(constraint="uq_query_tag_link")
        )
