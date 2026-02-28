"""Dashboard service — optimized SQL queries for project overview stats.

Replaces N+1 query patterns with efficient JOINs and aggregates.
"""

import logging
from datetime import date, timedelta
from uuid import UUID

from sqlalchemy import Float, and_, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import ROLE_HIERARCHY
from app.models.keyword import Keyword
from app.models.llm_query import LlmQuery
from app.models.llm_snapshot import LlmSnapshot
from app.models.project import Project
from app.models.serp import SerpSnapshot
from app.models.user import User
from app.models.user_project import UserProject

logger = logging.getLogger(__name__)


async def get_dashboard_data(db: AsyncSession, tenant_id: UUID, user: User | None = None) -> list[dict]:
    """Get dashboard overview for tenant projects visible to the user.

    For each project returns:
    - project info (id, name, domain, search_engine, is_active)
    - keyword_count
    - avg_position (today, across all keywords for primary engine)
    - trend (7-day: avg_pos_today vs avg_pos_7_days_ago; positive = improved)
    - last_collected date
    """
    today = date.today()
    week_ago = today - timedelta(days=7)

    # Get all projects with keyword counts
    stmt = (
        select(
            Project,
            func.count(func.distinct(Keyword.id)).label("keyword_count"),
        )
        .outerjoin(
            Keyword,
            and_(
                Keyword.project_id == Project.id,
                Keyword.is_active == True,  # noqa: E712
            ),
        )
        .where(Project.tenant_id == tenant_id, Project.is_active == True)  # noqa: E712
    )

    # Non-admin users: filter to assigned projects only
    if user is not None and ROLE_HIERARCHY.get(user.role, 0) < ROLE_HIERARCHY["admin"]:
        stmt = stmt.join(
            UserProject,
            and_(UserProject.project_id == Project.id, UserProject.user_id == user.id),
        )

    stmt = stmt.group_by(Project.id).order_by(Project.created_at.desc())
    projects_result = await db.execute(stmt)
    project_rows = projects_result.all()

    if not project_rows:
        return []

    project_ids = [r.Project.id for r in project_rows]

    # Aggregate SERP stats across all projects in a single query
    # We compute per-project: avg_position_today, avg_position_week_ago, last_date
    stats_stmt = (
        select(
            Keyword.project_id,
            # Average position today (only position > 0 means "found")
            func.avg(
                case(
                    (and_(SerpSnapshot.date == today, SerpSnapshot.position > 0), cast(SerpSnapshot.position, Float)),
                    else_=None,
                )
            ).label("avg_pos_today"),
            # Average position 7 days ago
            func.avg(
                case(
                    (
                        and_(SerpSnapshot.date == week_ago, SerpSnapshot.position > 0),
                        cast(SerpSnapshot.position, Float),
                    ),
                    else_=None,
                )
            ).label("avg_pos_week_ago"),
            # Last collection date
            func.max(SerpSnapshot.date).label("last_date"),
        )
        .join(Keyword, Keyword.id == SerpSnapshot.keyword_id)
        .where(
            Keyword.project_id.in_(project_ids),
            SerpSnapshot.tenant_id == tenant_id,
        )
        .group_by(Keyword.project_id)
    )

    stats_result = await db.execute(stats_stmt)
    stats_map: dict[int, dict] = {}
    for r in stats_result.all():
        avg_today = round(r.avg_pos_today, 1) if r.avg_pos_today else None
        avg_week = round(r.avg_pos_week_ago, 1) if r.avg_pos_week_ago else None

        trend = None
        if avg_today is not None and avg_week is not None:
            trend = round(avg_week - avg_today, 1)  # positive = improved (went up in ranking)

        stats_map[r.project_id] = {
            "avg_position": avg_today,
            "trend": trend,
            "last_collected": r.last_date.isoformat() if r.last_date else None,
        }

    # LLM stats: query count, mention rate (last 30 days), last collected
    llm_project_ids = [r.Project.id for r in project_rows if r.Project.track_llm]
    llm_stats_map: dict[int, dict] = {}

    if llm_project_ids:
        month_ago = today - timedelta(days=30)

        # LLM query count per project
        llm_count_stmt = (
            select(
                LlmQuery.project_id,
                func.count(LlmQuery.id).label("query_count"),
            )
            .where(
                LlmQuery.project_id.in_(llm_project_ids),
                LlmQuery.tenant_id == tenant_id,
                LlmQuery.is_active == True,  # noqa: E712
            )
            .group_by(LlmQuery.project_id)
        )
        llm_count_result = await db.execute(llm_count_stmt)
        for r in llm_count_result.all():
            llm_stats_map[r.project_id] = {"llm_query_count": r.query_count}

        # LLM mention rate + last collected per project (last 30 days)
        llm_snap_stmt = (
            select(
                LlmQuery.project_id,
                func.count(LlmSnapshot.id).label("total_checks"),
                func.sum(case((LlmSnapshot.brand_mentioned == True, 1), else_=0)).label("mentioned"),  # noqa: E712
                func.max(LlmSnapshot.date).label("last_llm_date"),
            )
            .join(LlmQuery, LlmQuery.id == LlmSnapshot.llm_query_id)
            .where(
                LlmQuery.project_id.in_(llm_project_ids),
                LlmSnapshot.tenant_id == tenant_id,
                LlmSnapshot.date >= month_ago,
            )
            .group_by(LlmQuery.project_id)
        )
        llm_snap_result = await db.execute(llm_snap_stmt)
        for r in llm_snap_result.all():
            entry = llm_stats_map.get(r.project_id, {})
            mention_rate = r.mentioned / r.total_checks if r.total_checks > 0 else 0.0
            entry["mention_rate"] = round(mention_rate, 4)
            entry["total_checks"] = r.total_checks
            entry["last_llm_collected"] = r.last_llm_date.isoformat() if r.last_llm_date else None
            llm_stats_map[r.project_id] = entry

    # Build response
    dashboard = []
    for row in project_rows:
        project = row.Project
        stats = stats_map.get(project.id, {})
        llm_stats = llm_stats_map.get(project.id, {})
        dashboard.append(
            {
                "id": project.id,
                "name": project.name,
                "domain": project.domain,
                "search_engine": project.search_engine,
                "is_active": project.is_active,
                "keyword_count": row.keyword_count,
                "avg_position": stats.get("avg_position"),
                "trend": stats.get("trend"),
                "last_collected": stats.get("last_collected"),
                # LLM fields
                "track_llm": project.track_llm,
                "brand_name": project.brand_name,
                "llm_query_count": llm_stats.get("llm_query_count", 0),
                "mention_rate": llm_stats.get("mention_rate"),
                "total_llm_checks": llm_stats.get("total_checks", 0),
                "last_llm_collected": llm_stats.get("last_llm_collected"),
            }
        )

    return dashboard
