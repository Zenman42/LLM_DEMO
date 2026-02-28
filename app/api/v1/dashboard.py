"""Dashboard API endpoint — project overview with stats."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_tenant_id, get_current_user
from app.db.postgres import get_db
from app.models.user import User
from app.services.dashboard_service import get_dashboard_data

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/")
async def dashboard(
    db: AsyncSession = Depends(get_db),
    tenant_id: UUID = Depends(get_current_tenant_id),
    user: User = Depends(get_current_user),
):
    """Get dashboard overview with all projects and their stats.

    Returns per project:
    - keyword_count, avg_position (today), trend (7-day), last_collected
    """
    data = await get_dashboard_data(db, tenant_id, user)
    return {"projects": data}
