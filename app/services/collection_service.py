"""Collection orchestrator — runs all collectors for a project or tenant."""

import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.base import CollectionResult
from app.collectors.google_search_console import GoogleSearchConsoleCollector
from app.collectors.justmagic import JustMagicCollector
from app.collectors.yandex_webmaster import YandexWebmasterCollector
from app.core.encryption import decrypt_value
from app.models.project import Project
from app.models.tenant import Tenant

logger = logging.getLogger(__name__)


@dataclass
class ProjectCollectionResult:
    project_id: int
    project_name: str
    domain: str
    justmagic: CollectionResult | None = None
    ywm: CollectionResult | None = None
    gsc: CollectionResult | None = None
    errors: list[str] = field(default_factory=list)


async def collect_project(
    db: AsyncSession,
    tenant_id: uuid.UUID,
    project_id: int,
) -> ProjectCollectionResult:
    """Run all configured collectors for a single project."""

    # Load tenant
    tenant = await db.get(Tenant, tenant_id)
    if not tenant:
        raise ValueError(f"Tenant {tenant_id} not found")

    # Load project
    stmt = select(Project).where(
        Project.id == project_id,
        Project.tenant_id == tenant_id,
        Project.is_active.is_(True),
    )
    row = await db.execute(stmt)
    project = row.scalar_one_or_none()
    if not project:
        raise ValueError(f"Project {project_id} not found or not active")

    result = ProjectCollectionResult(
        project_id=project.id,
        project_name=project.name,
        domain=project.domain or "",
    )

    # --- JustMagic ---
    jm_key = decrypt_value(tenant.justmagic_api_key) if tenant.justmagic_api_key else ""
    if jm_key and project.domain:
        try:
            collector = JustMagicCollector(api_key=jm_key, tenant_id=tenant_id)
            result.justmagic = await collector.collect(
                db=db,
                project_id=project.id,
                domain=project.domain,
                search_engine=project.search_engine,
                region_yandex=project.region_yandex,
                region_google=project.region_google,
            )
        except Exception as e:
            logger.error("JM collection failed for project %d: %s", project.id, e)
            result.errors.append(f"JustMagic: {e}")
    else:
        result.justmagic = CollectionResult(skipped=True, skip_reason="No JM API key or domain")

    # --- Yandex Webmaster ---
    ywm_token = decrypt_value(tenant.ywm_token) if tenant.ywm_token else ""
    if ywm_token and project.ywm_host_id:
        try:
            collector = YandexWebmasterCollector(ywm_token=ywm_token, tenant_id=tenant_id)
            result.ywm = await collector.collect(
                db=db,
                project_id=project.id,
                ywm_host_id=project.ywm_host_id,
            )
        except Exception as e:
            logger.error("YWM collection failed for project %d: %s", project.id, e)
            result.errors.append(f"YWM: {e}")
    else:
        result.ywm = CollectionResult(skipped=True, skip_reason="No YWM credentials or host_id")

    # --- Google Search Console ---
    gsc_json = decrypt_value(tenant.gsc_credentials_json) if tenant.gsc_credentials_json else ""
    if gsc_json and project.gsc_site_url:
        try:
            collector = GoogleSearchConsoleCollector(credentials_json=gsc_json, tenant_id=tenant_id)
            result.gsc = await collector.collect(
                db=db,
                project_id=project.id,
                gsc_site_url=project.gsc_site_url,
            )
        except Exception as e:
            logger.error("GSC collection failed for project %d: %s", project.id, e)
            result.errors.append(f"GSC: {e}")
    else:
        result.gsc = CollectionResult(skipped=True, skip_reason="No GSC credentials or site_url")

    return result


async def collect_all_projects(
    db: AsyncSession,
    tenant_id: uuid.UUID,
) -> list[ProjectCollectionResult]:
    """Run collection for all active projects of a tenant."""
    stmt = select(Project.id).where(
        Project.tenant_id == tenant_id,
        Project.is_active.is_(True),
    )
    rows = await db.execute(stmt)
    project_ids = rows.scalars().all()

    results = []
    for pid in project_ids:
        try:
            r = await collect_project(db, tenant_id, pid)
            results.append(r)
        except Exception as e:
            logger.error("Collection failed for project %d: %s", pid, e)
            results.append(
                ProjectCollectionResult(
                    project_id=pid,
                    project_name="?",
                    domain="?",
                    errors=[str(e)],
                )
            )

    return results
