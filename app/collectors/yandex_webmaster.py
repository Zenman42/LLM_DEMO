"""Yandex Webmaster search queries collector — async version."""

import logging
import uuid
from datetime import date, datetime, timedelta, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.base import BaseCollector, CollectionResult
from app.core.config import settings
from app.models.keyword import Keyword
from app.models.webmaster import WebmasterData

logger = logging.getLogger(__name__)


class YandexWebmasterCollector(BaseCollector):
    """Collect search query stats from Yandex Webmaster API."""

    def __init__(self, ywm_token: str, tenant_id: uuid.UUID):
        self.ywm_token = ywm_token
        self.tenant_id = tenant_id
        self.api_url = settings.ywm_api_url
        self.user_id = settings.ywm_user_id

    async def collect(
        self,
        *,
        db: AsyncSession,
        project_id: int,
        ywm_host_id: str,
    ) -> CollectionResult:
        result = CollectionResult()

        if not self.ywm_token or not self.user_id:
            result.skipped = True
            result.skip_reason = "No YWM credentials"
            return result
        if not ywm_host_id:
            result.skipped = True
            result.skip_reason = "No ywm_host_id for project"
            return result

        # Load keywords
        stmt = select(Keyword).where(
            Keyword.project_id == project_id,
            Keyword.tenant_id == self.tenant_id,
            Keyword.is_active.is_(True),
        )
        rows = await db.execute(stmt)
        keywords = rows.scalars().all()
        kw_map = {kw.keyword.lower(): kw for kw in keywords}

        if not kw_map:
            result.skipped = True
            result.skip_reason = "No keywords"
            return result

        today = date.today()
        date_to = today - timedelta(days=1)
        date_from = today - timedelta(days=7)

        headers = {
            "Authorization": f"OAuth {self.ywm_token}",
            "Content-Type": "application/json",
        }

        offset = 0
        limit = 500

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                url = f"{self.api_url}/user/{self.user_id}/hosts/{ywm_host_id}/search-queries/popular"
                params = {
                    "order_by": "TOTAL_CLICKS",
                    "query_indicator": "TOTAL_SHOWS,TOTAL_CLICKS,AVG_SHOW_POSITION,AVG_CLICK_POSITION",
                    "date_from": date_from.isoformat(),
                    "date_to": date_to.isoformat(),
                    "limit": limit,
                    "offset": offset,
                }

                try:
                    resp = await client.get(url, headers=headers, params=params)
                    if resp.status_code == 401:
                        result.errors.append("YWM authentication failed (401)")
                        break
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    result.errors.append(f"YWM API error: {e}")
                    break

                queries = data.get("queries", [])
                if not queries:
                    break

                for q in queries:
                    query_text = q.get("query_text", "").lower()
                    kw_obj = kw_map.get(query_text)
                    if not kw_obj:
                        continue

                    indicators = q.get("indicators", {})
                    impressions = indicators.get("TOTAL_SHOWS", 0)
                    clicks = indicators.get("TOTAL_CLICKS", 0)
                    avg_pos = indicators.get("AVG_SHOW_POSITION", 0.0)
                    ctr = (clicks / impressions * 100) if impressions > 0 else 0.0

                    stmt = (
                        pg_insert(WebmasterData)
                        .values(
                            tenant_id=self.tenant_id,
                            keyword_id=kw_obj.id,
                            date=date_to,
                            search_engine="yandex",
                            impressions=impressions,
                            clicks=clicks,
                            ctr=round(ctr, 2),
                            position=round(avg_pos, 1),
                            page_url=None,
                            collected_at=datetime.now(timezone.utc),
                        )
                        .on_conflict_do_update(
                            constraint="uq_webmaster_data",
                            set_={
                                "impressions": impressions,
                                "clicks": clicks,
                                "ctr": round(ctr, 2),
                                "position": round(avg_pos, 1),
                                "collected_at": datetime.now(timezone.utc),
                            },
                        )
                    )
                    await db.execute(stmt)
                    result.collected += 1

                offset += limit
                total_count = data.get("count", 0)
                if offset >= total_count:
                    break

        await db.commit()
        logger.info("YWM: saved %d records for project %d", result.collected, project_id)
        return result
