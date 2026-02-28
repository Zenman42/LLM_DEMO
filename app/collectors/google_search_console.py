"""Google Search Console collector — async version.

Uses httpx for HTTP calls to the GSC API with OAuth2 credentials stored per-tenant.
The tenant stores the full OAuth2 credentials JSON (encrypted) instead of file paths.
"""

import json
import logging
import uuid
from datetime import date, datetime, timedelta, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.base import BaseCollector, CollectionResult
from app.models.keyword import Keyword
from app.models.webmaster import WebmasterData

logger = logging.getLogger(__name__)

GSC_API_BASE = "https://www.googleapis.com/webmasters/v3"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


class GoogleSearchConsoleCollector(BaseCollector):
    """Collect search analytics from Google Search Console."""

    def __init__(self, credentials_json: str, tenant_id: uuid.UUID):
        """
        Args:
            credentials_json: JSON string with OAuth2 credentials
                (must contain access_token, refresh_token, client_id, client_secret)
            tenant_id: tenant UUID
        """
        self.tenant_id = tenant_id
        try:
            self.creds = json.loads(credentials_json)
        except (json.JSONDecodeError, TypeError):
            self.creds = {}

    async def collect(
        self,
        *,
        db: AsyncSession,
        project_id: int,
        gsc_site_url: str,
    ) -> CollectionResult:
        result = CollectionResult()

        if not gsc_site_url:
            result.skipped = True
            result.skip_reason = "No gsc_site_url for project"
            return result

        if not self.creds.get("access_token"):
            result.skipped = True
            result.skip_reason = "No GSC credentials"
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

        # Fetch all rows with date dimension
        raw_rows = await self._fetch_all_rows(gsc_site_url, date_from, date_to)
        if raw_rows is None:
            result.errors.append("GSC API fetch failed")
            return result

        if not raw_rows:
            result.skipped = True
            result.skip_reason = "No data from GSC"
            return result

        # Aggregate by (keyword_id, date) — pick best page per day
        agg: dict[tuple[int, str], list[dict]] = {}
        for row in raw_rows:
            keys = row.get("keys", [])
            if len(keys) < 3:
                continue
            row_date_str = keys[0]
            query_text = keys[1].lower()
            page_url = keys[2]

            kw_obj = kw_map.get(query_text)
            if not kw_obj:
                continue

            agg_key = (kw_obj.id, row_date_str)
            agg.setdefault(agg_key, []).append(
                {
                    "impressions": row.get("impressions", 0),
                    "clicks": row.get("clicks", 0),
                    "position": row.get("position", 0.0),
                    "ctr": row.get("ctr", 0.0),
                    "url": page_url,
                }
            )

        # Upsert into WebmasterData
        for (kw_id, date_str), pages in agg.items():
            row_date = date.fromisoformat(date_str)
            imp = sum(p["impressions"] for p in pages)
            clk = sum(p["clicks"] for p in pages)
            best = max(pages, key=lambda p: p["impressions"])
            avg_pos = round(best["position"], 1)
            ctr_val = round(best["ctr"] * 100, 2)  # GSC returns 0.0–1.0
            page = best["url"]

            stmt = (
                pg_insert(WebmasterData)
                .values(
                    tenant_id=self.tenant_id,
                    keyword_id=kw_id,
                    date=row_date,
                    search_engine="google",
                    impressions=imp,
                    clicks=clk,
                    ctr=ctr_val,
                    position=avg_pos,
                    page_url=page,
                    collected_at=datetime.now(timezone.utc),
                )
                .on_conflict_do_update(
                    constraint="uq_webmaster_data",
                    set_={
                        "impressions": imp,
                        "clicks": clk,
                        "ctr": ctr_val,
                        "position": avg_pos,
                        "page_url": page,
                        "collected_at": datetime.now(timezone.utc),
                    },
                )
            )
            await db.execute(stmt)
            result.collected += 1

        await db.commit()
        logger.info("GSC: saved %d day-records for project %d", result.collected, project_id)
        return result

    async def _fetch_all_rows(self, site_url: str, date_from: date, date_to: date) -> list[dict] | None:
        """Fetch all search analytics rows from GSC API, paginating as needed."""
        access_token = await self._ensure_token()
        if not access_token:
            return None

        headers = {"Authorization": f"Bearer {access_token}"}
        raw_rows = []
        start_row = 0
        row_limit = 5000

        async with httpx.AsyncClient(timeout=60) as client:
            while True:
                body = {
                    "startDate": date_from.isoformat(),
                    "endDate": date_to.isoformat(),
                    "dimensions": ["date", "query", "page"],
                    "rowLimit": row_limit,
                    "startRow": start_row,
                }

                try:
                    resp = await client.post(
                        f"{GSC_API_BASE}/sites/{site_url}/searchAnalytics/query",
                        headers=headers,
                        json=body,
                    )
                    if resp.status_code == 401:
                        # Try refreshing token once
                        access_token = await self._refresh_token()
                        if not access_token:
                            logger.error("GSC: token refresh failed")
                            return None
                        headers["Authorization"] = f"Bearer {access_token}"
                        resp = await client.post(
                            f"{GSC_API_BASE}/sites/{site_url}/searchAnalytics/query",
                            headers=headers,
                            json=body,
                        )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.error("GSC API error: %s", e)
                    return None

                rows = data.get("rows", [])
                if not rows:
                    break
                raw_rows.extend(rows)

                start_row += row_limit
                if len(rows) < row_limit:
                    break

        return raw_rows

    async def _ensure_token(self) -> str | None:
        """Get a valid access token, refreshing if needed."""
        token = self.creds.get("access_token")
        if not token:
            return await self._refresh_token()
        return token

    async def _refresh_token(self) -> str | None:
        """Refresh the OAuth2 access token using the refresh token."""
        refresh_token = self.creds.get("refresh_token")
        client_id = self.creds.get("client_id")
        client_secret = self.creds.get("client_secret")

        if not all([refresh_token, client_id, client_secret]):
            logger.error("GSC: missing refresh credentials")
            return None

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(
                    GOOGLE_TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                new_token = data.get("access_token")
                if new_token:
                    self.creds["access_token"] = new_token
                    return new_token
            except Exception as e:
                logger.error("GSC token refresh error: %s", e)
        return None
