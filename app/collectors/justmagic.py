"""JustMagic par_ps SERP collector — async version."""

import asyncio
import gzip
import logging
import uuid
from datetime import date, datetime, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.base import BaseCollector, CollectionResult
from app.core.config import settings
from app.db.clickhouse import get_ch_client
from app.models.keyword import Keyword
from app.models.serp import SerpSnapshot

logger = logging.getLogger(__name__)

POLL_INTERVAL = 30  # seconds (API minimum)
MAX_POLL_ATTEMPTS = 40  # 40 × 30s = 20 min max wait


class JustMagicCollector(BaseCollector):
    """Collect SERP positions from JustMagic par_ps API."""

    def __init__(self, api_key: str, tenant_id: uuid.UUID):
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.api_url = settings.justmagic_api_url

    async def collect(
        self,
        *,
        db: AsyncSession,
        project_id: int,
        domain: str,
        search_engine: str,
        region_yandex: int | None = None,
        region_google: str | None = None,
    ) -> CollectionResult:
        result = CollectionResult()

        # Load keywords
        stmt = select(Keyword).where(
            Keyword.project_id == project_id,
            Keyword.tenant_id == self.tenant_id,
            Keyword.is_active.is_(True),
        )
        rows = await db.execute(stmt)
        keywords = rows.scalars().all()

        if not keywords:
            result.skipped = True
            result.skip_reason = "No keywords"
            return result

        query_list = [kw.keyword for kw in keywords]
        kw_map = {kw.keyword.lower(): kw for kw in keywords}

        engines = []
        if search_engine in ("google", "both"):
            engines.append("google")
        if search_engine in ("yandex", "both"):
            engines.append("yandex")

        today = date.today()

        for engine in engines:
            try:
                tid = await self._create_task(
                    queries=query_list,
                    search_engine=engine,
                    domain=domain,
                    region_yandex=region_yandex if engine == "yandex" else None,
                    region_google=region_google if engine == "google" else None,
                )
                logger.info("JM task %d created for %s/%s (%d keywords)", tid, domain, engine, len(query_list))

                await self._wait_task(tid)
                logger.info("JM task %d finished, downloading...", tid)

                data_rows = await self._download(tid)
                positions, serp_data = self._parse_result(data_rows, domain)

                # Write to PostgreSQL (serp_snapshots) via upsert
                saved = await self._save_snapshots(db, positions, serp_data, kw_map, today, engine)
                result.collected += saved

                # Write to ClickHouse (serp_details) — best-effort
                self._save_to_clickhouse(serp_data, kw_map, today, engine)

            except Exception as e:
                logger.error("JM %s error for %s: %s", engine, domain, e)
                result.errors.append(f"JustMagic/{engine}: {e}")

        return result

    async def _create_task(
        self,
        queries: list[str],
        search_engine: str,
        domain: str,
        region_yandex: int | None = None,
        region_google: str | None = None,
    ) -> int:
        payload = {
            "action": "put_task",
            "apikey": self.api_key,
            "task": "par_ps",
            "search_engine": search_engine,
            "data": "\n".join(queries),
            "src_serp": "ver",
        }
        if domain:
            payload["par_ps_domain"] = domain
        if search_engine == "yandex" and region_yandex:
            payload["region"] = str(region_yandex)
        if search_engine == "google" and region_google:
            payload["google_lr"] = region_google

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.api_url, data=payload)
            resp.raise_for_status()
            data = resp.json()

        if data.get("err") != 0:
            raise RuntimeError(f"JM create task error: {data.get('errtxt', data)}")
        return int(data["tid"])

    async def _wait_task(self, tid: int) -> dict:
        for attempt in range(MAX_POLL_ATTEMPTS):
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    self.api_url,
                    data={
                        "action": "get_task",
                        "apikey": self.api_key,
                        "tid": tid,
                        "mode": "info",
                    },
                )
                resp.raise_for_status()
                info = resp.json()

            result = info.get("result", {})
            state = result.get("state", "")
            logger.info("JM poll tid=%d: state=%s (%d/%d)", tid, state, attempt + 1, MAX_POLL_ATTEMPTS)

            if state == "fin":
                fin_err = result.get("fin_err")
                if fin_err:
                    raise RuntimeError(f"JM task {tid} finished with error: {fin_err}")
                return result
            if state == "err":
                raise RuntimeError(f"JM task {tid} failed: {result}")

            await asyncio.sleep(POLL_INTERVAL)

        raise TimeoutError(f"JM task {tid} did not finish in {MAX_POLL_ATTEMPTS * POLL_INTERVAL}s")

    async def _download(self, tid: int) -> list[list[str]]:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                self.api_url,
                data={
                    "action": "get_task",
                    "apikey": self.api_key,
                    "tid": tid,
                    "mode": "csv",
                    "system": "unix",
                },
            )
            resp.raise_for_status()

        try:
            raw = gzip.decompress(resp.content)
        except gzip.BadGzipFile:
            logger.error("JM download: not gzip. Content-Type: %s", resp.headers.get("Content-Type"))
            raise RuntimeError(f"JM download: unexpected response format for tid={tid}")

        text = raw.decode("utf-8")
        rows = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                rows.append(line.split("\t"))

        logger.info("JM download tid=%d: %d rows parsed", tid, len(rows))
        return rows

    @staticmethod
    def _parse_result(data_rows: list[list[str]], domain: str) -> tuple[dict, dict]:
        """Parse JM TSV. Returns (positions, serp_details) dicts keyed by keyword_lower."""
        in_sources = False
        serp_details: dict[str, list[tuple[int, str]]] = {}

        for row in data_rows:
            if len(row) >= 2 and row[0] == "JM_PMETA":
                in_sources = row[1] == "Исходники"
                continue
            if not in_sources or len(row) < 3:
                continue

            kw = row[0].strip().lower()
            try:
                pos = int(row[1])
            except ValueError:
                continue
            url = row[2].strip()
            serp_details.setdefault(kw, []).append((pos, url))

        domain_lower = domain.lower().rstrip("/")
        positions: dict[str, dict] = {}
        for kw, entries in serp_details.items():
            found = False
            for pos, url in sorted(entries, key=lambda x: x[0]):
                if domain_lower in url.lower():
                    positions[kw] = {"position": pos, "found_url": url}
                    found = True
                    break
            if not found:
                positions[kw] = {"position": None, "found_url": ""}

        return positions, serp_details

    async def _save_snapshots(
        self,
        db: AsyncSession,
        positions: dict,
        serp_data: dict,
        kw_map: dict[str, Keyword],
        today: date,
        engine: str,
    ) -> int:
        saved = 0
        for kw_lower, pos_info in positions.items():
            kw_obj = kw_map.get(kw_lower)
            if not kw_obj:
                continue

            # Get previous position for tracking
            prev_stmt = (
                select(SerpSnapshot.position)
                .where(
                    SerpSnapshot.keyword_id == kw_obj.id,
                    SerpSnapshot.search_engine == engine,
                    SerpSnapshot.date < today,
                )
                .order_by(SerpSnapshot.date.desc())
                .limit(1)
            )
            prev_row = await db.execute(prev_stmt)
            prev_pos = prev_row.scalar_one_or_none()

            stmt = (
                pg_insert(SerpSnapshot)
                .values(
                    tenant_id=self.tenant_id,
                    keyword_id=kw_obj.id,
                    date=today,
                    search_engine=engine,
                    position=pos_info["position"],
                    found_url=pos_info["found_url"],
                    previous_position=prev_pos,
                    collected_at=datetime.now(timezone.utc),
                )
                .on_conflict_do_update(
                    constraint="uq_serp_snapshot",
                    set_={
                        "position": pos_info["position"],
                        "found_url": pos_info["found_url"],
                        "previous_position": prev_pos,
                        "collected_at": datetime.now(timezone.utc),
                    },
                )
            )
            await db.execute(stmt)
            saved += 1

        await db.commit()
        logger.info("JM %s: saved %d snapshots to PG", engine, saved)
        return saved

    def _save_to_clickhouse(
        self,
        serp_data: dict,
        kw_map: dict[str, Keyword],
        today: date,
        engine: str,
    ) -> None:
        """Batch insert SERP top-50 into ClickHouse. Best-effort — logs errors."""
        try:
            client = get_ch_client()
        except Exception as e:
            logger.warning("CH unavailable, skipping serp_details write: %s", e)
            return

        rows = []
        for kw_lower, entries in serp_data.items():
            kw_obj = kw_map.get(kw_lower)
            if not kw_obj:
                continue
            for pos, url in entries:
                # Extract domain from URL
                try:
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    url_domain = parsed.netloc or ""
                except Exception:
                    url_domain = ""

                rows.append(
                    [
                        self.tenant_id,
                        kw_obj.id,
                        today,
                        engine,
                        pos,
                        url,
                        url_domain,
                        "",  # title — not available from par_ps
                        "",  # snippet — not available from par_ps
                    ]
                )

        if not rows:
            return

        try:
            client.insert(
                "serp_details",
                rows,
                column_names=[
                    "tenant_id",
                    "keyword_id",
                    "date",
                    "search_engine",
                    "position",
                    "url",
                    "domain",
                    "title",
                    "snippet",
                ],
            )
            logger.info("CH: inserted %d serp_details rows for %s", len(rows), engine)
        except Exception as e:
            logger.error("CH insert failed: %s", e)
