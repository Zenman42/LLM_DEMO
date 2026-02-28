"""Tests for JustMagic collector with mocked HTTP."""

import gzip

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.justmagic import JustMagicCollector
from app.models.keyword import Keyword
from app.models.project import Project
from app.models.serp import SerpSnapshot
from app.models.tenant import Tenant


def _make_jm_tsv(keywords_positions: dict[str, list[tuple[int, str]]]) -> bytes:
    """Build a gzipped TSV mimicking JustMagic par_ps output."""
    lines = []
    # Header section (ignored by parser)
    lines.append("JM_PMETA\tОбщая")
    lines.append("keyword\tposition\turl")
    for kw, entries in keywords_positions.items():
        for pos, url in entries:
            lines.append(f"{kw}\t{pos}\t{url}")
    # Sources section (parsed by collector)
    lines.append("JM_PMETA\tИсточники")
    lines.append("JM_PMETA\tskip")
    lines.append("JM_PMETA\tИсходники")
    for kw, entries in keywords_positions.items():
        for pos, url in entries:
            lines.append(f"{kw}\t{pos}\t{url}")
    tsv = "\n".join(lines)
    return gzip.compress(tsv.encode("utf-8"))


@pytest.fixture
async def project_with_keywords(db: AsyncSession):
    """Create tenant, project, and keywords for testing."""
    tenant = Tenant(name="Test Corp", slug="jm-test")
    db.add(tenant)
    await db.flush()

    project = Project(
        tenant_id=tenant.id,
        name="Test Site",
        domain="example.com",
        search_engine="google",
    )
    db.add(project)
    await db.flush()

    kw1 = Keyword(tenant_id=tenant.id, project_id=project.id, keyword="buy laptop")
    kw2 = Keyword(tenant_id=tenant.id, project_id=project.id, keyword="best phones")
    db.add_all([kw1, kw2])
    await db.commit()
    return tenant, project, [kw1, kw2]


@pytest.mark.asyncio
async def test_parse_result():
    """Test TSV parsing logic."""
    serp_data = {
        "buy laptop": [
            (1, "https://example.com/laptops"),
            (2, "https://other.com/laptops"),
            (3, "https://third.com"),
        ],
        "best phones": [
            (1, "https://other.com/phones"),
            (2, "https://third.com/phones"),
        ],
    }
    rows_bytes = _make_jm_tsv(serp_data)
    raw = gzip.decompress(rows_bytes).decode("utf-8")
    data_rows = [line.split("\t") for line in raw.split("\n") if line.strip()]

    positions, serp = JustMagicCollector._parse_result(data_rows, "example.com")

    assert positions["buy laptop"]["position"] == 1
    assert positions["buy laptop"]["found_url"] == "https://example.com/laptops"
    assert positions["best phones"]["position"] is None  # not found for example.com
    assert len(serp["buy laptop"]) == 3
    assert len(serp["best phones"]) == 2


@pytest.mark.asyncio
async def test_collect_saves_snapshots(db: AsyncSession, project_with_keywords, monkeypatch):
    """Test that collect() saves SERP snapshots to PG."""
    tenant, project, keywords = project_with_keywords

    serp_data = {
        "buy laptop": [
            (1, "https://example.com/laptops"),
            (2, "https://other.com/laptops"),
        ],
        "best phones": [
            (1, "https://example.com/phones"),
            (2, "https://other.com"),
        ],
    }
    compressed_tsv = _make_jm_tsv(serp_data)

    call_count = {"create": 0, "poll": 0, "download": 0}

    async def mock_create_task(self, **kwargs):
        call_count["create"] += 1
        return 12345

    async def mock_wait_task(self, tid):
        call_count["poll"] += 1
        return {"state": "fin"}

    async def mock_download(self, tid):
        call_count["download"] += 1
        raw = gzip.decompress(compressed_tsv).decode("utf-8")
        return [line.split("\t") for line in raw.split("\n") if line.strip()]

    def mock_save_ch(self, *args, **kwargs):
        pass  # Skip CH in tests

    monkeypatch.setattr(JustMagicCollector, "_create_task", mock_create_task)
    monkeypatch.setattr(JustMagicCollector, "_wait_task", mock_wait_task)
    monkeypatch.setattr(JustMagicCollector, "_download", mock_download)
    monkeypatch.setattr(JustMagicCollector, "_save_to_clickhouse", mock_save_ch)

    collector = JustMagicCollector(api_key="test-key", tenant_id=tenant.id)
    result = await collector.collect(
        db=db,
        project_id=project.id,
        domain="example.com",
        search_engine="google",
    )

    assert result.collected == 2  # 2 keywords
    assert not result.errors
    assert call_count["create"] == 1
    assert call_count["download"] == 1

    # Verify snapshots in DB
    stmt = select(SerpSnapshot).where(SerpSnapshot.tenant_id == tenant.id)
    rows = await db.execute(stmt)
    snapshots = rows.scalars().all()
    assert len(snapshots) == 2

    snap_map = {s.keyword_id: s for s in snapshots}
    kw1 = keywords[0]  # "buy laptop"
    assert snap_map[kw1.id].position == 1
    assert "example.com" in snap_map[kw1.id].found_url
