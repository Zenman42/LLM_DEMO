"""Integration tests for BI Dashboard API endpoints."""

from datetime import date, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.llm_query import LlmQuery
from app.models.llm_snapshot import LlmSnapshot
from app.models.project import Project
from app.models.tenant import Tenant


@pytest.fixture
async def bi_project(db: AsyncSession, tenant_and_user):
    """Project with rich LLM data for BI dashboard testing."""
    tenant, user = tenant_and_user
    today = date.today()
    yesterday = today - timedelta(days=1)

    project = Project(
        tenant_id=tenant.id,
        name="BI Test Project",
        domain="mybrand.com",
        track_llm=True,
        brand_name="MyBrand",
        llm_providers=["chatgpt", "deepseek", "yandexgpt"],
    )
    db.add(project)
    await db.flush()

    # Query 1: comparison type
    q_comparison = LlmQuery(
        tenant_id=tenant.id,
        project_id=project.id,
        query_text="MyBrand vs CompetitorA comparison",
        query_type="comparison",
        category="Price comparison",
        target_brand="MyBrand",
        competitors=["CompetitorA", "CompetitorB"],
    )
    # Query 2: recommendation type
    q_recommendation = LlmQuery(
        tenant_id=tenant.id,
        project_id=project.id,
        query_text="Best tools for analytics",
        query_type="recommendation",
        category="Best alternatives",
        target_brand="MyBrand",
        competitors=["CompetitorA"],
    )
    # Query 3: brand_check type
    q_brand = LlmQuery(
        tenant_id=tenant.id,
        project_id=project.id,
        query_text="What is MyBrand?",
        query_type="brand_check",
        category="Brand awareness",
        target_brand="MyBrand",
    )
    db.add_all([q_comparison, q_recommendation, q_brand])
    await db.flush()

    snapshots = [
        # comparison + chatgpt: brand recommended
        LlmSnapshot(
            tenant_id=tenant.id,
            llm_query_id=q_comparison.id,
            date=today,
            llm_provider="chatgpt",
            llm_model="gpt-4o",
            brand_mentioned=True,
            mention_type="recommended",
            competitor_mentions={"CompetitorA": True, "CompetitorB": False},
            cited_urls=["https://review-site.com/comparison"],
            response_tokens=100,
            cost_usd=0.002,
        ),
        # comparison + chatgpt yesterday: brand mentioned (resilience)
        LlmSnapshot(
            tenant_id=tenant.id,
            llm_query_id=q_comparison.id,
            date=yesterday,
            llm_provider="chatgpt",
            llm_model="gpt-4o",
            brand_mentioned=True,
            mention_type="direct",
            competitor_mentions={"CompetitorA": True},
            cited_urls=["https://review-site.com/comparison"],
            response_tokens=90,
            cost_usd=0.002,
        ),
        # comparison + deepseek: brand not mentioned (competitor wins)
        LlmSnapshot(
            tenant_id=tenant.id,
            llm_query_id=q_comparison.id,
            date=today,
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            brand_mentioned=False,
            mention_type="none",
            competitor_mentions={"CompetitorA": True},
            cited_urls=["https://competitor-source.com/article"],
            response_tokens=80,
            cost_usd=0.001,
        ),
        # recommendation + chatgpt: brand recommended
        LlmSnapshot(
            tenant_id=tenant.id,
            llm_query_id=q_recommendation.id,
            date=today,
            llm_provider="chatgpt",
            llm_model="gpt-4o",
            brand_mentioned=True,
            mention_type="recommended",
            competitor_mentions={"CompetitorA": False},
            cited_urls=["https://mybrand.com/features"],
            response_tokens=120,
            cost_usd=0.003,
        ),
        # recommendation + deepseek: brand compared
        LlmSnapshot(
            tenant_id=tenant.id,
            llm_query_id=q_recommendation.id,
            date=today,
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            brand_mentioned=True,
            mention_type="compared",
            competitor_mentions={"CompetitorA": True},
            cited_urls=["https://independent-review.com"],
            response_tokens=100,
            cost_usd=0.001,
        ),
        # brand_check + yandexgpt: brand not mentioned at all
        LlmSnapshot(
            tenant_id=tenant.id,
            llm_query_id=q_brand.id,
            date=today,
            llm_provider="yandexgpt",
            llm_model="yandexgpt-lite",
            brand_mentioned=False,
            mention_type="none",
            competitor_mentions=None,
            cited_urls=None,
            response_tokens=50,
            cost_usd=0.0005,
        ),
    ]
    db.add_all(snapshots)
    await db.commit()

    return {"project": project, "tenant": tenant, "queries": [q_comparison, q_recommendation, q_brand]}


@pytest.fixture
async def empty_project(db: AsyncSession, tenant_and_user):
    """Project with no LLM snapshots."""
    tenant, user = tenant_and_user

    project = Project(
        tenant_id=tenant.id,
        name="Empty BI Project",
        domain="empty.com",
        track_llm=True,
        brand_name="EmptyBrand",
        llm_providers=["chatgpt"],
    )
    db.add(project)
    await db.commit()
    return {"project": project, "tenant": tenant}


# ===================================================================
# /llm/bi-dashboard/{project_id}
# ===================================================================


class TestBiDashboardEndpoint:
    @pytest.mark.asyncio
    async def test_happy_path(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/bi-dashboard/{pid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        # Top-level structure
        assert "global_metrics" in data
        assert "competitive_matrix" in data
        assert "heatmap" in data
        assert "citation_graph" in data
        assert "geo_action_plan" in data

    @pytest.mark.asyncio
    async def test_global_metrics(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/bi-dashboard/{pid}", headers=auth_headers)
        data = resp.json()["global_metrics"]

        assert 0 <= data["aivs"] <= 100
        assert 0 <= data["som"] <= 100
        assert 0 <= data["resilience_score"] <= 1
        assert data["total_responses"] == 6
        # 4 brand mentioned out of 6 (snapshots 1,2,4,5 have brand_mentioned=True)
        assert data["mention_rate"] == pytest.approx(4 / 6, abs=0.01)

    @pytest.mark.asyncio
    async def test_competitive_matrix(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/bi-dashboard/{pid}", headers=auth_headers)
        data = resp.json()["competitive_matrix"]

        assert data["target"]["name"] == "MyBrand"
        assert data["target"]["som"] > 0
        comp_names = {c["name"] for c in data["competitors"]}
        assert "CompetitorA" in comp_names

    @pytest.mark.asyncio
    async def test_empty_project(self, client: AsyncClient, auth_headers, empty_project):
        pid = empty_project["project"].id
        resp = await client.get(f"/api/v1/llm/bi-dashboard/{pid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        assert data["global_metrics"]["aivs"] == 0.0
        assert data["global_metrics"]["som"] == 0.0
        assert data["global_metrics"]["total_responses"] == 0
        assert data["heatmap"]["cells"] == []

    @pytest.mark.asyncio
    async def test_with_days_filter(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        # days=1 means date_from = today - 1 = yesterday, so yesterday is included
        # All 6 snapshots are within range (today + yesterday)
        resp = await client.get(f"/api/v1/llm/bi-dashboard/{pid}?days=1", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["global_metrics"]["total_responses"] == 6

        # days=365 also works
        resp2 = await client.get(f"/api/v1/llm/bi-dashboard/{pid}?days=365", headers=auth_headers)
        assert resp2.status_code == 200

    @pytest.mark.asyncio
    async def test_nonexistent_project(self, client: AsyncClient, auth_headers):
        resp = await client.get("/api/v1/llm/bi-dashboard/99999", headers=auth_headers)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, client: AsyncClient, auth_headers, bi_project, db: AsyncSession):
        """Other tenant's project returns 404."""
        other_tenant = Tenant(name="Other Org", slug="other-org")
        db.add(other_tenant)
        await db.flush()

        other_project = Project(
            tenant_id=other_tenant.id,
            name="Other Project",
            domain="other.com",
            track_llm=True,
            brand_name="OtherBrand",
        )
        db.add(other_project)
        await db.commit()

        resp = await client.get(f"/api/v1/llm/bi-dashboard/{other_project.id}", headers=auth_headers)
        assert resp.status_code == 404


# ===================================================================
# /llm/heatmap/{project_id}
# ===================================================================


class TestHeatmapEndpoint:
    @pytest.mark.asyncio
    async def test_happy_path(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/heatmap/{pid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        assert "cells" in data
        assert "scenarios" in data
        assert "vendors" in data
        assert len(data["cells"]) > 0

    @pytest.mark.asyncio
    async def test_scenarios_present(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/heatmap/{pid}", headers=auth_headers)
        data = resp.json()

        # We have Price comparison, Best alternatives, Brand awareness categories
        assert set(data["scenarios"]) == {"Price comparison", "Best alternatives", "Brand awareness"}

    @pytest.mark.asyncio
    async def test_vendors_present(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/heatmap/{pid}", headers=auth_headers)
        data = resp.json()

        assert "chatgpt" in data["vendors"]
        assert "deepseek" in data["vendors"]

    @pytest.mark.asyncio
    async def test_cell_structure(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/heatmap/{pid}", headers=auth_headers)
        data = resp.json()

        cell = data["cells"][0]
        assert "scenario" in cell
        assert "vendor" in cell
        assert "aivs" in cell
        assert "mention_count" in cell
        assert "total_count" in cell
        assert cell["zone"] in ("red", "yellow", "green")

    @pytest.mark.asyncio
    async def test_empty_project(self, client: AsyncClient, auth_headers, empty_project):
        pid = empty_project["project"].id
        resp = await client.get(f"/api/v1/llm/heatmap/{pid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["cells"] == []
        assert data["scenarios"] == []
        assert data["vendors"] == []

    @pytest.mark.asyncio
    async def test_zone_values(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/heatmap/{pid}", headers=auth_headers)
        data = resp.json()

        for cell in data["cells"]:
            assert cell["zone"] in ("red", "yellow", "green")
            assert 0 <= cell["aivs"] <= 100
            assert cell["mention_count"] >= 0
            assert cell["total_count"] > 0

    @pytest.mark.asyncio
    async def test_nonexistent_project(self, client: AsyncClient, auth_headers):
        resp = await client.get("/api/v1/llm/heatmap/99999", headers=auth_headers)
        assert resp.status_code == 404


# ===================================================================
# /llm/geo-advisor/{project_id}
# ===================================================================


class TestGeoAdvisorEndpoint:
    @pytest.mark.asyncio
    async def test_happy_path(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/geo-advisor/{pid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        assert "insights" in data
        assert "generated_at" in data
        assert isinstance(data["insights"], list)

    @pytest.mark.asyncio
    async def test_insight_structure(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/geo-advisor/{pid}", headers=auth_headers)
        data = resp.json()

        if data["insights"]:
            insight = data["insights"][0]
            assert "rule_id" in insight
            assert "severity" in insight
            assert insight["severity"] in ("critical", "warning", "info")
            assert "title_ru" in insight
            assert "description_ru" in insight
            assert "recommendation_ru" in insight

    @pytest.mark.asyncio
    async def test_empty_project_no_critical(self, client: AsyncClient, auth_headers, empty_project):
        pid = empty_project["project"].id
        resp = await client.get(f"/api/v1/llm/geo-advisor/{pid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        # Empty project → INVISIBLE_PROBLEM should be triggered
        rule_ids = {i["rule_id"] for i in data["insights"]}
        assert "INVISIBLE_PROBLEM" in rule_ids

    @pytest.mark.asyncio
    async def test_with_days_filter(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/geo-advisor/{pid}?days=1", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["insights"], list)

    @pytest.mark.asyncio
    async def test_nonexistent_project(self, client: AsyncClient, auth_headers):
        resp = await client.get("/api/v1/llm/geo-advisor/99999", headers=auth_headers)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_generated_at_present(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/geo-advisor/{pid}", headers=auth_headers)
        data = resp.json()
        assert data["generated_at"] is not None
        assert len(data["generated_at"]) > 0


# ===================================================================
# Citation Graph (via bi-dashboard)
# ===================================================================


class TestCitationGraphInDashboard:
    @pytest.mark.asyncio
    async def test_citation_domains_present(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/bi-dashboard/{pid}", headers=auth_headers)
        data = resp.json()["citation_graph"]

        assert data["total_citations"] > 0
        assert len(data["domains"]) > 0

    @pytest.mark.asyncio
    async def test_competitor_win_flagged(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/bi-dashboard/{pid}", headers=auth_headers)
        data = resp.json()["citation_graph"]

        # competitor-source.com was cited in a competitor-win response
        domains_map = {d["domain"]: d for d in data["domains"]}
        assert "competitor-source.com" in domains_map
        assert domains_map["competitor-source.com"]["appears_in_competitor_wins"] is True

    @pytest.mark.asyncio
    async def test_citation_domain_structure(self, client: AsyncClient, auth_headers, bi_project):
        pid = bi_project["project"].id
        resp = await client.get(f"/api/v1/llm/bi-dashboard/{pid}", headers=auth_headers)
        data = resp.json()["citation_graph"]

        domain = data["domains"][0]
        assert "domain" in domain
        assert "count" in domain
        assert "providers" in domain
        assert "appears_in_competitor_wins" in domain
        assert isinstance(domain["providers"], list)
