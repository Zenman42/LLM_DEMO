"""Tests for BI Dashboard pure computation functions."""

from datetime import date


from app.schemas.bi_dashboard import (
    CitationDomain,
    CitationTrustGraph,
    GlobalMetrics,
    HeatmapCell,
    HeatmapResponse,
)
from app.services.bi_service import (
    SnapshotRow,
    _classify_zone,
    _count_mentions,
    _extract_domain,
    compute_aivs,
    compute_bi_dashboard,
    compute_citation_graph,
    compute_competitive_matrix,
    compute_global_metrics,
    compute_heatmap,
    compute_resilience,
    compute_som,
    generate_geo_insights,
    intent_weight,
    rvs_for_mention_type,
)


def _row(
    mention_type: str = "none",
    brand_mentioned: bool = False,
    query_type: str = "custom",
    llm_provider: str = "chatgpt",
    llm_query_id: int = 1,
    dt: date | None = None,
    competitor_mentions: dict | None = None,
    cited_urls: list | None = None,
    category: str | None = None,
) -> SnapshotRow:
    return SnapshotRow(
        llm_query_id=llm_query_id,
        query_type=query_type,
        date=dt or date(2025, 1, 15),
        llm_provider=llm_provider,
        brand_mentioned=brand_mentioned,
        mention_type=mention_type,
        competitor_mentions=competitor_mentions,
        cited_urls=cited_urls,
        category=category,
    )


# ===================================================================
# RVS Lookup
# ===================================================================


class TestRvsLookup:
    def test_recommended(self):
        assert rvs_for_mention_type("recommended") == 1.2

    def test_direct(self):
        assert rvs_for_mention_type("direct") == 0.70

    def test_compared(self):
        assert rvs_for_mention_type("compared") == 0.50

    def test_negative(self):
        assert rvs_for_mention_type("negative") == 0.35

    def test_none(self):
        assert rvs_for_mention_type("none") == 0.0

    def test_unknown_defaults_to_zero(self):
        assert rvs_for_mention_type("unknown_type") == 0.0

    def test_empty_string(self):
        assert rvs_for_mention_type("") == 0.0


class TestIntentWeight:
    def test_comparison(self):
        assert intent_weight("comparison") == 1.5

    def test_recommendation(self):
        assert intent_weight("recommendation") == 1.2

    def test_brand_check(self):
        assert intent_weight("brand_check") == 1.0

    def test_custom(self):
        assert intent_weight("custom") == 0.8

    def test_unknown_defaults_to_custom(self):
        assert intent_weight("something_else") == 0.8


# ===================================================================
# AIVS
# ===================================================================


class TestComputeAivs:
    def test_empty_rows(self):
        assert compute_aivs([]) == 0.0

    def test_all_none_mentions(self):
        rows = [_row(mention_type="none") for _ in range(5)]
        assert compute_aivs(rows) == 0.0

    def test_single_recommended(self):
        rows = [_row(mention_type="recommended", brand_mentioned=True, query_type="custom")]
        # RVS=1.2, weight=0.8 → weighted_sum=0.96, weight_sum=0.8 → 0.96/0.8=1.2 → 120 → capped at 100
        assert compute_aivs(rows) == 100.0

    def test_single_direct_custom(self):
        rows = [_row(mention_type="direct", brand_mentioned=True, query_type="custom")]
        # RVS=0.70, weight=0.8 → 0.56/0.8=0.70 → 70.0
        assert compute_aivs(rows) == 70.0

    def test_mixed_types(self):
        rows = [
            _row(mention_type="recommended", query_type="custom"),
            _row(mention_type="none", query_type="custom"),
        ]
        # (1.2*0.8 + 0.0*0.8) / (0.8+0.8) = 0.96/1.6 = 0.6 → 60.0
        assert compute_aivs(rows) == 60.0

    def test_intent_weighting_matters(self):
        # Same mention_type but different query_types
        rows_comparison = [_row(mention_type="direct", query_type="comparison")]
        rows_custom = [_row(mention_type="direct", query_type="custom")]
        # Both should give same AIVS (70.0) since single row → weight cancels out
        assert compute_aivs(rows_comparison) == 70.0
        assert compute_aivs(rows_custom) == 70.0

    def test_intent_weighting_with_mixed(self):
        rows = [
            _row(mention_type="direct", query_type="comparison"),  # 0.70 * 1.5 = 1.05
            _row(mention_type="none", query_type="custom"),  # 0.0 * 0.8 = 0.0
        ]
        # weighted_sum = 1.05, weight_sum = 1.5 + 0.8 = 2.3
        # 1.05 / 2.3 * 100 = 45.65
        expected = round(1.05 / 2.3 * 100, 2)
        assert compute_aivs(rows) == expected

    def test_capped_at_100(self):
        rows = [_row(mention_type="recommended", query_type="comparison")]
        # RVS=1.2, weight=1.5 → 1.8/1.5 = 1.2 → 120 → capped at 100
        assert compute_aivs(rows) == 100.0

    def test_all_negative(self):
        rows = [_row(mention_type="negative") for _ in range(3)]
        # 0.35 * 0.8 * 3 / (0.8 * 3) = 0.35 → 35.0
        assert compute_aivs(rows) == 35.0


# ===================================================================
# SoM
# ===================================================================


class TestComputeSom:
    def test_all_brand(self):
        assert compute_som(10, 10) == 100.0

    def test_half_and_half(self):
        assert compute_som(5, 10) == 50.0

    def test_no_mentions(self):
        assert compute_som(0, 0) == 0.0

    def test_only_competitors(self):
        assert compute_som(0, 10) == 0.0

    def test_single_response(self):
        assert compute_som(1, 1) == 100.0

    def test_rounding(self):
        result = compute_som(1, 3)
        assert result == 33.33


class TestCountMentions:
    def test_brand_only(self):
        rows = [_row(brand_mentioned=True), _row(brand_mentioned=True)]
        brand, total, comps = _count_mentions(rows)
        assert brand == 2
        assert total == 2
        assert comps == {}

    def test_brand_and_competitors(self):
        rows = [
            _row(brand_mentioned=True, competitor_mentions={"Sber": True}),
            _row(brand_mentioned=False, competitor_mentions={"Sber": True, "Alfa": True}),
        ]
        brand, total, comps = _count_mentions(rows)
        assert brand == 1
        assert comps == {"Sber": 2, "Alfa": 1}
        assert total == 1 + 2 + 1  # brand + Sber + Alfa

    def test_competitor_false_not_counted(self):
        rows = [_row(brand_mentioned=True, competitor_mentions={"Sber": False})]
        brand, total, comps = _count_mentions(rows)
        assert brand == 1
        assert total == 1
        assert comps == {}

    def test_empty(self):
        brand, total, comps = _count_mentions([])
        assert brand == 0
        assert total == 0


# ===================================================================
# Resilience
# ===================================================================


class TestComputeResilience:
    def test_empty(self):
        assert compute_resilience([]) == 0.0

    def test_always_mentioned(self):
        rows = [
            _row(brand_mentioned=True, llm_query_id=1, llm_provider="chatgpt", dt=date(2025, 1, 1)),
            _row(brand_mentioned=True, llm_query_id=1, llm_provider="chatgpt", dt=date(2025, 1, 2)),
            _row(brand_mentioned=True, llm_query_id=1, llm_provider="chatgpt", dt=date(2025, 1, 3)),
        ]
        assert compute_resilience(rows) == 1.0

    def test_never_mentioned(self):
        rows = [
            _row(brand_mentioned=False, llm_query_id=1, llm_provider="chatgpt", dt=date(2025, 1, 1)),
            _row(brand_mentioned=False, llm_query_id=1, llm_provider="chatgpt", dt=date(2025, 1, 2)),
        ]
        assert compute_resilience(rows) == 0.0

    def test_partial(self):
        rows = [
            _row(brand_mentioned=True, llm_query_id=1, llm_provider="chatgpt", dt=date(2025, 1, 1)),
            _row(brand_mentioned=False, llm_query_id=1, llm_provider="chatgpt", dt=date(2025, 1, 2)),
        ]
        assert compute_resilience(rows) == 0.5

    def test_multiple_providers(self):
        rows = [
            _row(brand_mentioned=True, llm_query_id=1, llm_provider="chatgpt", dt=date(2025, 1, 1)),
            _row(brand_mentioned=True, llm_query_id=1, llm_provider="chatgpt", dt=date(2025, 1, 2)),
            _row(brand_mentioned=True, llm_query_id=1, llm_provider="deepseek", dt=date(2025, 1, 1)),
            _row(brand_mentioned=False, llm_query_id=1, llm_provider="deepseek", dt=date(2025, 1, 2)),
        ]
        # chatgpt: 2/2 = 1.0, deepseek: 1/2 = 0.5 → avg = 0.75
        assert compute_resilience(rows) == 0.75

    def test_multiple_queries(self):
        rows = [
            _row(brand_mentioned=True, llm_query_id=1, llm_provider="chatgpt"),
            _row(brand_mentioned=False, llm_query_id=2, llm_provider="chatgpt"),
        ]
        # query1+chatgpt: 1/1=1.0, query2+chatgpt: 0/1=0.0 → avg = 0.5
        assert compute_resilience(rows) == 0.5

    def test_single_run(self):
        rows = [_row(brand_mentioned=True)]
        assert compute_resilience(rows) == 1.0


# ===================================================================
# Global Metrics
# ===================================================================


class TestComputeGlobalMetrics:
    def test_happy_path(self):
        rows = [
            _row(
                brand_mentioned=True,
                mention_type="recommended",
                query_type="comparison",
                competitor_mentions={"Sber": True},
            ),
            _row(brand_mentioned=False, mention_type="none", query_type="custom", competitor_mentions={"Sber": True}),
        ]
        m = compute_global_metrics(rows)
        assert m.total_responses == 2
        assert m.mention_rate == 0.5
        assert m.som > 0
        assert m.aivs > 0
        assert 0 <= m.resilience_score <= 1

    def test_empty(self):
        m = compute_global_metrics([])
        assert m.aivs == 0.0
        assert m.som == 0.0
        assert m.resilience_score == 0.0
        assert m.total_responses == 0
        assert m.mention_rate == 0.0

    def test_all_mentioned(self):
        rows = [_row(brand_mentioned=True, mention_type="direct") for _ in range(3)]
        m = compute_global_metrics(rows)
        assert m.mention_rate == 1.0
        assert m.som == 100.0


# ===================================================================
# Competitive Matrix
# ===================================================================


class TestComputeCompetitiveMatrix:
    def test_single_competitor(self):
        rows = [
            _row(brand_mentioned=True, mention_type="recommended", competitor_mentions={"Sber": True}),
        ]
        matrix = compute_competitive_matrix(rows)
        assert matrix.target.name == "target"
        assert matrix.target.som > 0
        assert len(matrix.competitors) == 1
        assert matrix.competitors[0].name == "Sber"

    def test_multiple_competitors(self):
        rows = [
            _row(brand_mentioned=True, competitor_mentions={"Sber": True, "Alfa": True}),
            _row(brand_mentioned=False, competitor_mentions={"Sber": True}),
        ]
        matrix = compute_competitive_matrix(rows)
        assert len(matrix.competitors) == 2
        comp_names = {c.name for c in matrix.competitors}
        assert "Sber" in comp_names
        assert "Alfa" in comp_names

    def test_no_competitors(self):
        rows = [_row(brand_mentioned=True, mention_type="direct")]
        matrix = compute_competitive_matrix(rows)
        assert matrix.target.som == 100.0
        assert matrix.competitors == []

    def test_brand_not_mentioned(self):
        rows = [_row(brand_mentioned=False, competitor_mentions={"Sber": True})]
        matrix = compute_competitive_matrix(rows)
        assert matrix.target.som == 0.0
        assert matrix.target.mention_rate == 0.0

    def test_target_avg_rvs(self):
        rows = [
            _row(brand_mentioned=True, mention_type="recommended"),
            _row(brand_mentioned=True, mention_type="direct"),
        ]
        matrix = compute_competitive_matrix(rows)
        # avg of 1.2 and 0.7 = 0.95
        assert matrix.target.avg_rvs == 0.95

    def test_competitors_sorted_by_count(self):
        rows = [
            _row(competitor_mentions={"Alfa": True, "Sber": True}),
            _row(competitor_mentions={"Sber": True}),
        ]
        matrix = compute_competitive_matrix(rows)
        # Sber has 2, Alfa has 1 → Sber first
        assert matrix.competitors[0].name == "Sber"
        assert matrix.competitors[1].name == "Alfa"


# ===================================================================
# Heatmap
# ===================================================================


class TestComputeHeatmap:
    def test_empty(self):
        h = compute_heatmap([])
        assert h.cells == []
        assert h.scenarios == []
        assert h.vendors == []

    def test_single_cell(self):
        rows = [
            _row(
                mention_type="direct",
                query_type="comparison",
                llm_provider="chatgpt",
                brand_mentioned=True,
                category="Product Comparison",
            )
        ]
        h = compute_heatmap(rows)
        assert len(h.cells) == 1
        assert h.cells[0].scenario == "Product Comparison"
        assert h.cells[0].vendor == "chatgpt"
        assert h.cells[0].aivs == 70.0
        assert h.cells[0].mention_count == 1

    def test_multiple_scenarios_and_vendors(self):
        rows = [
            _row(
                query_type="comparison",
                llm_provider="chatgpt",
                mention_type="direct",
                brand_mentioned=True,
                category="Scenario A",
            ),
            _row(
                query_type="comparison",
                llm_provider="deepseek",
                mention_type="none",
                brand_mentioned=False,
                category="Scenario A",
            ),
            _row(
                query_type="brand_check",
                llm_provider="chatgpt",
                mention_type="recommended",
                brand_mentioned=True,
                category="Scenario B",
            ),
        ]
        h = compute_heatmap(rows)
        assert len(h.cells) == 3
        assert set(h.scenarios) == {"Scenario A", "Scenario B"}
        assert set(h.vendors) == {"chatgpt", "deepseek"}

    def test_zone_classification_green(self):
        rows = [_row(mention_type="direct", query_type="custom", category="Test")]
        h = compute_heatmap(rows)
        assert h.cells[0].zone == "green"  # AIVS = 70

    def test_zone_classification_yellow(self):
        rows = [_row(mention_type="negative", query_type="custom", category="Test")]
        h = compute_heatmap(rows)
        assert h.cells[0].aivs == 35.0
        assert h.cells[0].zone == "yellow"

    def test_zone_classification_red(self):
        rows = [_row(mention_type="none", query_type="custom", category="Test")]
        h = compute_heatmap(rows)
        assert h.cells[0].aivs == 0.0
        assert h.cells[0].zone == "red"

    def test_mixed_zone(self):
        rows = [
            _row(mention_type="direct", query_type="custom", category="Test"),  # 70 green
            _row(mention_type="none", query_type="custom", category="Test"),  # dilutes to 35
        ]
        h = compute_heatmap(rows)
        assert h.cells[0].aivs == 35.0
        assert h.cells[0].zone == "yellow"

    def test_mention_count_tracked(self):
        rows = [
            _row(
                brand_mentioned=True,
                mention_type="direct",
                query_type="custom",
                llm_provider="chatgpt",
                category="Test",
            ),
            _row(
                brand_mentioned=False, mention_type="none", query_type="custom", llm_provider="chatgpt", category="Test"
            ),
            _row(
                brand_mentioned=True,
                mention_type="direct",
                query_type="custom",
                llm_provider="chatgpt",
                category="Test",
            ),
        ]
        h = compute_heatmap(rows)
        assert h.cells[0].mention_count == 2
        assert h.cells[0].total_count == 3

    def test_null_category_becomes_uncategorized(self):
        rows = [_row(mention_type="direct", query_type="custom", category=None)]
        h = compute_heatmap(rows)
        assert len(h.cells) == 1
        assert h.cells[0].scenario == "Uncategorized"
        assert "Uncategorized" in h.scenarios


class TestClassifyZone:
    def test_red(self):
        assert _classify_zone(0.0) == "red"
        assert _classify_zone(19.9) == "red"

    def test_yellow(self):
        assert _classify_zone(20.0) == "yellow"
        assert _classify_zone(49.9) == "yellow"

    def test_green(self):
        assert _classify_zone(50.0) == "green"
        assert _classify_zone(100.0) == "green"


# ===================================================================
# Citation Trust Graph
# ===================================================================


class TestComputeCitationGraph:
    def test_empty(self):
        g = compute_citation_graph([])
        assert g.domains == []
        assert g.total_citations == 0

    def test_basic_citations(self):
        rows = [
            _row(cited_urls=["https://example.com/page1", "https://test.com/article"], brand_mentioned=True),
        ]
        g = compute_citation_graph(rows)
        assert len(g.domains) == 2
        assert g.total_citations == 2
        domain_names = {d.domain for d in g.domains}
        assert "example.com" in domain_names
        assert "test.com" in domain_names

    def test_competitor_win_flagged(self):
        rows = [
            _row(
                brand_mentioned=False,
                competitor_mentions={"Sber": True},
                cited_urls=["https://competitor-source.com/review"],
            ),
        ]
        g = compute_citation_graph(rows)
        assert len(g.domains) == 1
        assert g.domains[0].appears_in_competitor_wins is True

    def test_brand_mentioned_not_competitor_win(self):
        rows = [
            _row(
                brand_mentioned=True,
                competitor_mentions={"Sber": True},
                cited_urls=["https://both.com/article"],
            ),
        ]
        g = compute_citation_graph(rows)
        assert g.domains[0].appears_in_competitor_wins is False

    def test_deduplication_across_rows(self):
        rows = [
            _row(cited_urls=["https://example.com/page1"], llm_provider="chatgpt", brand_mentioned=True),
            _row(cited_urls=["https://example.com/page2"], llm_provider="deepseek", brand_mentioned=True),
        ]
        g = compute_citation_graph(rows)
        assert len(g.domains) == 1
        assert g.domains[0].count == 2
        assert set(g.domains[0].providers) == {"chatgpt", "deepseek"}

    def test_no_cited_urls(self):
        rows = [_row(brand_mentioned=True)]
        g = compute_citation_graph(rows)
        assert g.domains == []
        assert g.total_citations == 0

    def test_sorted_by_count(self):
        rows = [
            _row(cited_urls=["https://a.com/1"], brand_mentioned=True),
            _row(cited_urls=["https://b.com/1", "https://b.com/2"], brand_mentioned=True),
        ]
        # b.com cited 2 times, a.com 1 time
        g = compute_citation_graph(rows)
        assert g.domains[0].domain == "b.com"
        assert g.domains[0].count == 2


class TestExtractDomain:
    def test_simple(self):
        assert _extract_domain("https://example.com/page") == "example.com"

    def test_www_stripped(self):
        assert _extract_domain("https://www.example.com/page") == "example.com"

    def test_invalid_url(self):
        assert _extract_domain("not-a-url") == ""

    def test_with_port(self):
        assert _extract_domain("https://example.com:8080/page") == "example.com:8080"


# ===================================================================
# GEO Advisor
# ===================================================================


class TestGenerateGeoInsights:
    def _metrics(self, aivs=50.0, som=50.0, resilience=0.5, total=10, mention_rate=0.5):
        return GlobalMetrics(
            aivs=aivs,
            som=som,
            resilience_score=resilience,
            total_responses=total,
            mention_rate=mention_rate,
        )

    def _heatmap(self, cells=None, scenarios=None, vendors=None):
        return HeatmapResponse(
            cells=cells or [],
            scenarios=scenarios or [],
            vendors=vendors or [],
        )

    def _citation_graph(self, domains=None):
        return CitationTrustGraph(
            domains=domains or [],
            total_citations=sum(d.count for d in (domains or [])),
        )

    def test_invisible_problem_zero_som(self):
        m = self._metrics(aivs=0, som=0, mention_rate=0)
        plan = generate_geo_insights(m, self._heatmap(), self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "INVISIBLE_PROBLEM" in rule_ids

    def test_invisible_problem_low_aivs(self):
        m = self._metrics(aivs=3, som=10)
        plan = generate_geo_insights(m, self._heatmap(), self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "INVISIBLE_PROBLEM" in rule_ids

    def test_toxic_trail(self):
        m = self._metrics(aivs=25, som=40, mention_rate=0.5)
        plan = generate_geo_insights(m, self._heatmap(), self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "TOXIC_TRAIL" in rule_ids

    def test_heatmap_red_zone(self):
        cells = [
            HeatmapCell(
                scenario="Price comparison", vendor="chatgpt", aivs=10, mention_count=0, total_count=5, zone="red"
            )
        ]
        h = self._heatmap(cells=cells, scenarios=["Price comparison"], vendors=["chatgpt"])
        plan = generate_geo_insights(self._metrics(), h, self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "HEATMAP_RED_ZONE" in rule_ids

    def test_comparative_gap(self):
        """Scenario where ALL providers show AIVS < 20 triggers COMPARATIVE_GAP."""
        cells = [
            HeatmapCell(
                scenario="Best alternatives", vendor="chatgpt", aivs=10, mention_count=0, total_count=5, zone="red"
            ),
            HeatmapCell(
                scenario="Best alternatives", vendor="deepseek", aivs=5, mention_count=0, total_count=5, zone="red"
            ),
        ]
        h = self._heatmap(cells=cells, scenarios=["Best alternatives"], vendors=["chatgpt", "deepseek"])
        plan = generate_geo_insights(self._metrics(), h, self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "COMPARATIVE_GAP" in rule_ids

    def test_comparative_gap_not_triggered_when_one_provider_green(self):
        """If at least one provider has AIVS >= 20, COMPARATIVE_GAP should NOT fire."""
        cells = [
            HeatmapCell(
                scenario="Best alternatives", vendor="chatgpt", aivs=10, mention_count=0, total_count=5, zone="red"
            ),
            HeatmapCell(
                scenario="Best alternatives", vendor="deepseek", aivs=55, mention_count=3, total_count=5, zone="green"
            ),
        ]
        h = self._heatmap(cells=cells, scenarios=["Best alternatives"], vendors=["chatgpt", "deepseek"])
        plan = generate_geo_insights(self._metrics(), h, self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "COMPARATIVE_GAP" not in rule_ids

    def test_locality_curse(self):
        cells = [
            HeatmapCell(
                scenario="Custom scenario", vendor="chatgpt", aivs=70, mention_count=5, total_count=5, zone="green"
            ),
            HeatmapCell(
                scenario="Custom scenario", vendor="deepseek", aivs=0, mention_count=0, total_count=5, zone="red"
            ),
            HeatmapCell(
                scenario="Custom scenario", vendor="yandexgpt", aivs=0, mention_count=0, total_count=5, zone="red"
            ),
        ]
        h = self._heatmap(cells=cells, scenarios=["Custom scenario"], vendors=["chatgpt", "deepseek", "yandexgpt"])
        plan = generate_geo_insights(self._metrics(), h, self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "LOCALITY_CURSE" in rule_ids

    def test_citation_competitor_dominance(self):
        domains = [
            CitationDomain(domain="comp1.com", count=5, providers=["chatgpt"], appears_in_competitor_wins=True),
            CitationDomain(domain="comp2.com", count=3, providers=["chatgpt"], appears_in_competitor_wins=True),
            CitationDomain(domain="comp3.com", count=2, providers=["deepseek"], appears_in_competitor_wins=True),
            CitationDomain(domain="neutral.com", count=1, providers=["chatgpt"], appears_in_competitor_wins=False),
        ]
        g = self._citation_graph(domains=domains)
        plan = generate_geo_insights(self._metrics(), self._heatmap(), g)
        rule_ids = {i.rule_id for i in plan.insights}
        assert "CITATION_COMPETITOR_DOMINANCE" in rule_ids

    def test_low_resilience(self):
        m = self._metrics(resilience=0.2, mention_rate=0.5)
        plan = generate_geo_insights(m, self._heatmap(), self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "LOW_RESILIENCE" in rule_ids

    def test_hallucination_risk(self):
        m = self._metrics(aivs=60, resilience=0.15, mention_rate=0.5)
        plan = generate_geo_insights(m, self._heatmap(), self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "HALLUCINATION_RISK" in rule_ids

    def test_no_issues(self):
        m = self._metrics(aivs=60, som=60, resilience=0.8, mention_rate=0.7)
        plan = generate_geo_insights(m, self._heatmap(), self._citation_graph())
        assert len(plan.insights) == 0

    def test_multiple_rules_triggered(self):
        m = self._metrics(aivs=0, som=0, resilience=0.1, mention_rate=0)
        cells = [
            HeatmapCell(
                scenario="Price comparison", vendor="chatgpt", aivs=5, mention_count=0, total_count=5, zone="red"
            )
        ]
        h = self._heatmap(cells=cells, scenarios=["Price comparison"], vendors=["chatgpt"])
        plan = generate_geo_insights(m, h, self._citation_graph())
        rule_ids = {i.rule_id for i in plan.insights}
        assert "INVISIBLE_PROBLEM" in rule_ids
        assert "HEATMAP_RED_ZONE" in rule_ids

    def test_severity_levels(self):
        m = self._metrics(aivs=0, som=0, resilience=0.1, mention_rate=0)
        plan = generate_geo_insights(m, self._heatmap(), self._citation_graph())
        severities = {i.severity for i in plan.insights}
        # INVISIBLE_PROBLEM is critical
        assert "critical" in severities

    def test_generated_at_present(self):
        plan = generate_geo_insights(self._metrics(), self._heatmap(), self._citation_graph())
        assert plan.generated_at is not None


# ===================================================================
# Full BI Dashboard
# ===================================================================


class TestComputeBiDashboard:
    def test_happy_path(self):
        rows = [
            _row(
                brand_mentioned=True,
                mention_type="recommended",
                query_type="comparison",
                llm_provider="chatgpt",
                competitor_mentions={"Sber": True},
                cited_urls=["https://example.com/review"],
            ),
            _row(brand_mentioned=True, mention_type="direct", query_type="brand_check", llm_provider="deepseek"),
            _row(
                brand_mentioned=False,
                mention_type="none",
                query_type="custom",
                llm_provider="chatgpt",
                competitor_mentions={"Sber": True},
                cited_urls=["https://competitor-source.com/article"],
            ),
        ]
        result = compute_bi_dashboard(rows)

        assert result.global_metrics.total_responses == 3
        assert result.global_metrics.aivs > 0
        assert result.global_metrics.som > 0
        assert len(result.competitive_matrix.competitors) > 0
        assert len(result.heatmap.cells) > 0
        assert len(result.citation_graph.domains) > 0
        assert result.geo_action_plan.generated_at is not None

    def test_empty_rows(self):
        result = compute_bi_dashboard([])
        assert result.global_metrics.aivs == 0.0
        assert result.global_metrics.som == 0.0
        assert result.heatmap.cells == []
        assert result.citation_graph.domains == []

    def test_all_fields_present(self):
        rows = [_row(brand_mentioned=True, mention_type="direct")]
        result = compute_bi_dashboard(rows)
        # Verify all top-level fields exist
        assert result.global_metrics is not None
        assert result.competitive_matrix is not None
        assert result.heatmap is not None
        assert result.citation_graph is not None
        assert result.geo_action_plan is not None
