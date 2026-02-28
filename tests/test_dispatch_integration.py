"""Tests for LLM collection dispatch integration and pipeline bridge."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.collectors.llm_base import (
    AnalysisResult,
    BaseLlmCollector,
    LlmResponse,
    _PIPELINE_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Concrete test collector (since BaseLlmCollector is abstract)
# ---------------------------------------------------------------------------


class FakeCollector(BaseLlmCollector):
    """Test collector that returns a canned response."""

    provider = "chatgpt"

    def __init__(self, api_key="fake", tenant_id=None, *, response_text="", model="gpt-4o-mini"):
        super().__init__(api_key=api_key, tenant_id=tenant_id or uuid.uuid4())
        self._response_text = response_text
        self._model = model

    async def query_llm(self, prompt: str) -> LlmResponse:
        return LlmResponse(
            text=self._response_text,
            model=self._model,
            tokens=100,
            cost_usd=0.001,
        )


# ---------------------------------------------------------------------------
# Test: _dispatch_llm_collection
# ---------------------------------------------------------------------------


class TestDispatchLlmCollection:
    """Tests for _dispatch_llm_collection in collection_tasks.py."""

    @patch("app.tasks.collection_tasks._run_async")
    @patch("app.tasks.collection_tasks._find_llm_projects")
    def test_dispatch_calls_llm_task(self, mock_find, mock_run_async):
        """Dispatches collect_llm_project_task for each LLM project."""
        from app.tasks.collection_tasks import _dispatch_llm_collection

        # _run_async wraps async call → return project IDs
        mock_run_async.return_value = [1, 5, 42]

        with patch("app.tasks.llm_collection_tasks.collect_llm_project_task") as mock_task:
            mock_task.delay = MagicMock()
            count = _dispatch_llm_collection("tenant-123")

        assert count == 3
        assert mock_task.delay.call_count == 3
        mock_task.delay.assert_any_call("tenant-123", 1)
        mock_task.delay.assert_any_call("tenant-123", 5)
        mock_task.delay.assert_any_call("tenant-123", 42)

    @patch("app.tasks.collection_tasks._run_async")
    def test_dispatch_no_llm_projects(self, mock_run_async):
        """Returns 0 when no projects have track_llm enabled."""
        from app.tasks.collection_tasks import _dispatch_llm_collection

        mock_run_async.return_value = []

        with patch("app.tasks.llm_collection_tasks.collect_llm_project_task") as mock_task:
            mock_task.delay = MagicMock()
            count = _dispatch_llm_collection("tenant-456")

        assert count == 0
        mock_task.delay.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Pipeline integration in BaseLlmCollector
# ---------------------------------------------------------------------------


class TestPipelineAvailability:
    """Verify that Module 3 pipeline is importable."""

    def test_pipeline_available(self):
        """Module 3 should be available in the test environment."""
        assert _PIPELINE_AVAILABLE is True


class TestAnalyzeWithPipeline:
    """Tests for _analyze_with_pipeline bridging LlmResponse → Module 3."""

    def _make_query(self, query_text="test", target_brand="TestBrand", competitors=None, query_type="brand_check"):
        """Create a mock LlmQuery."""
        q = MagicMock()
        q.id = 1
        q.query_text = query_text
        q.target_brand = target_brand
        q.competitors = competitors or []
        q.query_type = query_type
        return q

    @pytest.mark.asyncio
    async def test_pipeline_detects_brand_mention(self):
        """Pipeline correctly detects brand mentioned in response."""
        collector = FakeCollector(
            response_text="We recommend TestBrand for its great features.",
        )
        query = self._make_query()
        llm_resp = LlmResponse(text="We recommend TestBrand for its great features.", model="gpt-4o-mini", tokens=50)

        result = await collector._analyze_with_pipeline(
            llm_resp=llm_resp,
            query=query,
            target_brand="TestBrand",
            competitors=["CompA"],
            project_id=1,
        )

        assert isinstance(result, AnalysisResult)
        assert result.brand_mentioned is True
        assert result.mention_type in ("direct", "recommended", "compared", "negative")

    @pytest.mark.asyncio
    async def test_pipeline_no_brand_mention(self):
        """Pipeline correctly identifies when brand is NOT mentioned."""
        collector = FakeCollector()
        query = self._make_query()
        llm_resp = LlmResponse(text="Here are some generic SEO tools.", model="gpt-4o-mini", tokens=30)

        result = await collector._analyze_with_pipeline(
            llm_resp=llm_resp,
            query=query,
            target_brand="TestBrand",
            competitors=[],
            project_id=1,
        )

        assert result.brand_mentioned is False
        assert result.mention_type == "none"

    @pytest.mark.asyncio
    async def test_pipeline_competitor_detection(self):
        """Pipeline detects competitor mentions."""
        collector = FakeCollector()
        query = self._make_query(competitors=["CompA", "CompB"])
        text = "TestBrand and CompA are both great tools. CompB is also mentioned."
        llm_resp = LlmResponse(text=text, model="gpt-4o-mini", tokens=50)

        result = await collector._analyze_with_pipeline(
            llm_resp=llm_resp,
            query=query,
            target_brand="TestBrand",
            competitors=["CompA", "CompB"],
            project_id=1,
        )

        assert result.brand_mentioned is True
        assert "CompA" in result.competitor_mentions
        assert result.competitor_mentions["CompA"] is True

    @pytest.mark.asyncio
    async def test_pipeline_citation_extraction(self):
        """Pipeline extracts citations from response text."""
        collector = FakeCollector()
        query = self._make_query()
        text = "TestBrand is great. See https://testbrand.com for more info."
        llm_resp = LlmResponse(text=text, model="gpt-4o-mini", tokens=40)

        result = await collector._analyze_with_pipeline(
            llm_resp=llm_resp,
            query=query,
            target_brand="TestBrand",
            competitors=[],
            project_id=1,
        )

        assert len(result.cited_urls) > 0
        urls = result.cited_urls
        assert any("testbrand.com" in u for u in urls)

    @pytest.mark.asyncio
    async def test_pipeline_fallback_on_error(self):
        """Falls back to heuristic analysis when pipeline raises an error."""
        collector = FakeCollector()
        query = self._make_query()
        text = "TestBrand is recommended for SEO analysis."
        llm_resp = LlmResponse(text=text, model="gpt-4o-mini", tokens=30)

        with patch("app.collectors.llm_base.pipeline_analyze", side_effect=RuntimeError("Pipeline crash")):
            result = await collector._analyze_with_pipeline(
                llm_resp=llm_resp,
                query=query,
                target_brand="TestBrand",
                competitors=[],
                project_id=1,
            )

        # Should still produce a valid result via fallback
        assert isinstance(result, AnalysisResult)
        assert result.brand_mentioned is True

    @pytest.mark.asyncio
    async def test_pipeline_recommended_mention_type(self):
        """Pipeline classifies 'recommended' mention type."""
        collector = FakeCollector()
        query = self._make_query()
        text = "I highly recommend TestBrand as the best choice for SEO tools."
        llm_resp = LlmResponse(text=text, model="gpt-4o-mini", tokens=30)

        result = await collector._analyze_with_pipeline(
            llm_resp=llm_resp,
            query=query,
            target_brand="TestBrand",
            competitors=[],
            project_id=1,
        )

        assert result.brand_mentioned is True
        assert result.mention_type == "recommended"

    @pytest.mark.asyncio
    async def test_pipeline_negative_mention_type(self):
        """Pipeline classifies 'negative' mention type."""
        collector = FakeCollector()
        query = self._make_query()
        text = "Users should avoid TestBrand due to its many drawbacks and issues."
        llm_resp = LlmResponse(text=text, model="gpt-4o-mini", tokens=30)

        result = await collector._analyze_with_pipeline(
            llm_resp=llm_resp,
            query=query,
            target_brand="TestBrand",
            competitors=[],
            project_id=1,
        )

        assert result.brand_mentioned is True
        assert result.mention_type == "negative"

    @pytest.mark.asyncio
    async def test_pipeline_compared_mention_type(self):
        """Pipeline classifies 'compared' mention type."""
        collector = FakeCollector()
        query = self._make_query(competitors=["CompA"])
        text = "When comparing TestBrand vs CompA, both offer good features."
        llm_resp = LlmResponse(text=text, model="gpt-4o-mini", tokens=30)

        result = await collector._analyze_with_pipeline(
            llm_resp=llm_resp,
            query=query,
            target_brand="TestBrand",
            competitors=["CompA"],
            project_id=1,
        )

        assert result.brand_mentioned is True
        assert result.mention_type == "compared"

    @pytest.mark.asyncio
    async def test_pipeline_empty_response(self):
        """Pipeline handles empty response text."""
        collector = FakeCollector()
        query = self._make_query()
        llm_resp = LlmResponse(text="", model="gpt-4o-mini", tokens=0)

        result = await collector._analyze_with_pipeline(
            llm_resp=llm_resp,
            query=query,
            target_brand="TestBrand",
            competitors=[],
            project_id=1,
        )

        assert result.brand_mentioned is False


# ---------------------------------------------------------------------------
# Test: Inline heuristic fallback
# ---------------------------------------------------------------------------


class TestInlineHeuristics:
    """Tests for the regex-based fallback analyze_response method."""

    def test_heuristic_brand_detection(self):
        collector = FakeCollector()
        result = collector.analyze_response(
            "TestBrand is a great tool.",
            target_brand="TestBrand",
            competitors=[],
        )
        assert result.brand_mentioned is True

    def test_heuristic_no_brand(self):
        collector = FakeCollector()
        result = collector.analyze_response(
            "Some generic SEO tools exist.",
            target_brand="TestBrand",
            competitors=[],
        )
        assert result.brand_mentioned is False

    def test_heuristic_recommended(self):
        collector = FakeCollector()
        result = collector.analyze_response(
            "I highly recommend TestBrand for SEO.",
            target_brand="TestBrand",
            competitors=[],
        )
        assert result.mention_type == "recommended"

    def test_heuristic_negative(self):
        collector = FakeCollector()
        result = collector.analyze_response(
            "Users should avoid TestBrand due to issues.",
            target_brand="TestBrand",
            competitors=[],
        )
        assert result.mention_type == "negative"

    def test_heuristic_compared(self):
        collector = FakeCollector()
        result = collector.analyze_response(
            "Comparing TestBrand vs CompA shows differences.",
            target_brand="TestBrand",
            competitors=["CompA"],
        )
        assert result.mention_type == "compared"

    def test_heuristic_url_extraction(self):
        collector = FakeCollector()
        result = collector.analyze_response(
            "Visit https://example.com for more on TestBrand.",
            target_brand="TestBrand",
            competitors=[],
        )
        assert "https://example.com" in result.cited_urls

    def test_heuristic_competitor_detection(self):
        collector = FakeCollector()
        result = collector.analyze_response(
            "TestBrand and CompA are both available.",
            target_brand="TestBrand",
            competitors=["CompA", "CompB"],
        )
        assert result.competitor_mentions["CompA"] is True
        assert result.competitor_mentions["CompB"] is False

    def test_heuristic_empty_brand(self):
        collector = FakeCollector()
        result = collector.analyze_response(
            "Some text here.",
            target_brand="",
            competitors=[],
        )
        assert result.brand_mentioned is False

    def test_heuristic_case_insensitive(self):
        collector = FakeCollector()
        result = collector.analyze_response(
            "TESTBRAND is a great tool.",
            target_brand="TestBrand",
            competitors=[],
        )
        assert result.brand_mentioned is True


# ---------------------------------------------------------------------------
# Test: collect() method integration
# ---------------------------------------------------------------------------


class TestCollectMethod:
    """Tests for BaseLlmCollector.collect() orchestration."""

    @pytest.mark.asyncio
    async def test_collect_skips_empty_queries(self):
        collector = FakeCollector(response_text="Irrelevant")
        db = AsyncMock()
        result = await collector.collect(
            db=db,
            project_id=1,
            queries=[],
            brand_name="TestBrand",
        )
        assert result.skipped is True
        assert result.collected == 0

    @pytest.mark.asyncio
    async def test_collect_processes_query(self):
        collector = FakeCollector(response_text="TestBrand is recommended.")
        db = AsyncMock()

        query = MagicMock()
        query.id = 1
        query.query_text = "What is the best tool?"
        query.target_brand = "TestBrand"
        query.competitors = []

        with patch.object(collector, "_save_snapshot", new_callable=AsyncMock):
            result = await collector.collect(
                db=db,
                project_id=1,
                queries=[query],
                brand_name="TestBrand",
            )

        assert result.collected == 1
        assert result.skipped is False
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_collect_handles_query_error(self):
        collector = FakeCollector(response_text="ok")
        db = AsyncMock()

        query = MagicMock()
        query.id = 1
        query.query_text = "test"
        query.target_brand = "TestBrand"
        query.competitors = []

        # Make query_llm raise an error
        collector.query_llm = AsyncMock(side_effect=RuntimeError("API Error"))

        result = await collector.collect(
            db=db,
            project_id=1,
            queries=[query],
            brand_name="TestBrand",
        )

        assert result.collected == 0
        assert len(result.errors) == 1
        assert "API Error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_collect_merges_competitors(self):
        collector = FakeCollector(response_text="TestBrand and CompA and CompB info.")
        db = AsyncMock()

        query = MagicMock()
        query.id = 1
        query.query_text = "test"
        query.target_brand = "TestBrand"
        query.competitors = ["CompA"]

        saved_analyses = []

        async def capture_save(
            *, db, llm_query, today, llm_model, analysis, raw_response, response_tokens, cost_usd, cited_urls
        ):
            saved_analyses.append(analysis)

        collector._save_snapshot = capture_save

        result = await collector.collect(
            db=db,
            project_id=1,
            queries=[query],
            brand_name="TestBrand",
            competitors=["CompB"],
        )

        assert result.collected == 1
        # Both CompA (from query) and CompB (from project) should be checked
        analysis = saved_analyses[0]
        assert "CompA" in analysis.competitor_mentions
        assert "CompB" in analysis.competitor_mentions
