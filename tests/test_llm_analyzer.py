"""Tests for LLM response analyzer (heuristic-based brand mention analysis)."""

import pytest

from app.collectors.llm_base import BaseLlmCollector, LlmResponse


class FakeLlmCollector(BaseLlmCollector):
    """Concrete subclass for testing analyze_response without hitting an API."""

    provider = "fake"

    async def query_llm(self, prompt: str) -> LlmResponse:
        return LlmResponse(text="", model="fake-1")


@pytest.fixture
def analyzer():
    import uuid

    return FakeLlmCollector(api_key="fake", tenant_id=uuid.uuid4())


class TestBrandMention:
    def test_brand_found_exact(self, analyzer):
        result = analyzer.analyze_response(
            response_text="I recommend using Ahrefs for backlink analysis.",
            target_brand="Ahrefs",
            competitors=[],
        )
        assert result.brand_mentioned is True

    def test_brand_found_case_insensitive(self, analyzer):
        result = analyzer.analyze_response(
            response_text="ahrefs is a popular SEO tool.",
            target_brand="Ahrefs",
            competitors=[],
        )
        assert result.brand_mentioned is True

    def test_brand_not_found(self, analyzer):
        result = analyzer.analyze_response(
            response_text="There are many SEO tools available on the market.",
            target_brand="Ahrefs",
            competitors=[],
        )
        assert result.brand_mentioned is False
        assert result.mention_type == "none"

    def test_empty_brand(self, analyzer):
        result = analyzer.analyze_response(
            response_text="Some text here.",
            target_brand="",
            competitors=[],
        )
        assert result.brand_mentioned is False

    def test_brand_word_boundary(self, analyzer):
        """Brand 'Moz' should not match 'Mozilla'."""
        result = analyzer.analyze_response(
            response_text="Mozilla Firefox is a web browser.",
            target_brand="Moz",
            competitors=[],
        )
        assert result.brand_mentioned is False

    def test_brand_with_special_chars(self, analyzer):
        """Brand names with dots/special chars should be escaped in regex."""
        result = analyzer.analyze_response(
            response_text="Check out SE Ranking for tracking positions.",
            target_brand="SE Ranking",
            competitors=[],
        )
        assert result.brand_mentioned is True


class TestMentionType:
    def test_direct_mention(self, analyzer):
        result = analyzer.analyze_response(
            response_text="Ahrefs provides comprehensive data for SEO professionals.",
            target_brand="Ahrefs",
            competitors=[],
        )
        assert result.brand_mentioned is True
        assert result.mention_type == "direct"

    def test_recommended_mention(self, analyzer):
        result = analyzer.analyze_response(
            response_text="I would recommend Ahrefs as the best choice for backlink analysis.",
            target_brand="Ahrefs",
            competitors=[],
        )
        assert result.brand_mentioned is True
        assert result.mention_type == "recommended"

    def test_negative_mention(self, analyzer):
        result = analyzer.analyze_response(
            response_text="You should avoid Ahrefs due to issues with data accuracy.",
            target_brand="Ahrefs",
            competitors=[],
        )
        assert result.brand_mentioned is True
        assert result.mention_type == "negative"

    def test_compared_mention(self, analyzer):
        result = analyzer.analyze_response(
            response_text="Compared to SEMrush, Ahrefs offers better backlink data vs other tools.",
            target_brand="Ahrefs",
            competitors=["SEMrush"],
        )
        assert result.brand_mentioned is True
        assert result.mention_type == "compared"

    def test_negative_takes_priority(self, analyzer):
        """Negative patterns should be detected even when recommend words present."""
        result = analyzer.analyze_response(
            response_text="Some recommend Ahrefs but I would avoid it due to problems with pricing.",
            target_brand="Ahrefs",
            competitors=[],
        )
        assert result.mention_type == "negative"


class TestCompetitorMentions:
    def test_competitor_found(self, analyzer):
        result = analyzer.analyze_response(
            response_text="Both Ahrefs and SEMrush are popular. Moz is another option.",
            target_brand="Ahrefs",
            competitors=["SEMrush", "Moz", "Majestic"],
        )
        assert result.competitor_mentions["SEMrush"] is True
        assert result.competitor_mentions["Moz"] is True
        assert result.competitor_mentions["Majestic"] is False

    def test_no_competitors(self, analyzer):
        result = analyzer.analyze_response(
            response_text="Ahrefs is a great tool.",
            target_brand="Ahrefs",
            competitors=[],
        )
        assert result.competitor_mentions == {}


class TestUrlExtraction:
    def test_urls_extracted(self, analyzer):
        result = analyzer.analyze_response(
            response_text="Visit https://ahrefs.com/blog and also https://moz.com/learn for more.",
            target_brand="Ahrefs",
            competitors=[],
        )
        assert "https://ahrefs.com/blog" in result.cited_urls
        assert "https://moz.com/learn" in result.cited_urls

    def test_urls_with_trailing_punct(self, analyzer):
        result = analyzer.analyze_response(
            response_text="Check https://example.com/page.",
            target_brand="TestBrand",
            competitors=[],
        )
        assert "https://example.com/page" in result.cited_urls

    def test_no_urls(self, analyzer):
        result = analyzer.analyze_response(
            response_text="No links here, just text.",
            target_brand="TestBrand",
            competitors=[],
        )
        assert result.cited_urls == []


class TestMentionContext:
    def test_context_extracted(self, analyzer):
        long_text = "x" * 200 + " Ahrefs is excellent " + "y" * 200
        result = analyzer.analyze_response(
            response_text=long_text,
            target_brand="Ahrefs",
            competitors=[],
        )
        assert result.brand_mentioned is True
        assert "Ahrefs" in result.mention_context
        assert len(result.mention_context) <= 160  # ~75 + brand + ~75


class TestClassifyMention:
    def test_classify_direct(self):
        assert BaseLlmCollector._classify_mention("Ahrefs provides great data.", []) == "direct"

    def test_classify_recommended(self):
        assert BaseLlmCollector._classify_mention("I recommend Ahrefs as the best choice.", []) == "recommended"

    def test_classify_negative(self):
        assert BaseLlmCollector._classify_mention("Avoid Ahrefs due to issues.", []) == "negative"

    def test_classify_compared_with_competitor(self):
        result = BaseLlmCollector._classify_mention("Compared to SEMrush, Ahrefs is better.", ["SEMrush"])
        assert result == "compared"

    def test_classify_compare_without_competitor_is_direct(self):
        """If compare words are present but no competitor, classify as recommended or direct."""
        result = BaseLlmCollector._classify_mention("Ahrefs is the best choice for SEO.", ["SEMrush"])
        # SEMrush not in context, so not compared
        assert result in ("direct", "recommended")
