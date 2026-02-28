"""Tests for analysis types and DTOs."""

from app.analysis.types import (
    AnalyzedResponse,
    BrandMention,
    CitationInfo,
    MentionType,
    SanitizationFlag,
    SentimentLabel,
    StructureType,
)


class TestEnums:
    """Test all analysis enums."""

    def test_sanitization_flag_values(self):
        assert SanitizationFlag.CLEAN == "clean"
        assert SanitizationFlag.DEEPSEEK_THINK_STRIPPED == "deepseek_think_stripped"
        assert SanitizationFlag.VENDOR_REFUSAL == "vendor_refusal"
        assert SanitizationFlag.CENSORED == "censored"
        assert SanitizationFlag.EMPTY_RESPONSE == "empty_response"

    def test_mention_type_values(self):
        assert MentionType.DIRECT == "direct"
        assert MentionType.RECOMMENDED == "recommended"
        assert MentionType.COMPARED == "compared"
        assert MentionType.NEGATIVE == "negative"
        assert MentionType.NONE == "none"

    def test_structure_type_values(self):
        assert StructureType.NUMBERED_LIST == "numbered_list"
        assert StructureType.BULLETED_LIST == "bulleted_list"
        assert StructureType.NARRATIVE == "narrative"
        assert StructureType.TABLE == "table"
        assert StructureType.MIXED == "mixed"

    def test_sentiment_label_values(self):
        assert SentimentLabel.POSITIVE == "positive"
        assert SentimentLabel.NEUTRAL == "neutral"
        assert SentimentLabel.NEGATIVE == "negative"
        assert SentimentLabel.MIXED == "mixed"


class TestBrandMention:
    """Test BrandMention dataclass."""

    def test_default_values(self):
        m = BrandMention(name="TestBrand")
        assert m.name == "TestBrand"
        assert m.is_mentioned is False
        assert m.position_rank == 0
        assert m.position_weight == 0.0
        assert m.mention_type == MentionType.NONE
        assert m.sentiment_score == 0.0
        assert m.sentiment_label == SentimentLabel.NEUTRAL
        assert m.sentiment_multiplier == 1.0
        assert m.aliases_matched == []
        assert m.context_tags == []
        assert m.is_recommended is False
        assert m.is_hallucination is False

    def test_populated_values(self):
        m = BrandMention(
            name="Тинькофф",
            aliases_matched=["Т-Банк", "T-Bank"],
            is_mentioned=True,
            position_rank=1,
            position_weight=1.0,
            mention_type=MentionType.RECOMMENDED,
            mention_context="Тинькофф — лучший банк для бизнеса",
            char_offset=0,
            sentence_index=0,
            sentiment_score=0.8,
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_multiplier=1.2,
            context_tags=["quality", "features"],
            is_recommended=True,
        )
        assert m.is_mentioned is True
        assert m.position_rank == 1
        assert m.sentiment_multiplier == 1.2


class TestAnalyzedResponse:
    """Test AnalyzedResponse DTO and serialization."""

    def test_default_values(self):
        ar = AnalyzedResponse()
        assert ar.request_id == ""
        assert ar.response_visibility_score == 0.0
        assert ar.share_of_model_local == 0.0

    def test_to_dict_structure(self):
        ar = AnalyzedResponse(
            request_id="test-123",
            vendor="chatgpt",
            model_version="gpt-4o-mini",
            response_visibility_score=0.85,
            share_of_model_local=0.5,
            latency_ms=1200,
            cost_usd=0.003,
        )
        ar.target_brand = BrandMention(
            name="MyBrand",
            is_mentioned=True,
            position_rank=1,
            position_weight=1.0,
            sentiment_score=0.7,
            sentiment_multiplier=1.2,
            mention_type=MentionType.RECOMMENDED,
            is_recommended=True,
        )

        d = ar.to_dict()

        assert d["request_id"] == "test-123"
        assert d["vendor"] == "chatgpt"
        assert d["extracted_metrics"]["target_brand"]["name"] == "MyBrand"
        assert d["extracted_metrics"]["target_brand"]["is_mentioned"] is True
        assert d["extracted_metrics"]["target_brand"]["mention_type"] == "recommended"
        assert d["calculated_scores"]["response_visibility_score"] == 0.85
        assert d["calculated_scores"]["share_of_model_local"] == 0.5
        assert d["latency_ms"] == 1200

    def test_to_dict_with_competitors(self):
        ar = AnalyzedResponse()
        ar.target_brand = BrandMention(name="Target", is_mentioned=True)
        ar.competitors = [
            BrandMention(name="Comp1", is_mentioned=True, mention_type=MentionType.COMPARED),
            BrandMention(name="Comp2", is_mentioned=False),
        ]

        d = ar.to_dict()
        # Only mentioned competitors should appear
        comps = d["extracted_metrics"]["competitors_mentioned"]
        assert len(comps) == 1
        assert comps[0]["name"] == "Comp1"

    def test_to_dict_with_citations(self):
        ar = AnalyzedResponse()
        ar.target_brand = BrandMention(name="Target")
        ar.citations = [
            CitationInfo(url="https://example.com/page", domain="example.com"),
        ]

        d = ar.to_dict()
        cits = d["extracted_metrics"]["citations_found"]
        assert len(cits) == 1
        assert cits[0]["domain"] == "example.com"
