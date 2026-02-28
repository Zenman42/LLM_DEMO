"""Tests for Context Evaluator / LLM-as-a-Judge."""

from app.analysis.context_evaluator import (
    build_judge_prompt,
    evaluate_all_heuristic,
    evaluate_brand_heuristic,
    parse_judge_response,
)
from app.analysis.types import BrandMention, SentimentLabel


class TestHeuristicEvaluation:
    """Test rule-based sentiment evaluation."""

    def test_positive_context(self):
        brand = BrandMention(
            name="Тинькофф",
            is_mentioned=True,
            mention_context="Тинькофф — лучший и надёжный банк с отличным обслуживанием",
        )
        evaluate_brand_heuristic(brand)
        assert brand.sentiment_score > 0
        assert brand.sentiment_label == SentimentLabel.POSITIVE
        assert brand.sentiment_multiplier == 1.2

    def test_negative_context(self):
        brand = BrandMention(
            name="BadBank",
            is_mentioned=True,
            mention_context="BadBank has many problems and is unreliable and expensive",
        )
        evaluate_brand_heuristic(brand)
        assert brand.sentiment_score < 0
        assert brand.sentiment_label == SentimentLabel.NEGATIVE
        assert brand.sentiment_multiplier == 0.5

    def test_neutral_context(self):
        brand = BrandMention(
            name="Тинькофф",
            is_mentioned=True,
            mention_context="Тинькофф работает в России",
        )
        evaluate_brand_heuristic(brand)
        assert brand.sentiment_label == SentimentLabel.NEUTRAL
        assert brand.sentiment_multiplier == 1.0

    def test_not_mentioned_brand_unchanged(self):
        brand = BrandMention(name="Тинькофф", is_mentioned=False)
        evaluate_brand_heuristic(brand)
        assert brand.sentiment_score == 0.0
        assert brand.sentiment_multiplier == 1.0

    def test_empty_context_unchanged(self):
        brand = BrandMention(
            name="Тинькофф",
            is_mentioned=True,
            mention_context="",
        )
        evaluate_brand_heuristic(brand)
        assert brand.sentiment_score == 0.0


class TestContextTags:
    """Test context tag extraction."""

    def test_price_tag(self):
        brand = BrandMention(
            name="Bank",
            is_mentioned=True,
            mention_context="Bank offers competitive pricing and low fees",
        )
        evaluate_brand_heuristic(brand)
        assert "price" in brand.context_tags

    def test_quality_tag(self):
        brand = BrandMention(
            name="Банк",
            is_mentioned=True,
            mention_context="Банк надёжный и качественный",
        )
        evaluate_brand_heuristic(brand)
        assert "quality" in brand.context_tags

    def test_multiple_tags(self):
        brand = BrandMention(
            name="Service",
            is_mentioned=True,
            mention_context="Service with great customer support, fast speed, and security features",
        )
        evaluate_brand_heuristic(brand)
        assert "support" in brand.context_tags
        assert "speed" in brand.context_tags
        assert "security" in brand.context_tags


class TestParseJudgeResponse:
    """Test parsing of LLM judge JSON responses."""

    def test_valid_json(self):
        brand = BrandMention(name="TestBrand", is_mentioned=True)
        raw = '{"sentiment_score": 0.8, "sentiment_label": "positive", "is_recommended": true, "context_tags": ["price", "quality"], "is_hallucination": false}'
        result = parse_judge_response(raw, brand)

        assert result.sentiment_score == 0.8
        assert result.sentiment_label == SentimentLabel.POSITIVE
        assert result.is_recommended is True
        assert "price" in result.context_tags
        assert result.is_hallucination is False
        assert result.sentiment_multiplier == 1.2

    def test_json_in_code_block(self):
        brand = BrandMention(name="TestBrand", is_mentioned=True)
        raw = '```json\n{"sentiment_score": -0.5, "sentiment_label": "negative", "is_recommended": false, "context_tags": [], "is_hallucination": true}\n```'
        result = parse_judge_response(raw, brand)

        assert result.sentiment_score == -0.5
        assert result.sentiment_label == SentimentLabel.NEGATIVE
        assert result.is_hallucination is True

    def test_invalid_json_fallback(self):
        brand = BrandMention(
            name="TestBrand",
            is_mentioned=True,
            mention_context="TestBrand is the best and most reliable option",
        )
        result = parse_judge_response("not valid json", brand)
        # Should fall back to heuristic
        assert result.sentiment_score != 0.0 or result.sentiment_label != SentimentLabel.NEUTRAL

    def test_score_clamped(self):
        brand = BrandMention(name="TestBrand", is_mentioned=True)
        raw = '{"sentiment_score": 5.0, "sentiment_label": "positive"}'
        result = parse_judge_response(raw, brand)
        assert result.sentiment_score == 1.0  # Clamped


class TestBuildJudgePrompt:
    """Test judge prompt construction."""

    def test_prompt_contains_brand(self):
        brand = BrandMention(
            name="Тинькофф",
            is_mentioned=True,
            mention_context="Тинькофф — лучший банк для бизнеса",
        )
        system, user = build_judge_prompt(brand, vendor="chatgpt")
        assert "Тинькофф" in user
        assert "chatgpt" in user
        assert "JSON" in system

    def test_context_truncated(self):
        brand = BrandMention(
            name="Brand",
            is_mentioned=True,
            mention_context="A" * 2000,
        )
        _, user = build_judge_prompt(brand)
        # Context should be truncated to 1000 chars
        assert len(user) < 1100 + 200  # template overhead


class TestEvaluateAllHeuristic:
    """Test batch heuristic evaluation."""

    def test_evaluates_target_and_competitors(self):
        target = BrandMention(
            name="Target",
            is_mentioned=True,
            mention_context="Target is the best and most reliable choice",
        )
        comp1 = BrandMention(
            name="Comp1",
            is_mentioned=True,
            mention_context="Comp1 has many problems and disadvantages",
        )
        comp2 = BrandMention(
            name="Comp2",
            is_mentioned=False,
        )

        t, comps = evaluate_all_heuristic(target, [comp1, comp2])

        assert t.sentiment_label == SentimentLabel.POSITIVE
        assert comps[0].sentiment_label == SentimentLabel.NEGATIVE
        assert comps[1].sentiment_score == 0.0  # Not mentioned
