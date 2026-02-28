"""Tests for Micro-Scoring Calculator."""

from app.analysis.scoring import (
    apply_scores,
    calculate_rvs,
    calculate_share_of_model_local,
)
from app.analysis.types import BrandMention


class TestCalculateRVS:
    """Test Response Visibility Score calculation."""

    def test_not_mentioned(self):
        brand = BrandMention(name="Brand", is_mentioned=False)
        assert calculate_rvs(brand) == 0.0

    def test_top_ranked_positive(self):
        brand = BrandMention(
            name="Brand",
            is_mentioned=True,
            position_weight=1.0,
            sentiment_multiplier=1.2,
        )
        rvs = calculate_rvs(brand)
        assert rvs == 1.2  # 1.0 × 1.0 × 1.2

    def test_second_ranked_neutral(self):
        brand = BrandMention(
            name="Brand",
            is_mentioned=True,
            position_weight=0.9,
            sentiment_multiplier=1.0,
        )
        rvs = calculate_rvs(brand)
        assert rvs == 0.9  # 1.0 × 0.9 × 1.0

    def test_mentioned_negative(self):
        brand = BrandMention(
            name="Brand",
            is_mentioned=True,
            position_weight=0.8,
            sentiment_multiplier=0.5,
        )
        rvs = calculate_rvs(brand)
        assert rvs == 0.4  # 1.0 × 0.8 × 0.5

    def test_low_ranked(self):
        brand = BrandMention(
            name="Brand",
            is_mentioned=True,
            position_weight=0.1,
            sentiment_multiplier=1.0,
        )
        rvs = calculate_rvs(brand)
        assert rvs == 0.1

    def test_zero_weight(self):
        brand = BrandMention(
            name="Brand",
            is_mentioned=True,
            position_weight=0.0,
            sentiment_multiplier=1.0,
        )
        assert calculate_rvs(brand) == 0.0


class TestShareOfModelLocal:
    """Test share_of_model_local calculation."""

    def test_target_only_mentioned(self):
        target = BrandMention(name="Target", is_mentioned=True)
        comps = [
            BrandMention(name="Comp1", is_mentioned=False),
            BrandMention(name="Comp2", is_mentioned=False),
        ]
        share = calculate_share_of_model_local(target, comps)
        assert share == 1.0  # 1/1

    def test_target_and_one_competitor(self):
        target = BrandMention(name="Target", is_mentioned=True)
        comps = [BrandMention(name="Comp1", is_mentioned=True)]
        share = calculate_share_of_model_local(target, comps)
        assert share == 0.5  # 1/2

    def test_target_and_two_competitors(self):
        target = BrandMention(name="Target", is_mentioned=True)
        comps = [
            BrandMention(name="Comp1", is_mentioned=True),
            BrandMention(name="Comp2", is_mentioned=True),
        ]
        share = calculate_share_of_model_local(target, comps)
        assert abs(share - 0.3333) < 0.001  # 1/3

    def test_target_not_mentioned(self):
        target = BrandMention(name="Target", is_mentioned=False)
        comps = [BrandMention(name="Comp1", is_mentioned=True)]
        share = calculate_share_of_model_local(target, comps)
        assert share == 0.0

    def test_no_competitors(self):
        target = BrandMention(name="Target", is_mentioned=True)
        share = calculate_share_of_model_local(target, [])
        assert share == 1.0

    def test_five_brands_mentioned(self):
        target = BrandMention(name="Target", is_mentioned=True)
        comps = [BrandMention(name=f"Comp{i}", is_mentioned=True) for i in range(4)]
        share = calculate_share_of_model_local(target, comps)
        assert share == 0.2  # 1/5


class TestApplyScores:
    """Test combined scoring."""

    def test_apply_scores_returns_tuple(self):
        target = BrandMention(
            name="Brand",
            is_mentioned=True,
            position_weight=1.0,
            sentiment_multiplier=1.2,
        )
        comps = [BrandMention(name="Comp", is_mentioned=True)]

        rvs, share = apply_scores(target, comps)
        assert rvs == 1.2
        assert share == 0.5

    def test_apply_scores_not_mentioned(self):
        target = BrandMention(name="Brand", is_mentioned=False)
        rvs, share = apply_scores(target, [])
        assert rvs == 0.0
        assert share == 0.0
