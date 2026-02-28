"""Tests for NER & Entity Resolution Engine."""

from app.analysis.entity_resolver import resolve_brand, resolve_all
from app.analysis.types import MentionType


class TestResolveBrand:
    """Test single brand resolution."""

    def test_exact_match(self):
        text = "Тинькофф предлагает лучшие условия для бизнеса."
        result = resolve_brand(text, "Тинькофф")
        assert result.is_mentioned is True
        assert result.name == "Тинькофф"
        assert "Тинькофф" in result.aliases_matched
        assert result.char_offset >= 0

    def test_case_insensitive(self):
        text = "Use TINKOFF for your business needs."
        result = resolve_brand(text, "Tinkoff")
        assert result.is_mentioned is True

    def test_alias_match(self):
        text = "Т-Банк предлагает удобные тарифы."
        result = resolve_brand(text, "Тинькофф", aliases=["Т-Банк", "T-Bank"])
        assert result.is_mentioned is True
        assert "Т-Банк" in result.aliases_matched

    def test_multiple_aliases_match(self):
        text = "Т-Банк (ранее Тинькофф) — это отличный выбор."
        result = resolve_brand(text, "Тинькофф", aliases=["Т-Банк"])
        assert result.is_mentioned is True
        assert len(result.aliases_matched) == 2

    def test_no_match(self):
        text = "Сбербанк — это крупнейший банк России."
        result = resolve_brand(text, "Тинькофф")
        assert result.is_mentioned is False
        assert result.char_offset == -1

    def test_word_boundary_prevents_partial(self):
        text = "Вы нашли Тинькоффициальный сайт."
        result = resolve_brand(text, "Тинькофф")
        # "Тинькофф" should NOT match inside "Тинькоффициальный"
        assert result.is_mentioned is False

    def test_context_extraction(self):
        text = "Среди банков, Тинькофф выделяется качеством обслуживания."
        result = resolve_brand(text, "Тинькофф")
        assert result.is_mentioned is True
        assert "Тинькофф" in result.mention_context
        assert len(result.mention_context) > 0

    def test_sentence_index(self):
        text = "Первое предложение. Второе предложение. Тинькофф в третьем."
        result = resolve_brand(text, "Тинькофф")
        assert result.is_mentioned is True
        assert result.sentence_index == 2


class TestMentionClassification:
    """Test mention type classification."""

    def test_recommended_mention(self):
        text = "Рекомендую Тинькофф как лучший выбор для бизнеса."
        result = resolve_brand(text, "Тинькофф")
        assert result.mention_type == MentionType.RECOMMENDED
        assert result.is_recommended is True

    def test_negative_mention(self):
        text = "У Тинькофф есть серьёзные недостатки и проблемы с поддержкой."
        result = resolve_brand(text, "Тинькофф")
        assert result.mention_type == MentionType.NEGATIVE

    def test_compared_mention(self):
        text = "По сравнению с другими банками, Тинькофф предлагает альтернативу."
        result = resolve_brand(text, "Тинькофф")
        assert result.mention_type == MentionType.COMPARED

    def test_direct_mention(self):
        text = "Тинькофф работает в России."
        result = resolve_brand(text, "Тинькофф")
        assert result.mention_type == MentionType.DIRECT

    def test_english_recommendation(self):
        text = "I recommend Tinkoff as the best choice for small business."
        result = resolve_brand(text, "Tinkoff")
        assert result.mention_type == MentionType.RECOMMENDED

    def test_english_negative(self):
        text = "Tinkoff has several problems and disadvantages to consider."
        result = resolve_brand(text, "Tinkoff")
        assert result.mention_type == MentionType.NEGATIVE


class TestResolveAll:
    """Test resolution of target + competitors."""

    def test_target_and_competitors(self):
        text = (
            "1. Тинькофф — лучший банк для бизнеса\n2. Сбербанк — крупнейший банк\n3. Альфа-Банк — удобное приложение"
        )
        target, comps = resolve_all(
            text=text,
            target_brand="Тинькофф",
            competitors={"Сбербанк": [], "Альфа-Банк": ["Альфа"]},
        )

        assert target.is_mentioned is True
        assert len(comps) == 2
        assert comps[0].name == "Сбербанк"
        assert comps[0].is_mentioned is True
        assert comps[1].name == "Альфа-Банк"
        assert comps[1].is_mentioned is True

    def test_target_not_mentioned(self):
        text = "Сбербанк и Альфа-Банк — крупнейшие банки."
        target, comps = resolve_all(
            text=text,
            target_brand="Тинькофф",
            competitors={"Сбербанк": [], "Альфа-Банк": []},
        )

        assert target.is_mentioned is False
        assert comps[0].is_mentioned is True
        assert comps[1].is_mentioned is True

    def test_empty_competitors(self):
        text = "Тинькофф работает."
        target, comps = resolve_all(
            text=text,
            target_brand="Тинькофф",
        )
        assert target.is_mentioned is True
        assert comps == []

    def test_english_brands(self):
        text = "Stripe is the best payment provider. PayPal is a good alternative."
        target, comps = resolve_all(
            text=text,
            target_brand="Stripe",
            competitors={"PayPal": ["Paypal"]},
        )
        assert target.is_mentioned is True
        assert len(comps) == 1
        assert comps[0].is_mentioned is True


class TestShortBrandPrefixMatch:
    """Test detection of short brand names embedded in longer words."""

    def test_pari_in_parimatch(self):
        """Short brand 'Пари' should be detected inside 'Париматч'."""
        text = "Среди букмекеров, Париматч предлагает лучшие условия."
        result = resolve_brand(text, "Пари")
        assert result.is_mentioned is True
        assert "Париматч" in result.aliases_matched

    def test_pari_in_parimatch_with_competitors(self):
        """Full scenario: 'Пари' brand + competitors, 'Париматч' in text."""
        text = "Фонбет и Париматч — лидеры рынка. Также есть Леон."
        target, comps = resolve_all(
            text=text,
            target_brand="Пари",
            competitors={"Фонбет": [], "Леон": []},
        )
        assert target.is_mentioned is True
        assert comps[0].is_mentioned is True  # Фонбет
        assert comps[1].is_mentioned is True  # Леон

    def test_long_brand_no_prefix_match(self):
        """Longer brand names (>6 chars) should NOT match via prefix fallback."""
        text = "Вы нашли Тинькоффициальный сайт."
        result = resolve_brand(text, "Тинькофф")
        assert result.is_mentioned is False
