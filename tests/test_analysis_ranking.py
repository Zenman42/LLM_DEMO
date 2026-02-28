"""Tests for Structural & Ranking Parser."""

from app.analysis.ranking_parser import (
    analyze_structure,
    detect_structure,
    position_weight_bullet,
    position_weight_narrative,
    position_weight_numbered,
)
from app.analysis.types import BrandMention, StructureType


class TestPositionWeights:
    """Test position weight calculations."""

    def test_numbered_rank_1(self):
        assert position_weight_numbered(1) == 1.0

    def test_numbered_rank_2(self):
        assert position_weight_numbered(2) == 0.9

    def test_numbered_rank_5(self):
        assert position_weight_numbered(5) == 0.6

    def test_numbered_rank_10(self):
        assert position_weight_numbered(10) == 0.1

    def test_numbered_rank_15(self):
        assert position_weight_numbered(15) == 0.1  # Capped at 0.1

    def test_numbered_rank_0(self):
        assert position_weight_numbered(0) == 0.0

    def test_bullet_weight(self):
        assert position_weight_bullet() == 0.8

    def test_narrative_start(self):
        assert position_weight_narrative(0, 1000) == 1.0

    def test_narrative_first_20pct(self):
        assert position_weight_narrative(100, 1000) == 1.0

    def test_narrative_middle(self):
        assert position_weight_narrative(350, 1000) == 0.7

    def test_narrative_late(self):
        assert position_weight_narrative(650, 1000) == 0.4

    def test_narrative_end(self):
        assert position_weight_narrative(900, 1000) == 0.2

    def test_narrative_zero_length(self):
        assert position_weight_narrative(0, 0) == 0.5


class TestStructureDetection:
    """Test structure type detection."""

    def test_numbered_list(self):
        text = "1. Тинькофф\n2. Сбербанк\n3. Альфа-Банк"
        assert detect_structure(text) == StructureType.NUMBERED_LIST

    def test_numbered_with_parens(self):
        text = "1) Тинькофф\n2) Сбербанк\n3) Альфа-Банк"
        assert detect_structure(text) == StructureType.NUMBERED_LIST

    def test_bulleted_list_dash(self):
        text = "- Тинькофф\n- Сбербанк\n- Альфа-Банк"
        assert detect_structure(text) == StructureType.BULLETED_LIST

    def test_bulleted_list_asterisk(self):
        text = "* Тинькофф\n* Сбербанк\n* Альфа-Банк"
        assert detect_structure(text) == StructureType.BULLETED_LIST

    def test_bulleted_list_bullet(self):
        text = "• Тинькофф\n• Сбербанк\n• Альфа-Банк"
        assert detect_structure(text) == StructureType.BULLETED_LIST

    def test_narrative(self):
        text = "Тинькофф — это крупный российский банк, который предлагает различные услуги."
        assert detect_structure(text) == StructureType.NARRATIVE

    def test_table(self):
        text = "| Банк | Тариф |\n|---|---|\n| Тинькофф | 0 руб |\n| Сбербанк | 500 руб |"
        assert detect_structure(text) == StructureType.TABLE

    def test_mixed_numbered_and_bulleted(self):
        text = "1. Первая группа:\n- Тинькофф\n- Сбербанк\n2. Вторая группа:\n- Альфа"
        assert detect_structure(text) == StructureType.MIXED

    def test_single_numbered_item_is_narrative(self):
        text = "1. Единственный пункт в тексте."
        assert detect_structure(text) == StructureType.NARRATIVE


class TestAnalyzeStructure:
    """Test full structure analysis with brand position weights."""

    def test_numbered_list_assigns_weights(self):
        text = "1. Тинькофф — лучший\n2. Сбербанк — крупнейший\n3. Альфа-Банк — удобный"

        brands = [
            BrandMention(name="Тинькофф", is_mentioned=True, char_offset=3),
            BrandMention(name="Сбербанк", is_mentioned=True, char_offset=30),
            BrandMention(name="Альфа-Банк", is_mentioned=True, char_offset=55),
        ]

        result = analyze_structure(text, brands)

        assert result.structure_type == StructureType.NUMBERED_LIST
        assert result.total_items == 3
        assert brands[0].position_rank == 1
        assert brands[0].position_weight == 1.0
        assert brands[1].position_rank == 2
        assert brands[1].position_weight == 0.9
        assert brands[2].position_rank == 3
        assert brands[2].position_weight == 0.8

    def test_bulleted_list_uniform_weight(self):
        text = "- Тинькофф\n- Сбербанк\n- Альфа-Банк"

        brands = [
            BrandMention(name="Тинькофф", is_mentioned=True, char_offset=2),
            BrandMention(name="Сбербанк", is_mentioned=True, char_offset=15),
        ]

        result = analyze_structure(text, brands)
        assert result.structure_type == StructureType.BULLETED_LIST
        assert brands[0].position_weight == 0.8
        assert brands[1].position_weight == 0.8

    def test_narrative_proximity_weights(self):
        text = "Тинькофф is mentioned at the beginning. " + "x" * 500 + " Сбербанк appears much later."

        brands = [
            BrandMention(name="Тинькофф", is_mentioned=True, char_offset=0),
            BrandMention(name="Сбербанк", is_mentioned=True, char_offset=540),
        ]

        result = analyze_structure(text, brands)
        assert result.structure_type == StructureType.NARRATIVE
        # Тинькофф is early → high weight
        assert brands[0].position_weight >= 0.7
        # Сбербанк is late → lower weight
        assert brands[1].position_weight < brands[0].position_weight

    def test_unmentioned_brand_gets_zero(self):
        text = "1. Тинькофф\n2. Сбербанк"

        brands = [
            BrandMention(name="Тинькофф", is_mentioned=True, char_offset=3),
            BrandMention(name="Альфа-Банк", is_mentioned=False),
        ]

        analyze_structure(text, brands)
        assert brands[1].position_rank == 0
        assert brands[1].position_weight == 0.0

    def test_brands_in_list_ordered(self):
        text = "1. Тинькофф — лучший\n2. Сбербанк — крупнейший"
        brands = [
            BrandMention(name="Тинькофф", is_mentioned=True, char_offset=3),
            BrandMention(name="Сбербанк", is_mentioned=True, char_offset=25),
        ]
        result = analyze_structure(text, brands)
        assert "Тинькофф" in result.brands_in_list
        assert "Сбербанк" in result.brands_in_list
