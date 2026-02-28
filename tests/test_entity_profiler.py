"""Tests for entity_profiler module."""

from app.analysis.entity_profiler import (
    EntityMention,
    _aggregate_mentions,
    _normalise_key,
    parse_entity_mentions,
)


def test_normalise_key_basic():
    assert _normalise_key("  Сбербанк  ") == "сбербанк"
    assert _normalise_key("Яндекс.Карты") == "яндекскарты"
    assert _normalise_key("VK,Inc") == "vkinc"


def test_normalise_key_preserves_case_insensitive():
    assert _normalise_key("Google") == _normalise_key("google")
    assert _normalise_key("PARI") == _normalise_key("pari")


def test_aggregate_mentions_groups_by_key():
    mentions = [
        EntityMention(name="Сбербанк", context="ctx1", snapshot_id=1),
        EntityMention(name="сбербанк", context="ctx2", snapshot_id=2),
        EntityMention(name="Yandex", context="ctx3", snapshot_id=1),
    ]
    groups = _aggregate_mentions(mentions)
    assert "сбербанк" in groups
    assert len(groups["сбербанк"]) == 2
    assert "yandex" in groups
    assert len(groups["yandex"]) == 1


def test_aggregate_mentions_skips_empty_names():
    mentions = [
        EntityMention(name="", context="empty"),
        EntityMention(name="  ", context="spaces"),
        EntityMention(name="Valid", context="ok"),
    ]
    groups = _aggregate_mentions(mentions)
    assert len(groups) == 1
    assert "valid" in groups


def test_parse_entity_mentions_old_format():
    """Test parsing old format: list[str]."""
    rows = [
        (100, ["Сбербанк", "Тинькофф", "VK"]),
        (101, ["Google", "Яндекс"]),
    ]
    result = parse_entity_mentions(rows)
    assert len(result) == 5
    assert result[0].name == "Сбербанк"
    assert result[0].snapshot_id == 100
    assert result[0].position == 0
    assert result[0].context == ""


def test_parse_entity_mentions_new_format():
    """Test parsing new enriched format: list[dict]."""
    rows = [
        (
            200,
            [
                {"name": "Сбербанк", "context": "Сбербанк предлагает...", "type": "company"},
                {"name": "СберМаркет", "context": "СберМаркет — доставка...", "type": "product"},
            ],
        ),
    ]
    result = parse_entity_mentions(rows)
    assert len(result) == 2
    assert result[0].name == "Сбербанк"
    assert result[0].context == "Сбербанк предлагает..."
    assert result[0].entity_type == "company"
    assert result[0].snapshot_id == 200


def test_parse_entity_mentions_mixed_format():
    """Test parsing with mixed old/new format across snapshots."""
    rows = [
        (100, ["Сбербанк", "VK"]),  # old
        (200, [{"name": "Google", "context": "ctx", "type": "company"}]),  # new
    ]
    result = parse_entity_mentions(rows)
    assert len(result) == 3


def test_parse_entity_mentions_empty():
    assert parse_entity_mentions([]) == []
    assert parse_entity_mentions([(1, None)]) == []
    assert parse_entity_mentions([(1, [])]) == []
