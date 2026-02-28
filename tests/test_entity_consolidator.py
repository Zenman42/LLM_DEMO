"""Tests for entity consolidation cluster guards."""

from app.analysis.entity_consolidator import _split_overbroad_known_clusters


def test_split_overbroad_known_cluster_separates_unknown_members():
    clusters = [["MWS AI", "ML Space", "CloudMTS", "VK Cloud Disk"]]

    result = _split_overbroad_known_clusters(
        clusters=clusters,
        brand_name="MTS",
        brand_aliases=["МТС"],
        competitors=["VK", "Сбер"],
        brand_sub_brands={"MWS AI": []},
        competitor_sub_brands={},
        competitor_aliases={},
    )

    assert result == [["MWS AI"], ["ML Space"], ["CloudMTS"], ["VK Cloud Disk"]]


def test_split_overbroad_known_cluster_keeps_pure_single_root_cluster():
    clusters = [["MWS AI", "MWS.AI"]]

    result = _split_overbroad_known_clusters(
        clusters=clusters,
        brand_name="MTS",
        brand_aliases=["МТС"],
        competitors=["VK", "Сбер"],
        brand_sub_brands={"MWS AI": []},
        competitor_sub_brands={},
        competitor_aliases={},
    )

    assert result == [["MWS AI", "MWS.AI"]]


def test_split_overbroad_known_cluster_splits_multiple_known_roots():
    clusters = [["MWS AI", "SberCloud.AI", "VK Cloud"]]

    result = _split_overbroad_known_clusters(
        clusters=clusters,
        brand_name="MTS",
        brand_aliases=["МТС"],
        competitors=["Сбер", "VK"],
        brand_sub_brands={"MWS AI": []},
        competitor_sub_brands={},
        competitor_aliases={"Сбер": ["SberCloud"], "VK": ["VK Cloud"]},
    )

    assert result == [["MWS AI"], ["SberCloud.AI"], ["VK Cloud"]]


def test_split_overbroad_known_cluster_keeps_discovered_only_cluster():
    clusters = [["ML Space", "ML Platform"]]

    result = _split_overbroad_known_clusters(
        clusters=clusters,
        brand_name="MTS",
        brand_aliases=["МТС"],
        competitors=["VK", "Сбер"],
        brand_sub_brands={"MWS AI": []},
        competitor_sub_brands={},
        competitor_aliases={},
    )

    assert result == [["ML Space", "ML Platform"]]
