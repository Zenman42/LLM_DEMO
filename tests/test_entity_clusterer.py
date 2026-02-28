"""Tests for entity_clusterer module."""

import numpy as np
import pytest

from app.analysis.entity_clusterer import (
    _cosine_distance,
    _manual_agglomerative,
    _pre_cluster_fuzzy,
    apply_merges,
    cluster_by_vectors,
    find_borderline_pairs,
)


# ---------------------------------------------------------------------------
# Helper: minimal EntityProfile-like object for tests
# ---------------------------------------------------------------------------


class FakeProfile:
    """Minimal profile stub for testing (avoids DB model import issues)."""

    def __init__(self, entity_key: str, canonical_name: str):
        self.entity_key = entity_key
        self.canonical_name = canonical_name
        self.entity_type = "company"
        self.industry = ""
        self.description = ""
        self.attributes = {}
        self.embedding = None


# ---------------------------------------------------------------------------
# Cosine distance
# ---------------------------------------------------------------------------


def test_cosine_distance_identical_vectors():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert _cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert _cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)


def test_cosine_distance_opposite_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert _cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)


def test_cosine_distance_zero_vector():
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert _cosine_distance(a, b) == 1.0


# ---------------------------------------------------------------------------
# Fuzzy pre-clustering
# ---------------------------------------------------------------------------


def test_pre_cluster_fuzzy_groups_transliterations():
    """Fuzzy clustering should group Cyrillic/Latin variants."""
    profiles = {
        "сбербанк": FakeProfile("сбербанк", "Сбербанк"),
        "sberbank": FakeProfile("sberbank", "Sberbank"),
        "google": FakeProfile("google", "Google"),
    }
    groups = _pre_cluster_fuzzy(list(profiles.keys()), profiles)
    # Сбербанк and Sberbank should be in one group
    assert len(groups) == 2
    sber_group = [g for g in groups if "сбербанк" in g][0]
    assert "sberbank" in sber_group


def test_pre_cluster_fuzzy_keeps_different_entities_separate():
    profiles = {
        "google": FakeProfile("google", "Google"),
        "яндекс": FakeProfile("яндекс", "Яндекс"),
        "vk": FakeProfile("vk", "VK"),
    }
    groups = _pre_cluster_fuzzy(list(profiles.keys()), profiles)
    assert len(groups) == 3  # All different entities


# ---------------------------------------------------------------------------
# Manual agglomerative clustering
# ---------------------------------------------------------------------------


def test_manual_agglomerative_merges_close_vectors():
    # Two very similar vectors and one distant
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.99, 0.1, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    vectors = [v1, v2, v3]
    members = [["a"], ["b"], ["c"]]

    result = _manual_agglomerative(vectors, members, threshold=0.15)
    # v1 and v2 should merge, v3 stays separate
    assert len(result) == 2
    merged = [c for c in result if len(c) > 1]
    assert len(merged) == 1
    assert set(merged[0]) == {"a", "b"}


def test_manual_agglomerative_no_merge_above_threshold():
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)

    vectors = [v1, v2]
    members = [["a"], ["b"]]

    result = _manual_agglomerative(vectors, members, threshold=0.5)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Full cluster_by_vectors
# ---------------------------------------------------------------------------


def test_cluster_by_vectors_empty():
    assert cluster_by_vectors({}, []) == []


def test_cluster_by_vectors_single_entity():
    p = FakeProfile("test", "Test")
    v = np.array([1.0] * 10, dtype=np.float32)
    result = cluster_by_vectors({"test": v}, [p])
    assert len(result) == 1
    assert result[0] == ["test"]


def test_cluster_by_vectors_merges_similar():
    # Two profiles with nearly identical vectors should cluster
    p1 = FakeProfile("apple", "Apple")
    p2 = FakeProfile("apple inc", "Apple Inc")
    p3 = FakeProfile("google", "Google")

    dim = 32
    v1 = np.random.RandomState(42).randn(dim).astype(np.float32)
    v2 = v1 + np.random.RandomState(43).randn(dim).astype(np.float32) * 0.01  # very close
    v3 = np.random.RandomState(44).randn(dim).astype(np.float32)  # different

    vectors = {"apple": v1, "apple inc": v2, "google": v3}
    profiles = [p1, p2, p3]

    result = cluster_by_vectors(vectors, profiles, brand_threshold=0.15)
    # apple and apple inc should cluster (very close vectors)
    merged = [c for c in result if len(c) > 1]
    assert len(merged) >= 1


# ---------------------------------------------------------------------------
# find_borderline_pairs
# ---------------------------------------------------------------------------


def test_find_borderline_pairs_detects_borderline():
    dim = 32
    rng = np.random.RandomState(42)
    v1 = rng.randn(dim).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    # Create a vector at ~0.15 cosine distance
    v2 = v1.copy()
    v2[:5] += 0.3
    v2 /= np.linalg.norm(v2)

    vectors = {"a": v1, "b": v2}
    clusters = [["a"], ["b"]]

    dist = _cosine_distance(v1, v2)
    pairs = find_borderline_pairs(vectors, clusters, low=dist - 0.01, high=dist + 0.01)
    assert len(pairs) == 1


def test_find_borderline_pairs_empty_clusters():
    assert find_borderline_pairs({}, []) == []
    assert find_borderline_pairs({"a": np.zeros(3)}, [["a"]]) == []


# ---------------------------------------------------------------------------
# apply_merges
# ---------------------------------------------------------------------------


def test_apply_merges_combines_clusters():
    clusters = [["a", "b"], ["c"], ["d", "e"]]
    merges = [[["a", "b"], ["c"]]]
    result = apply_merges(clusters, merges)
    assert len(result) == 2
    merged = [c for c in result if len(c) > 2]
    assert len(merged) == 1
    assert set(merged[0]) == {"a", "b", "c"}


def test_apply_merges_no_merges():
    clusters = [["a"], ["b"], ["c"]]
    result = apply_merges(clusters, [])
    assert result == clusters


def test_apply_merges_chain():
    """Multiple merges that chain together."""
    clusters = [["a"], ["b"], ["c"]]
    merges = [[["a"], ["b"]], [["b"], ["c"]]]
    result = apply_merges(clusters, merges)
    # All should end up in one cluster
    assert len(result) == 1
    assert set(result[0]) == {"a", "b", "c"}
