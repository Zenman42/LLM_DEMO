"""Hierarchical Entity Clusterer — Phase 4 of new clustering pipeline.

Implements TaxoAdapt-inspired hierarchical clustering using cosine similarity
of entity profile embeddings (from Phase 3).

Three-level hierarchy:
  Level 3 (cheap): Fuzzy variant matching via _names_match()
  Level 1: Brand-level clusters (loose cosine distance threshold)
  Level 2: Sub-brand groups within each brand cluster (tighter threshold)

Borderline pairs (similarity in the uncertainty zone) are optionally
confirmed via LLM to reduce false merges.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import httpx
import numpy as np

from app.analysis.llm_entity_extractor import _names_match
from app.models.entity_profile import EntityProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Cosine *distance* thresholds (distance = 1 - similarity)
BRAND_THRESHOLD = 0.15  # Level 1: "is this the same brand?"

# Borderline zone for LLM confirmation (cosine distance)
BORDERLINE_LOW = 0.12
BORDERLINE_HIGH = 0.18

# LLM confirmation config
_CONFIRM_MODEL = "gpt-4o-mini"
_CONFIRM_API_URL = "https://api.openai.com/v1/chat/completions"
_CONFIRM_TIMEOUT = 30
_CONFIRM_MAX_CHECKS = 20  # max LLM checks per consolidation run

# Minimum cosine similarity for fuzzy-matched entities to stay grouped.
# If embeddings are available and similarity < this threshold, the fuzzy
# match is considered a false positive and the entity is split out.
# 0.70 means entities must be at least 70% similar (distance < 0.30).
_FUZZY_EMBEDDING_MIN_SIMILARITY = 0.70

_CONFIRM_SYSTEM_PROMPT = """\
You are an entity disambiguation expert. Determine whether two entity \
descriptions refer to the SAME real-world entity (just written differently) \
or DIFFERENT entities.

Output ONLY valid JSON (no markdown):
{"same_entity": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}

Rules:
- "same_entity": true ONLY if both names clearly refer to the same company/brand/product
- Different products of the same parent company are DIFFERENT entities
- A sub-brand and its parent are DIFFERENT entities
- When in doubt, answer false"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EntityCluster:
    """A cluster of entity profiles believed to represent the same entity."""

    members: list[EntityProfile] = field(default_factory=list)
    canonical_name: str = ""
    level: int = 1  # 1=brand, 2=sub-brand, 3=variant
    sub_clusters: list[EntityCluster] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_by_vectors(
    vectors: dict[str, np.ndarray],
    profiles: list[EntityProfile],
    brand_threshold: float = BRAND_THRESHOLD,
) -> list[list[str]]:
    """Cluster entity profiles using hierarchical agglomerative clustering.

    Returns list of clusters, where each cluster is a list of entity_key strings.
    Compatible with the existing consolidator format (list[list[str]]).

    Args:
        vectors: dict mapping entity_key → numpy vector (1536-d).
        profiles: list of EntityProfile ORM objects.
        brand_threshold: cosine distance threshold for brand-level clusters.
    """
    if not profiles:
        return []
    if not vectors:
        # No vectors available — return each profile as a singleton cluster
        return [[p.entity_key] for p in profiles]

    # Build lookup: entity_key → profile
    key_to_profile = {p.entity_key: p for p in profiles}

    # Filter to profiles that have vectors
    keys_with_vectors = [k for k in vectors if k in key_to_profile]

    if not keys_with_vectors:
        # No vectors available — return each profile as singleton
        return [[p.entity_key] for p in profiles]

    if len(keys_with_vectors) == 1:
        # Single entity — also add profiles without vectors as singletons
        result = [[keys_with_vectors[0]]]
        for p in profiles:
            if p.entity_key not in vectors:
                result.append([p.entity_key])
        return result

    # Level 3: Pre-cluster by fuzzy matching (free, no API calls)
    fuzzy_groups = _pre_cluster_fuzzy(keys_with_vectors, key_to_profile)

    logger.info(
        "Fuzzy pre-clustering: %d entities → %d groups",
        len(keys_with_vectors),
        len(fuzzy_groups),
    )

    # Embedding guard: validate fuzzy groups against vector similarity.
    # Fuzzy matching can produce false positives (e.g. "Cloud ML" ↔ "CloudMTS").
    # Split any fuzzy group where a member is too far from the group centroid.
    fuzzy_groups = _validate_fuzzy_groups_with_vectors(fuzzy_groups, vectors)

    logger.info(
        "After embedding validation: %d groups",
        len(fuzzy_groups),
    )

    # Compute representative vectors for each fuzzy group (mean of members)
    group_keys: list[str] = []  # representative key for each group
    group_vectors: list[np.ndarray] = []
    group_members: list[list[str]] = []  # all keys in each group

    for group in fuzzy_groups:
        member_vecs = [vectors[k] for k in group if k in vectors]
        if not member_vecs:
            continue
        rep_vec = np.mean(member_vecs, axis=0)
        group_keys.append(group[0])  # first member as representative
        group_vectors.append(rep_vec)
        group_members.append(group)

    if len(group_vectors) <= 1:
        # Only one group after fuzzy clustering
        result = list(group_members) if group_members else []
        # Add profiles without vectors as singletons
        for p in profiles:
            if p.entity_key not in vectors:
                result.append([p.entity_key])
        return result

    # Level 1: Agglomerative clustering by cosine distance
    brand_clusters = _agglomerative_cluster(
        group_vectors,
        group_members,
        threshold=brand_threshold,
    )

    logger.info(
        "Vector clustering: %d fuzzy groups → %d brand clusters (threshold=%.3f)",
        len(group_members),
        len(brand_clusters),
        brand_threshold,
    )

    # Add profiles without vectors as singletons
    for p in profiles:
        if p.entity_key not in vectors:
            brand_clusters.append([p.entity_key])

    return brand_clusters


def find_borderline_pairs(
    vectors: dict[str, np.ndarray],
    clusters: list[list[str]],
    low: float = BORDERLINE_LOW,
    high: float = BORDERLINE_HIGH,
) -> list[tuple[list[str], list[str], float]]:
    """Find pairs of clusters whose distance falls in the borderline zone.

    Returns list of (cluster_a, cluster_b, distance) tuples for LLM confirmation.
    Only returns up to _CONFIRM_MAX_CHECKS pairs.
    """
    if len(clusters) < 2:
        return []

    # Compute representative vectors per cluster (mean)
    cluster_reps: list[tuple[list[str], np.ndarray]] = []
    for cluster in clusters:
        member_vecs = [vectors[k] for k in cluster if k in vectors]
        if member_vecs:
            rep = np.mean(member_vecs, axis=0)
            cluster_reps.append((cluster, rep))

    # Find borderline pairs
    pairs: list[tuple[list[str], list[str], float]] = []
    for i in range(len(cluster_reps)):
        for j in range(i + 1, len(cluster_reps)):
            c_a, v_a = cluster_reps[i]
            c_b, v_b = cluster_reps[j]
            dist = _cosine_distance(v_a, v_b)
            if low <= dist <= high:
                pairs.append((c_a, c_b, dist))

    # Sort by distance (closest first — most likely to merge)
    pairs.sort(key=lambda x: x[2])
    return pairs[:_CONFIRM_MAX_CHECKS]


async def confirm_borderline_pairs(
    pairs: list[tuple[list[str], list[str], float]],
    profiles: list[EntityProfile],
    api_key: str,
) -> list[list[str]]:
    """Use LLM to confirm whether borderline cluster pairs should merge.

    Returns list of merged clusters (pairs confirmed as same entity).
    Pairs NOT confirmed remain separate (handled by caller).
    """
    if not pairs or not api_key:
        return []

    key_to_profile = {p.entity_key: p for p in profiles}
    merges: list[list[str]] = []

    for cluster_a, cluster_b, dist in pairs:
        # Get representative profiles
        prof_a = key_to_profile.get(cluster_a[0])
        prof_b = key_to_profile.get(cluster_b[0])
        if not prof_a or not prof_b:
            continue

        should_merge = await _llm_confirm_pair(prof_a, prof_b, api_key)
        if should_merge:
            merges.append([cluster_a, cluster_b])
            logger.info(
                "LLM confirmed merge: '%s' + '%s' (dist=%.3f)",
                prof_a.canonical_name,
                prof_b.canonical_name,
                dist,
            )
        else:
            logger.debug(
                "LLM rejected merge: '%s' + '%s' (dist=%.3f)",
                prof_a.canonical_name,
                prof_b.canonical_name,
                dist,
            )

    return merges


def apply_merges(
    clusters: list[list[str]],
    merges: list[list[str]],
) -> list[list[str]]:
    """Apply confirmed merges to cluster list.

    Each merge is a pair [cluster_a_keys, cluster_b_keys] to combine.
    """
    if not merges:
        return clusters

    # Build identity map: key → cluster index
    key_to_idx: dict[str, int] = {}
    for i, cluster in enumerate(clusters):
        for key in cluster:
            key_to_idx[key] = i

    # Union-find for merging
    parent: list[int] = list(range(len(clusters)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for merge_pair in merges:
        if len(merge_pair) != 2:
            continue
        keys_a, keys_b = merge_pair
        # Find cluster indices for first member of each side
        idx_a = key_to_idx.get(keys_a[0] if keys_a else "")
        idx_b = key_to_idx.get(keys_b[0] if keys_b else "")
        if idx_a is not None and idx_b is not None:
            union(idx_a, idx_b)

    # Rebuild clusters from union-find
    merged_groups: dict[int, list[str]] = {}
    for i, cluster in enumerate(clusters):
        root = find(i)
        merged_groups.setdefault(root, []).extend(cluster)

    return list(merged_groups.values())


# ---------------------------------------------------------------------------
# Internal: Embedding guard for fuzzy groups
# ---------------------------------------------------------------------------


def _validate_fuzzy_groups_with_vectors(
    groups: list[list[str]],
    vectors: dict[str, np.ndarray],
    min_similarity: float = _FUZZY_EMBEDDING_MIN_SIMILARITY,
) -> list[list[str]]:
    """Validate fuzzy-matched groups against embedding similarity.

    For each fuzzy group with >1 member, check that every pair has cosine
    similarity >= min_similarity. Members that are too dissimilar are split
    out into their own singleton groups.

    This catches false positives from fuzzy matching (e.g. "Cloud ML" ↔ "CloudMTS")
    where names look similar but semantics differ.
    """
    validated: list[list[str]] = []
    splits = 0

    for group in groups:
        if len(group) <= 1:
            validated.append(group)
            continue

        # Get vectors for group members
        group_with_vecs = [(k, vectors[k]) for k in group if k in vectors]
        group_without_vecs = [k for k in group if k not in vectors]

        if len(group_with_vecs) <= 1:
            # Can't validate without vectors — keep as-is
            validated.append(group)
            continue

        # Find the anchor (first member) and check each other member against it
        # Use pairwise minimum: a member stays if it's similar enough to at
        # least one other member already in the validated subgroup.
        kept = [group_with_vecs[0]]
        ejected: list[str] = []

        for k, vec in group_with_vecs[1:]:
            # Check similarity against all kept members
            max_sim = max(
                float(np.dot(vec, kept_vec) / (np.linalg.norm(vec) * np.linalg.norm(kept_vec) + 1e-10))
                for _, kept_vec in kept
            )

            if max_sim >= min_similarity:
                kept.append((k, vec))
            else:
                ejected.append(k)
                splits += 1
                logger.debug(
                    "Embedding guard: ejected '%s' from fuzzy group (max_sim=%.3f < %.3f)",
                    k,
                    max_sim,
                    min_similarity,
                )

        # Rebuild the group from kept members + members without vectors
        if group_without_vecs:
            logger.debug(
                "Embedding guard: %d members without vectors in fuzzy group, keeping unvalidated",
                len(group_without_vecs),
            )
        validated_group = [k for k, _ in kept] + group_without_vecs
        validated.append(validated_group)

        # Each ejected member becomes a singleton
        for k in ejected:
            validated.append([k])

    if splits:
        logger.info(
            "Embedding guard: split %d entities from fuzzy groups (min_similarity=%.2f)",
            splits,
            min_similarity,
        )

    return validated


# ---------------------------------------------------------------------------
# Internal: Fuzzy pre-clustering (Level 3)
# ---------------------------------------------------------------------------


# Entity types that are fundamentally incompatible and should never merge
# via fuzzy matching alone (would need explicit vector/LLM confirmation).
_INCOMPATIBLE_TYPES: set[frozenset[str]] = {
    frozenset({"location", "company"}),
    frozenset({"location", "product"}),
    frozenset({"location", "service"}),
    frozenset({"location", "technology"}),
    frozenset({"location", "organization"}),
    frozenset({"person", "company"}),
    frozenset({"person", "product"}),
    frozenset({"person", "service"}),
    frozenset({"person", "location"}),
}


def _entity_types_compatible(type_a: str | None, type_b: str | None) -> bool:
    """Check if two entity types are compatible for fuzzy merging.

    Returns True if the types are compatible (merge allowed),
    False if they are fundamentally incompatible.
    Unknown types (None) are always considered compatible.
    """
    if not type_a or not type_b:
        return True  # unknown type — allow merge

    pair = frozenset({type_a.lower().strip(), type_b.lower().strip()})
    return pair not in _INCOMPATIBLE_TYPES


def _pre_cluster_fuzzy(
    entity_keys: list[str],
    key_to_profile: dict[str, EntityProfile],
) -> list[list[str]]:
    """Pre-cluster entities using _names_match (free, no API calls).

    Uses canonical_name from profiles for matching.
    Also checks entity_type compatibility to prevent merging
    fundamentally different entities (e.g. location + product).
    """
    used: set[str] = set()
    groups: list[list[str]] = []

    for i, key_a in enumerate(entity_keys):
        if key_a in used:
            continue

        group = [key_a]
        used.add(key_a)
        prof_a = key_to_profile.get(key_a)
        if not prof_a:
            groups.append(group)
            continue

        for j in range(i + 1, len(entity_keys)):
            key_b = entity_keys[j]
            if key_b in used:
                continue
            prof_b = key_to_profile.get(key_b)
            if not prof_b:
                continue

            # Guard: incompatible entity types should not be merged
            if not _entity_types_compatible(prof_a.entity_type, prof_b.entity_type):
                continue

            # Compare canonical names using existing fuzzy matcher
            if _names_match(prof_a.canonical_name, prof_b.canonical_name):
                group.append(key_b)
                used.add(key_b)

        groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# Internal: Agglomerative clustering (Level 1)
# ---------------------------------------------------------------------------


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return float(1.0 - dot / norm)


def _agglomerative_cluster(
    vectors: list[np.ndarray],
    members: list[list[str]],
    threshold: float,
) -> list[list[str]]:
    """Agglomerative hierarchical clustering using scipy.

    Falls back to manual pairwise clustering if scipy is not available.
    """
    n = len(vectors)
    if n <= 1:
        return list(members)

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        # Compute pairwise cosine distances
        vec_matrix = np.array(vectors, dtype=np.float32)

        # Normalise vectors for cosine distance
        norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vec_matrix = vec_matrix / norms

        distances = pdist(vec_matrix, metric="cosine")

        # Hierarchical clustering with average linkage
        Z = linkage(distances, method="average")

        # Cut dendrogram at threshold
        labels = fcluster(Z, t=threshold, criterion="distance")

        # Group members by cluster label
        label_groups: dict[int, list[str]] = {}
        for i, label in enumerate(labels):
            label_groups.setdefault(int(label), []).extend(members[i])

        return list(label_groups.values())

    except ImportError:
        logger.warning("scipy not available, falling back to manual clustering")
        return _manual_agglomerative(vectors, members, threshold)


def _manual_agglomerative(
    vectors: list[np.ndarray],
    members: list[list[str]],
    threshold: float,
) -> list[list[str]]:
    """Simple greedy agglomerative clustering without scipy.

    O(n^2) pairwise comparison — fine for typical entity counts (<1000).
    """
    n = len(vectors)
    # Compute full distance matrix
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _cosine_distance(vectors[i], vectors[j])
            dist[i][j] = d
            dist[j][i] = d

    # Greedy merge: repeatedly merge closest pair under threshold
    parent = list(range(n))
    cluster_members: list[list[str]] = [list(m) for m in members]
    cluster_vecs: list[np.ndarray] = [v.copy() for v in vectors]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active = set(range(n))

    while len(active) > 1:
        # Find closest pair among active clusters
        best_dist = float("inf")
        best_i, best_j = -1, -1

        active_list = sorted(active)
        for ii in range(len(active_list)):
            for jj in range(ii + 1, len(active_list)):
                i, j = active_list[ii], active_list[jj]
                d = _cosine_distance(cluster_vecs[i], cluster_vecs[j])
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        if best_dist > threshold or best_i < 0:
            break

        # Merge best_j into best_i
        cluster_members[best_i].extend(cluster_members[best_j])
        # Update representative vector (mean of all member vectors)
        all_member_vecs = []
        for key in cluster_members[best_i]:
            for orig_idx, orig_members in enumerate(members):
                if key in orig_members:
                    all_member_vecs.append(vectors[orig_idx])
                    break
        if all_member_vecs:
            cluster_vecs[best_i] = np.mean(all_member_vecs, axis=0)

        parent[best_j] = best_i
        active.discard(best_j)

    return [cluster_members[i] for i in active]


# ---------------------------------------------------------------------------
# Internal: LLM confirmation for borderline pairs
# ---------------------------------------------------------------------------


async def _llm_confirm_pair(
    prof_a: EntityProfile,
    prof_b: EntityProfile,
    api_key: str,
) -> bool:
    """Ask LLM whether two entity profiles refer to the same entity.

    Returns True if LLM confirms they are the same entity with high confidence.
    """
    desc_a = _profile_summary(prof_a)
    desc_b = _profile_summary(prof_b)

    user_prompt = (
        f'Entity A: "{prof_a.canonical_name}"\n'
        f"Description: {desc_a}\n\n"
        f'Entity B: "{prof_b.canonical_name}"\n'
        f"Description: {desc_b}\n\n"
        "Are these the SAME real-world entity? Answer with JSON."
    )

    payload = {
        "model": _CONFIRM_MODEL,
        "messages": [
            {"role": "system", "content": _CONFIRM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    try:
        async with httpx.AsyncClient(timeout=_CONFIRM_TIMEOUT) as client:
            resp = await client.post(
                _CONFIRM_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"] or ""

        import re

        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if json_match:
            raw = json_match.group()

        parsed = json.loads(raw)
        same = parsed.get("same_entity", False)
        confidence = parsed.get("confidence", 0.0)

        # Only merge if high confidence
        if same and confidence >= 0.8:
            return True
        return False

    except Exception as e:
        logger.warning("LLM confirmation failed for '%s' vs '%s': %s", prof_a.canonical_name, prof_b.canonical_name, e)
        return False


def _profile_summary(p: EntityProfile) -> str:
    """Build a short text summary of an entity profile for LLM confirmation."""
    parts = []
    if p.entity_type:
        parts.append(f"Type: {p.entity_type}")
    if p.industry:
        parts.append(f"Industry: {p.industry}")
    if p.description:
        parts.append(p.description)
    if p.attributes and isinstance(p.attributes, dict):
        for k, v in list(p.attributes.items())[:5]:
            if not k.startswith("_"):  # skip internal metadata keys
                parts.append(f"{k}: {v}")
    return " | ".join(parts) if parts else "(no description)"
