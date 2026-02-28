"""Post-collection Entity Consolidation via ProLEA + TaxoAdapt Pipeline (Phase 3).

After all LLM providers have been queried (Phase 1) and per-snapshot analysis
is done (Phase 2), this module consolidates ALL extracted entities across
today's snapshots for a project.

New pipeline (replaces old LLM-based clustering):
  3a. SELECT all today's raw_entities from llm_snapshots
  3b. Parse enriched entity mentions (name + context + type)
  3c. Build semantic profiles via LLM (ProLEA Phase 2)
  3d. Vectorize profiles via OpenAI embeddings (Phase 3)
  3e. Hierarchical clustering by cosine similarity (TaxoAdapt Phase 4)
  3f. LLM confirmation for borderline pairs
  3g. Guard against overbroad clusters
  3h. Pick canonical names
  3i. Match clusters to brand/competitors
  3j. UPDATE competitor_mentions and brand_mentioned in snapshots
  3k. Rewrite discovered_entities with canonical names
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from uuid import UUID

import httpx
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.entity_clusterer import (
    apply_merges,
    cluster_by_vectors,
    confirm_borderline_pairs,
    find_borderline_pairs,
)
from app.analysis.entity_profiler import (
    build_entity_profiles,
    parse_entity_mentions,
)
from app.analysis.entity_vectorizer import vectorize_profiles
from app.analysis.llm_entity_extractor import _names_match, get_entity_names
from app.models.discovered_entity import DiscoveredEntity
from app.models.llm_query import LlmQuery
from app.models.llm_snapshot import LlmSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of post-collection entity consolidation."""

    total_raw_entities: int = 0
    unique_entities: int = 0
    clusters_found: int = 0
    entities_merged: int = 0  # unique entities that were grouped with others
    brand_matches: int = 0  # clusters matched to target brand
    competitor_matches: int = 0  # clusters matched to known competitors
    new_discovered: int = 0  # clusters that are new (not brand/competitor)
    snapshots_updated: int = 0


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_post_collection_consolidation(
    db: AsyncSession,
    project_id: int,
    tenant_id: UUID,
    today: date,
    brand_name: str,
    brand_aliases: list[str],
    competitors: list[str],
    openai_api_key: str,
    rpm_limiter=None,
    tpm_limiter=None,
    gemini_api_key: str | None = None,
    brand_sub_brands: dict[str, list[str]] | None = None,
    competitor_sub_brands: dict[str, dict[str, list[str]]] | None = None,
    perplexity_api_key: str | None = None,
) -> ConsolidationResult:
    """Run Phase 3 entity consolidation for a project's today's snapshots.

    New pipeline (ProLEA + TaxoAdapt):
      1. Load all raw_entities from today's snapshots
      2. Parse enriched entity mentions (name + context + type)
      3. Build/update semantic profiles via LLM (ProLEA)
      4. Vectorize profiles via OpenAI embeddings
      5. Hierarchical clustering by cosine similarity (TaxoAdapt)
      6. LLM confirmation for borderline pairs
      7. Guard against overbroad clusters
      8. Pick canonical names
      9. Match clusters to brand/competitors
      10. Update snapshots with corrected competitor_mentions / brand_mentioned
      11. Rewrite discovered_entities with canonical names
    """
    result = ConsolidationResult()

    # Step 1: Load all raw_entities from today's snapshots for this project
    stmt = (
        select(LlmSnapshot.id, LlmSnapshot.raw_entities)
        .join(LlmQuery, LlmQuery.id == LlmSnapshot.llm_query_id)
        .where(
            LlmQuery.project_id == project_id,
            LlmSnapshot.tenant_id == tenant_id,
            LlmSnapshot.date == today,
            LlmSnapshot.raw_entities.isnot(None),
        )
    )
    rows = (await db.execute(stmt)).all()

    if not rows:
        logger.info("Consolidation: no raw_entities found for project=%d date=%s", project_id, today)
        return result

    # Build per-snapshot entity lists and global frequency map
    # Handle both old format (list[str]) and new format (list[dict])
    snapshot_entities: dict[int, list[str]] = {}  # snapshot_id → [entity_name, ...]
    freq: Counter = Counter()

    for snapshot_id, raw_ents in rows:
        if not raw_ents:
            continue
        # Extract plain entity names from both formats
        names = get_entity_names(raw_ents)
        snapshot_entities[snapshot_id] = names
        for name in names:
            freq[name] += 1

    result.total_raw_entities = sum(len(v) for v in snapshot_entities.values())
    result.unique_entities = len(freq)

    if not freq:
        return result

    logger.info(
        "Consolidation: project=%d, total_raw=%d, unique=%d",
        project_id,
        result.total_raw_entities,
        result.unique_entities,
    )

    # Step 2: Parse enriched entity mentions (for profiling)
    snapshot_rows_for_mentions = [(snapshot_id, raw_ents) for snapshot_id, raw_ents in rows if raw_ents]
    mentions = parse_entity_mentions(snapshot_rows_for_mentions)

    # Step 3: Build/update semantic profiles (ProLEA Phase 2)
    # Guard: the full ProLEA pipeline requires an OpenAI API key for
    # profiling (GPT-4o-mini) and embeddings (text-embedding-3-small).
    # Gemini-only tenants get graceful degradation to fuzzy-only clustering.
    if openai_api_key:
        profiles = await build_entity_profiles(
            mentions=mentions,
            db=db,
            project_id=project_id,
            tenant_id=tenant_id,
            openai_api_key=openai_api_key,
            perplexity_api_key=perplexity_api_key,
            today=today,
        )
    else:
        logger.warning(
            "Consolidation: openai_api_key not available, falling back to fuzzy-only clustering. "
            "Full ProLEA pipeline requires OpenAI API key for profiling and embeddings."
        )
        profiles = []

    if profiles:
        logger.info("Profiler produced %d entity profiles", len(profiles))

        # Step 4: Vectorize profiles (Phase 3)
        vectors = await vectorize_profiles(
            profiles=profiles,
            api_key=openai_api_key,
            db=db,
        )

        # Step 5: Hierarchical clustering by cosine similarity (Phase 4)
        clusters_by_key = cluster_by_vectors(
            vectors=vectors,
            profiles=profiles,
        )

        # Step 6: LLM confirmation for borderline pairs
        borderline = find_borderline_pairs(vectors, clusters_by_key)
        if borderline:
            logger.info("Found %d borderline pairs for LLM confirmation", len(borderline))
            merges = await confirm_borderline_pairs(borderline, profiles, openai_api_key)
            if merges:
                clusters_by_key = apply_merges(clusters_by_key, merges)

        # Convert from entity_key-based clusters to entity_name-based clusters
        # (downstream code operates on raw entity names from snapshots)

        # Build mapping: entity_key → all raw entity names that normalise to this key
        from app.analysis.entity_profiler import _normalise_key

        name_to_key: dict[str, str] = {}
        for name in freq:
            key = _normalise_key(name)
            name_to_key[name] = key

        # Convert key-based clusters to name-based clusters
        clusters: list[list[str]] = []
        for key_cluster in clusters_by_key:
            name_cluster = []
            for key in key_cluster:
                # Find all raw names that map to this entity_key
                for name, mapped_key in name_to_key.items():
                    if mapped_key == key:
                        name_cluster.append(name)
            if name_cluster:
                clusters.append(name_cluster)

        # Add any names that weren't matched to a profile (edge case)
        clustered_names = {n for c in clusters for n in c}
        for name in freq:
            if name not in clustered_names:
                clusters.append([name])
    else:
        # Fallback: no profiles generated, use fuzzy-only clustering
        logger.warning("No profiles generated, falling back to fuzzy-only clustering")
        clusters = _fuzzy_only_clusters(list(freq.items()))

    # Step 7a: Reset old alias relationships for entities seen today,
    # BEFORE loading competitor aliases.  This prevents a feedback loop:
    # old incorrect aliases from previous (possibly buggy) runs would
    # otherwise perpetuate wrong merges.
    # SCOPED: only reset aliases for entity names in today's collection run.
    # Historical correct aliases for entities not seen today are preserved.
    today_entity_names = list(freq.keys())
    if today_entity_names:
        reset_count_result = await db.execute(
            update(DiscoveredEntity)
            .where(
                DiscoveredEntity.project_id == project_id,
                DiscoveredEntity.status == "alias",
                DiscoveredEntity.entity_name.in_(today_entity_names),
            )
            .values(alias_of_id=None, status="pending")
        )
        reset_count = reset_count_result.rowcount
        if reset_count:
            logger.info("Reset %d old alias relationships for today's entities", reset_count)
        await db.flush()  # ensure reset is visible to subsequent queries

    # Step 7b: Enrich sub-brand dicts with alias names from entity relationships
    brand_sub_brands, competitor_sub_brands = await _enrich_sub_brands_from_db(
        db,
        project_id,
        brand_sub_brands or {},
        competitor_sub_brands or {},
    )

    # Load competitor entity aliases for matching (now clean after reset)
    competitor_aliases = await _load_competitor_aliases(db, project_id, competitors)

    # Step 7b: Guard against overbroad clusters
    clusters = _split_overbroad_known_clusters(
        clusters=clusters,
        brand_name=brand_name,
        brand_aliases=brand_aliases,
        competitors=competitors,
        brand_sub_brands=brand_sub_brands,
        competitor_sub_brands=competitor_sub_brands,
        competitor_aliases=competitor_aliases,
    )

    result.clusters_found = len(clusters)
    result.entities_merged = sum(1 for c in clusters if len(c) > 1 for _ in c[1:])

    # Step 8: Pick canonical name for each cluster
    canonical_map: dict[str, str] = {}  # variant -> canonical
    for cluster in clusters:
        canonical = _pick_canonical(cluster, freq)
        for variant in cluster:
            canonical_map[variant] = canonical

    # Step 9: Match clusters to brand/competitors
    brand_canonical: set[str] = set()  # canonical names matched to target brand
    competitor_canonical: dict[str, str] = {}  # canonical → competitor_name
    discovered_canonical: list[str] = []  # canonical names not matched to anything

    for cluster in clusters:
        canonical = canonical_map[cluster[0]]
        match_result = _match_cluster_to_known(
            cluster=cluster,
            brand_name=brand_name,
            brand_aliases=brand_aliases,
            competitors=competitors,
            brand_sub_brands=brand_sub_brands,
            competitor_sub_brands=competitor_sub_brands,
            competitor_aliases=competitor_aliases,
        )
        if match_result == "__brand__":
            brand_canonical.add(canonical)
            result.brand_matches += 1
        elif match_result is not None and match_result.startswith("__brand__::"):
            # Sub-brand of main brand — counts as brand match AND records sub-brand
            brand_canonical.add(canonical)
            competitor_canonical[canonical] = match_result  # store "__brand__::SubName"
            result.brand_matches += 1
        elif match_result is not None and "::" in match_result:
            # Sub-brand of competitor — competitor_canonical stores "CompName::SubName"
            competitor_canonical[canonical] = match_result
            result.competitor_matches += 1
        elif match_result is not None:
            # match_result is the competitor name (direct match)
            competitor_canonical[canonical] = match_result
            result.competitor_matches += 1
        else:
            discovered_canonical.append(canonical)
            result.new_discovered += 1

    logger.info(
        "Consolidation clusters: total=%d, brand=%d, competitor=%d, discovered=%d",
        len(clusters),
        result.brand_matches,
        result.competitor_matches,
        result.new_discovered,
    )

    # Step 10: Update snapshots — rebuild brand_mentioned & competitor_mentions
    updates_count = 0

    # Pre-build the initial competitor_mentions template with sub-brand :: keys
    base_comp_mentions: dict[str, bool] = {c: False for c in competitors}
    if brand_sub_brands:
        for sub_name in brand_sub_brands:
            base_comp_mentions[f"__brand__::{sub_name}"] = False
    if competitor_sub_brands:
        for comp_name, subs in competitor_sub_brands.items():
            for sub_name in subs:
                base_comp_mentions[f"{comp_name}::{sub_name}"] = False

    for snapshot_id, raw_ents in snapshot_entities.items():
        if not raw_ents:
            continue

        # Brand detection: any entity in this snapshot matched brand?
        new_brand_mentioned = any(canonical_map.get(e) in brand_canonical for e in raw_ents)

        # Build competitor_mentions from scratch using entity matching only
        new_comp_mentions: dict[str, bool] = dict(base_comp_mentions)

        for entity in raw_ents:
            canon = canonical_map.get(entity)
            if canon and canon in competitor_canonical:
                match_key = competitor_canonical[canon]
                if match_key.startswith("__brand__::"):
                    new_brand_mentioned = True
                    new_comp_mentions[match_key] = True
                elif "::" in match_key:
                    parent = match_key.split("::")[0]
                    new_comp_mentions[parent] = True
                    new_comp_mentions[match_key] = True
                else:
                    new_comp_mentions[match_key] = True

        await db.execute(
            update(LlmSnapshot)
            .where(LlmSnapshot.id == snapshot_id)
            .values(
                brand_mentioned=new_brand_mentioned,
                competitor_mentions=new_comp_mentions or None,
            )
        )
        updates_count += 1

    result.snapshots_updated = updates_count

    # Step 11: Rewrite discovered_entities with canonical names
    canonical_freq: Counter = Counter()
    for cluster in clusters:
        canonical = canonical_map[cluster[0]]
        for variant in cluster:
            canonical_freq[canonical] += freq[variant]

    for canonical in discovered_canonical:
        total_mentions = canonical_freq.get(canonical, 1)
        stmt = (
            pg_insert(DiscoveredEntity)
            .values(
                tenant_id=tenant_id,
                project_id=project_id,
                entity_name=canonical,
                mention_count=total_mentions,
                first_seen=today,
                last_seen=today,
                status="pending",
            )
            .on_conflict_do_update(
                constraint="uq_project_entity",
                set_={
                    "mention_count": total_mentions,
                    "last_seen": today,
                },
            )
        )
        await db.execute(stmt)

    # Ensure DiscoveredEntity records exist for brand + competitors + sub-brands
    canonical_to_target: dict[str, str] = {}

    for canonical in brand_canonical:
        if canonical not in competitor_canonical:
            canonical_to_target[canonical] = brand_name

    for canonical, match_result in competitor_canonical.items():
        if "::" in match_result:
            _, sub_name = match_result.split("::", 1)
            canonical_to_target[canonical] = sub_name
        else:
            canonical_to_target[canonical] = match_result

    promoted_targets = {brand_name} | set(competitors)
    if brand_sub_brands:
        promoted_targets.update(brand_sub_brands.keys())
    if competitor_sub_brands:
        for _comp, subs in competitor_sub_brands.items():
            promoted_targets.update(subs.keys())

    for target_name in promoted_targets:
        stmt = (
            pg_insert(DiscoveredEntity)
            .values(
                tenant_id=tenant_id,
                project_id=project_id,
                entity_name=target_name,
                mention_count=0,
                first_seen=today,
                last_seen=today,
                status="promoted",
            )
            .on_conflict_do_nothing(constraint="uq_project_entity")
        )
        await db.execute(stmt)

    # NOTE: Alias reset already happened in Step 7a (before loading competitor
    # aliases) to prevent feedback loops from old incorrect aliases.

    # Alias redundant variants to their canonical entity
    all_variants = set(freq.keys())
    all_canonicals = set(canonical_map.values())
    redundant_variants = all_variants - all_canonicals

    for canonical, target_name in canonical_to_target.items():
        if canonical != target_name:
            redundant_variants.add(canonical)

    if redundant_variants:
        variants_by_target: dict[str, list[str]] = {}
        for variant in redundant_variants:
            canonical = canonical_map.get(variant, variant)
            target = canonical_to_target.get(canonical, canonical)
            if target != variant:
                variants_by_target.setdefault(target, []).append(variant)

        for target_name, variants in variants_by_target.items():
            target_q = await db.execute(
                select(DiscoveredEntity.id).where(
                    DiscoveredEntity.project_id == project_id,
                    DiscoveredEntity.entity_name == target_name,
                )
            )
            target_row = target_q.fetchone()
            if not target_row:
                continue
            target_id = target_row[0]

            for variant_name in variants:
                variant_mentions = freq.get(variant_name, 0)
                stmt = (
                    pg_insert(DiscoveredEntity)
                    .values(
                        tenant_id=tenant_id,
                        project_id=project_id,
                        entity_name=variant_name,
                        mention_count=variant_mentions,
                        first_seen=today,
                        last_seen=today,
                        status="alias",
                        alias_of_id=target_id,
                    )
                    .on_conflict_do_update(
                        constraint="uq_project_entity",
                        set_={
                            "mention_count": variant_mentions,
                            "last_seen": today,
                            "alias_of_id": target_id,
                            "status": "alias",
                        },
                        # Guard: never overwrite manually curated entities
                        where=DiscoveredEntity.status.in_(["pending", "alias"]),
                    )
                )
                await db.execute(stmt)

    await db.commit()

    logger.info(
        "Consolidation complete: project=%d, snapshots_updated=%d, discovered=%d, merged=%d",
        project_id,
        updates_count,
        result.new_discovered,
        result.entities_merged,
    )

    return result


# ---------------------------------------------------------------------------
# Fuzzy-only fallback clustering
# ---------------------------------------------------------------------------


def _fuzzy_only_clusters(
    entities_with_counts: list[tuple[str, int]],
) -> list[list[str]]:
    """Pure fuzzy matching clustering (no vectors, no LLM).

    Used as fallback when profile generation fails.
    """
    sorted_entities = sorted(entities_with_counts, key=lambda x: -x[1])
    clusters: list[list[str]] = []
    used: set[str] = set()

    for i, (name_a, _count_a) in enumerate(sorted_entities):
        if name_a in used:
            continue

        cluster = [name_a]
        used.add(name_a)

        for j in range(i + 1, len(sorted_entities)):
            name_b, _count_b = sorted_entities[j]
            if name_b in used:
                continue
            if _names_match(name_a, name_b):
                cluster.append(name_b)
                used.add(name_b)

        clusters.append(cluster)

    remaining = [name for name, _ in sorted_entities if name not in used]
    for name in remaining:
        clusters.append([name])

    logger.info(
        "Fuzzy-only clustering: %d entities → %d clusters",
        len(sorted_entities),
        len(clusters),
    )
    return clusters


# ---------------------------------------------------------------------------
# Helpers (preserved from original)
# ---------------------------------------------------------------------------


async def _enrich_sub_brands_from_db(
    db: AsyncSession,
    project_id: int,
    brand_sub_brands: dict[str, list[str]],
    competitor_sub_brands: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """Enrich sub-brand dicts with alias names from entity relationships (alias_of_id)."""
    all_sub_names = list(brand_sub_brands.keys())
    for comp_name, subs in competitor_sub_brands.items():
        if isinstance(subs, dict):
            all_sub_names.extend(subs.keys())

    if not all_sub_names:
        return brand_sub_brands, competitor_sub_brands

    result = await db.execute(
        select(DiscoveredEntity.id, DiscoveredEntity.entity_name).where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.entity_name.in_(all_sub_names),
        )
    )
    sub_entities = {row.entity_name: row.id for row in result.fetchall()}

    alias_map: dict[str, list[str]] = {}
    if sub_entities:
        alias_result = await db.execute(
            select(DiscoveredEntity.entity_name, DiscoveredEntity.alias_of_id).where(
                DiscoveredEntity.project_id == project_id,
                DiscoveredEntity.alias_of_id.in_(list(sub_entities.values())),
            )
        )
        id_to_name = {v: k for k, v in sub_entities.items()}
        for alias_name, alias_of_id in alias_result.fetchall():
            parent_name = id_to_name.get(alias_of_id)
            if parent_name:
                alias_map.setdefault(parent_name, []).append(alias_name)

    enriched_brand: dict[str, list[str]] = {}
    for sub_name, existing_aliases in brand_sub_brands.items():
        enriched_brand[sub_name] = list(set((existing_aliases or []) + alias_map.get(sub_name, [])))

    enriched_comp: dict[str, dict[str, list[str]]] = {}
    for comp_name, subs in competitor_sub_brands.items():
        enriched_comp[comp_name] = {}
        if isinstance(subs, dict):
            for sub_name, existing_aliases in subs.items():
                enriched_comp[comp_name][sub_name] = list(set((existing_aliases or []) + alias_map.get(sub_name, [])))

    return enriched_brand, enriched_comp


async def _load_competitor_aliases(
    db: AsyncSession,
    project_id: int,
    competitors: list[str],
) -> dict[str, list[str]]:
    """Load entity aliases for competitors from the DB."""
    if not competitors:
        return {}

    result = await db.execute(
        select(DiscoveredEntity.id, DiscoveredEntity.entity_name).where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.entity_name.in_(competitors),
        )
    )
    comp_entities = {row.entity_name: row.id for row in result.fetchall()}

    if not comp_entities:
        return {}

    alias_result = await db.execute(
        select(DiscoveredEntity.entity_name, DiscoveredEntity.alias_of_id).where(
            DiscoveredEntity.project_id == project_id,
            DiscoveredEntity.alias_of_id.in_(list(comp_entities.values())),
        )
    )
    id_to_name = {v: k for k, v in comp_entities.items()}
    alias_map: dict[str, list[str]] = {}
    for alias_name, alias_of_id in alias_result.fetchall():
        comp_name = id_to_name.get(alias_of_id)
        if comp_name:
            alias_map.setdefault(comp_name, []).append(alias_name)

    return alias_map


def _pick_canonical(cluster: list[str], freq: Counter) -> str:
    """Pick the canonical name from a cluster.

    Picks the most frequent variant. On tie, picks the shortest.
    """
    if len(cluster) == 1:
        return cluster[0]

    return sorted(cluster, key=lambda n: (-freq.get(n, 0), len(n), n))[0]


def _variant_match_roots(
    variant: str,
    brand_name: str,
    brand_aliases: list[str],
    competitors: list[str],
    brand_sub_brands: dict[str, list[str]] | None = None,
    competitor_sub_brands: dict[str, dict[str, list[str]]] | None = None,
    competitor_aliases: dict[str, list[str]] | None = None,
) -> set[str]:
    """Return top-level brand roots this variant can match."""
    roots: set[str] = set()

    for bn in [brand_name] + (brand_aliases or []):
        if bn and _names_match(variant, bn):
            roots.add("__brand__")
            break

    if brand_sub_brands:
        for sub_name, sub_aliases in brand_sub_brands.items():
            for sa in [sub_name] + (sub_aliases or []):
                if sa and _names_match(variant, sa):
                    roots.add("__brand__")
                    break
            if "__brand__" in roots:
                break

    for comp in competitors:
        comp_names = [comp] + (competitor_aliases or {}).get(comp, [])
        for cn in comp_names:
            if cn and _names_match(variant, cn):
                roots.add(comp)
                break

    if competitor_sub_brands:
        for comp_name, sub_brands in competitor_sub_brands.items():
            for sub_name, sub_aliases in sub_brands.items():
                for sa in [sub_name] + (sub_aliases or []):
                    if sa and _names_match(variant, sa):
                        roots.add(comp_name)
                        break

    return roots


def _split_overbroad_known_clusters(
    clusters: list[list[str]],
    brand_name: str,
    brand_aliases: list[str],
    competitors: list[str],
    brand_sub_brands: dict[str, list[str]] | None = None,
    competitor_sub_brands: dict[str, dict[str, list[str]]] | None = None,
    competitor_aliases: dict[str, list[str]] | None = None,
) -> list[list[str]]:
    """Split clusters that mix different known top-level brands."""
    if not clusters:
        return []

    sanitized: list[list[str]] = []
    split_count = 0
    split_members = 0

    for cluster in clusters:
        if len(cluster) <= 1:
            sanitized.append(cluster)
            continue

        by_root: dict[str, list[str]] = {}
        unknown_or_ambiguous: list[str] = []

        for variant in cluster:
            roots = _variant_match_roots(
                variant=variant,
                brand_name=brand_name,
                brand_aliases=brand_aliases,
                competitors=competitors,
                brand_sub_brands=brand_sub_brands,
                competitor_sub_brands=competitor_sub_brands,
                competitor_aliases=competitor_aliases,
            )

            if len(roots) == 1:
                root = next(iter(roots))
                by_root.setdefault(root, []).append(variant)
            else:
                unknown_or_ambiguous.append(variant)

        if not by_root:
            sanitized.append(cluster)
            continue

        if len(by_root) == 1 and not unknown_or_ambiguous:
            sanitized.append(cluster)
            continue

        split_count += 1
        split_members += len(cluster)

        for bucket in by_root.values():
            if bucket:
                sanitized.append(bucket)
        for name in unknown_or_ambiguous:
            sanitized.append([name])

    if split_count:
        logger.info(
            "Cluster sanitizer split %d mixed clusters (%d members) -> %d clusters",
            split_count,
            split_members,
            len(sanitized),
        )

    return sanitized


def _match_cluster_to_known(
    cluster: list[str],
    brand_name: str,
    brand_aliases: list[str],
    competitors: list[str],
    brand_sub_brands: dict[str, list[str]] | None = None,
    competitor_sub_brands: dict[str, dict[str, list[str]]] | None = None,
    competitor_aliases: dict[str, list[str]] | None = None,
) -> str | None:
    """Try to match a cluster to the target brand, sub-brand, or a known competitor.

    Returns:
        "__brand__" if cluster matches target brand.
        "__brand__::SubName" if cluster matches a sub-brand of the target brand.
        "CompName" if cluster matches a competitor.
        "CompName::SubName" if cluster matches a sub-brand of a competitor.
        None if no match found (= new discovered entity).
    """
    brand_names = [brand_name] + (brand_aliases or [])
    for variant in cluster:
        for bn in brand_names:
            if _names_match(variant, bn):
                return "__brand__"

    if brand_sub_brands:
        for sub_name, sub_aliases in brand_sub_brands.items():
            sub_all = [sub_name] + (sub_aliases or [])
            for variant in cluster:
                for sa in sub_all:
                    if _names_match(variant, sa):
                        return f"__brand__::{sub_name}"

    for variant in cluster:
        for comp in competitors:
            comp_names = [comp] + (competitor_aliases or {}).get(comp, [])
            for cn in comp_names:
                if _names_match(variant, cn):
                    return comp

    if competitor_sub_brands:
        for comp_name, sub_brands in competitor_sub_brands.items():
            for sub_name, sub_aliases in sub_brands.items():
                sub_all = [sub_name] + (sub_aliases or [])
                for variant in cluster:
                    for sa in sub_all:
                        if _names_match(variant, sa):
                            return f"{comp_name}::{sub_name}"

    return None


# ---------------------------------------------------------------------------
# Sub-brand auto-detection (preserved from original)
# ---------------------------------------------------------------------------

_CLUSTER_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_SUB_BRAND_DETECT_PROMPT = """You are given a list of entities discovered in LLM responses about a specific brand and its competitors.
Your task: determine which entities are sub-brands (products, product lines, residential complexes, services, etc.) of the known brands.

Target brand: {brand_name}
Known competitors: {competitors_json}

Discovered entities (not yet classified):
{entities_json}

For each entity that looks like a sub-brand of one of the known brands (target or competitors), return:
- "entity": the entity name exactly as given
- "parent": the parent brand name (must be either "{brand_name}" or one of the competitors exactly as listed above)
- "confidence": "high" or "medium"

Only include entities you are reasonably confident about. Skip entities that are independent brands, generic terms, or unclear.

Return JSON: {{"sub_brands": [{{"entity": "...", "parent": "...", "confidence": "..."}}]}}
If none found, return: {{"sub_brands": []}}"""

_SUB_BRAND_DETECT_MODEL = "gemini-3-flash-preview"
_SUB_BRAND_DETECT_TIMEOUT = 300


async def detect_sub_brands(
    db: AsyncSession,
    project_id: int,
    tenant_id: UUID,
    brand_name: str,
    competitors: list[str],
    gemini_api_key: str,
) -> list[dict]:
    """Detect potential sub-brands among discovered entities using LLM.

    Reads pending/confirmed discovered entities and asks Gemini to identify
    which ones are sub-brands of known brands.

    Returns list of {"entity": str, "parent": str, "confidence": str}.
    Also updates suggested_parent field on matching discovered entities.
    """
    # Load unmatched discovered entities
    stmt = select(DiscoveredEntity).where(
        DiscoveredEntity.project_id == project_id,
        DiscoveredEntity.tenant_id == tenant_id,
        DiscoveredEntity.status.in_(["pending", "confirmed"]),
    )
    rows = (await db.execute(stmt)).scalars().all()

    logger.info(
        "[sub-brand] project=%d brand=%r competitors=%r → loaded %d entities",
        project_id,
        brand_name,
        competitors,
        len(rows),
    )

    if not rows:
        logger.info("[sub-brand] No pending/confirmed entities found, returning early")
        return []

    entity_names = [e.entity_name for e in rows]
    if not entity_names:
        return []

    logger.info("[sub-brand] Entity names sample (first 10): %s", entity_names[:10])

    # Build prompt
    prompt = _SUB_BRAND_DETECT_PROMPT.format(
        brand_name=brand_name,
        competitors_json=json.dumps(competitors, ensure_ascii=False),
        entities_json=json.dumps(entity_names, ensure_ascii=False),
    )

    url = _CLUSTER_API_URL.format(model=_SUB_BRAND_DETECT_MODEL)
    logger.info("[sub-brand] Calling Gemini model=%s, prompt length=%d chars", _SUB_BRAND_DETECT_MODEL, len(prompt))

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_SUB_BRAND_DETECT_TIMEOUT, connect=30.0),
        ) as client:
            resp = await client.post(
                url,
                params={"key": gemini_api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 65536,
                        "responseMimeType": "application/json",
                        "responseSchema": {
                            "type": "OBJECT",
                            "properties": {
                                "sub_brands": {
                                    "type": "ARRAY",
                                    "items": {
                                        "type": "OBJECT",
                                        "properties": {
                                            "entity": {"type": "STRING"},
                                            "parent": {"type": "STRING"},
                                            "confidence": {"type": "STRING", "enum": ["high", "medium"]},
                                        },
                                        "required": ["entity", "parent", "confidence"],
                                    },
                                }
                            },
                            "required": ["sub_brands"],
                        },
                        "thinkingConfig": {
                            "thinkingLevel": "minimal",
                        },
                    },
                },
            )
            logger.info("[sub-brand] Gemini response status=%d", resp.status_code)
            if resp.status_code != 200:
                logger.error("[sub-brand] Gemini error %d: %s", resp.status_code, resp.text[:500])
                return []

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            logger.warning("[sub-brand] No candidates in Gemini response. Full response keys: %s", list(data.keys()))
            return []

        finish_reason = candidates[0].get("finishReason", "unknown")
        logger.info("[sub-brand] Gemini finishReason=%s", finish_reason)

        parts = candidates[0].get("content", {}).get("parts", [])
        content = ""
        for part in parts:
            if "thought" not in part and "text" in part:
                content += part["text"]

        logger.info("[sub-brand] Gemini raw content length=%d, first 500 chars: %s", len(content), content[:500])

        if not content:
            logger.warning("[sub-brand] Empty content from Gemini (all parts were thoughts?)")
            return []

        # Parse JSON response (with multiple fallback strategies)
        parsed = None

        try:
            parsed = json.loads(content)
            logger.info("[sub-brand] Direct JSON parse succeeded")
        except json.JSONDecodeError as e1:
            logger.info("[sub-brand] Direct parse failed: %s", e1)

        if parsed is None:
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                raw_json = json_match.group()
                fixed = re.sub(r",\s*([}\]])", r"\1", raw_json)
                try:
                    parsed = json.loads(fixed)
                    logger.info("[sub-brand] Regex+trailing-comma fix succeeded")
                except json.JSONDecodeError as e2:
                    logger.info("[sub-brand] Regex+fix failed: %s", e2)

        if parsed is None:
            logger.info("[sub-brand] Attempting truncation recovery...")
            arr_match = re.search(r'"sub_brands"\s*:\s*\[', content)
            if arr_match:
                arr_start = arr_match.end()
                items = []
                depth = 0
                obj_start = None
                for i in range(arr_start, len(content)):
                    ch = content[i]
                    if ch == "{":
                        if depth == 0:
                            obj_start = i
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and obj_start is not None:
                            obj_str = content[obj_start : i + 1]
                            try:
                                item = json.loads(obj_str)
                                items.append(item)
                            except json.JSONDecodeError:
                                obj_fixed = re.sub(r",\s*([}\]])", r"\1", obj_str)
                                try:
                                    item = json.loads(obj_fixed)
                                    items.append(item)
                                except json.JSONDecodeError:
                                    pass
                            obj_start = None
                if items:
                    parsed = {"sub_brands": items}
                    logger.info("[sub-brand] Truncation recovery: extracted %d complete items", len(items))

        if parsed is None:
            logger.warning(
                "[sub-brand] All JSON parse attempts failed. Content tail (last 300 chars): %s", content[-300:]
            )
            return []

        suggestions = parsed.get("sub_brands", [])
        logger.info("[sub-brand] Parsed %d raw suggestions from LLM", len(suggestions))

        if not isinstance(suggestions, list):
            logger.warning("[sub-brand] suggestions is not a list: %s", type(suggestions))
            return []

        # Validate: parent must be brand_name or one of competitors
        valid_parents = {brand_name} | set(competitors)
        logger.info("[sub-brand] Valid parents: %s", valid_parents)

        def _normalize_name(name: str) -> str:
            return name.replace("\\", "").replace('"', "").replace("'", "").strip()

        norm_to_db: dict[str, str] = {}
        for e in rows:
            norm_to_db[_normalize_name(e.entity_name)] = e.entity_name
        entity_name_set = {e.entity_name for e in rows}

        valid_suggestions = []
        for s in suggestions:
            if not isinstance(s, dict):
                continue
            entity = s.get("entity", "")
            parent = s.get("parent", "")
            confidence = s.get("confidence", "medium")

            if entity in entity_name_set:
                db_entity = entity
            else:
                db_entity = norm_to_db.get(_normalize_name(entity))

            parent_ok = parent in valid_parents
            if db_entity and parent_ok:
                valid_suggestions.append(
                    {
                        "entity": db_entity,
                        "parent": parent,
                        "confidence": confidence,
                    }
                )
            else:
                logger.info(
                    "[sub-brand] Rejected: entity=%r (matched=%s) parent=%r (valid=%s)",
                    entity,
                    db_entity is not None,
                    parent,
                    parent_ok,
                )

        # Update suggested_parent in DB
        if valid_suggestions:
            suggestion_map = {s["entity"]: s["parent"] for s in valid_suggestions}
            for entity_row in rows:
                if entity_row.entity_name in suggestion_map:
                    entity_row.suggested_parent = suggestion_map[entity_row.entity_name]
            await db.commit()

        logger.info(
            "[sub-brand] Detection done: %d entities → %d raw → %d valid suggestions for project=%d",
            len(entity_names),
            len(suggestions),
            len(valid_suggestions),
            project_id,
        )
        return valid_suggestions

    except Exception as e:
        logger.warning("[sub-brand] Detection failed (non-fatal): %s: %s", type(e).__name__, e)
        return []
