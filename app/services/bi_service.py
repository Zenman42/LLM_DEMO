"""BI Dashboard, Aggregation & Actionable Insights Engine.

Pure-function computation layer + async DB fetch layer.
All business logic is in pure functions (no DB dependency) for testability.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

from app.schemas.bi_dashboard import (
    BiDashboardResponse,
    CitationDomain,
    CitationTrustGraph,
    CompetitiveMatrix,
    CompetitorRow,
    GeoActionPlan,
    GeoInsight,
    GlobalMetrics,
    HeatmapCell,
    HeatmapResponse,
)
from app.schemas.llm_debug import (
    AivsDebug,
    DebugSnapshotItem,
    LlmDebugResponse,
    MentionRateDebug,
    ResilienceDebug,
    ResilienceGroupDebug,
    SomDebug,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RVS_LOOKUP: dict[str, float] = {
    "recommended": 1.2,
    "direct": 0.70,
    "compared": 0.50,
    "negative": 0.35,
    "none": 0.0,
}

INTENT_WEIGHTS: dict[str, float] = {
    "comparison": 1.5,
    "recommendation": 1.2,
    "brand_check": 1.0,
    "custom": 0.8,
}


# ---------------------------------------------------------------------------
# SnapshotRow — lightweight DTO for passing data between DB and pure funcs
# ---------------------------------------------------------------------------


@dataclass
class SnapshotRow:
    """Minimal row extracted from JOIN of llm_snapshots + llm_queries."""

    llm_query_id: int
    query_type: str  # from LlmQuery
    date: date
    llm_provider: str
    brand_mentioned: bool
    mention_type: str  # direct | recommended | compared | negative | none
    competitor_mentions: dict | None = None
    cited_urls: list | None = None
    # GEO metrics (populated by Module 3 pipeline; None for legacy rows)
    sentiment_label: str | None = None  # positive | neutral | negative | mixed
    position_rank: int | None = None  # 0=not in list, 1+=position
    structure_type: str | None = None  # numbered_list | bulleted_list | narrative | table | mixed
    is_hallucination: bool | None = None
    category: str | None = None  # scenario from LlmQuery.category


@dataclass
class DebugSnapshotRow:
    """Extended row with all fields needed for the Debug Console."""

    id: int
    llm_query_id: int
    query_text: str
    query_type: str
    date: date
    llm_provider: str
    llm_model: str
    brand_mentioned: bool
    mention_type: str
    mention_context: str | None = None
    competitor_mentions: dict | None = None
    cited_urls: list | None = None
    response_tokens: int | None = None
    cost_usd: float | None = None
    target_brand: str | None = None  # for brand_in_prompt detection


def _brand_in_prompt(query_text: str, target_brand: str | None) -> bool:
    """Check if the target brand name appears in the query text.

    If brand is explicitly mentioned in the prompt, LLM is almost guaranteed
    to mention it back — making Mention Rate artificially 100%.
    """
    if not target_brand:
        return False
    return target_brand.lower() in query_text.lower()


# ---------------------------------------------------------------------------
# Pure functions: AIVS
# ---------------------------------------------------------------------------


def rvs_for_mention_type(mention_type: str) -> float:
    """Map mention_type to approximate RVS value."""
    return RVS_LOOKUP.get(mention_type, 0.0)


def intent_weight(query_type: str) -> float:
    """Map query_type to intent weight."""
    return INTENT_WEIGHTS.get(query_type, 0.8)


def compute_aivs(rows: list[SnapshotRow]) -> float:
    """Compute AI Visibility Score (0-100).

    AIVS = (sum(RVS_i * intent_weight_i) / sum(intent_weight_i)) * 100
    Weighted average of per-response RVS scaled by intent importance.
    """
    if not rows:
        return 0.0

    weighted_sum = 0.0
    weight_sum = 0.0

    for r in rows:
        rvs = rvs_for_mention_type(r.mention_type)
        w = intent_weight(r.query_type)
        weighted_sum += rvs * w
        weight_sum += w

    if weight_sum == 0.0:
        return 0.0

    # RVS max is 1.2 so raw ratio can exceed 1.0; cap at 100
    raw = (weighted_sum / weight_sum) * 100.0
    return round(min(raw, 100.0), 2)


# ---------------------------------------------------------------------------
# Pure functions: Share of Model
# ---------------------------------------------------------------------------


def compute_som(brand_mentions: int, total_mentions: int) -> float:
    """Share of Model = brand_mentions / total_all_brand_mentions * 100%.

    total_mentions includes brand + all competitors.
    """
    if total_mentions == 0:
        return 0.0
    return round((brand_mentions / total_mentions) * 100.0, 2)


def _count_mentions(rows: list[SnapshotRow]) -> tuple[int, int, dict[str, int]]:
    """Count brand mentions, total mentions (brand + competitors), and per-competitor counts."""
    brand_count = 0
    competitor_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        if r.brand_mentioned:
            brand_count += 1
        if r.competitor_mentions:
            for comp, mentioned in r.competitor_mentions.items():
                if mentioned:
                    # Skip sub-brand :: keys (roll-up: parent already counted)
                    if "::" in comp:
                        continue
                    competitor_counts[comp] += 1

    total = brand_count + sum(competitor_counts.values())
    return brand_count, total, dict(competitor_counts)


# ---------------------------------------------------------------------------
# Pure functions: Resilience Score
# ---------------------------------------------------------------------------


def compute_resilience(rows: list[SnapshotRow]) -> float:
    """Resilience = fraction of (query, provider) pairs that consistently mention brand.

    For each (llm_query_id, llm_provider) group:
      resilience_i = dates_with_mention / total_dates
    Global resilience = average of all resilience_i (only for groups with >0 rows).
    """
    if not rows:
        return 0.0

    # Group by (query_id, provider)
    groups: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: {"mentioned": 0, "total": 0})
    for r in rows:
        key = (r.llm_query_id, r.llm_provider)
        groups[key]["total"] += 1
        if r.brand_mentioned:
            groups[key]["mentioned"] += 1

    if not groups:
        return 0.0

    resiliences = [g["mentioned"] / g["total"] for g in groups.values() if g["total"] > 0]
    if not resiliences:
        return 0.0

    return round(sum(resiliences) / len(resiliences), 4)


# ---------------------------------------------------------------------------
# Pure functions: Net Sentiment Score (NSS)
# ---------------------------------------------------------------------------

# Fallback mapping: mention_type → sentiment for legacy rows without sentiment_label
_MENTION_TYPE_TO_SENTIMENT: dict[str, str] = {
    "recommended": "positive",
    "negative": "negative",
    "direct": "neutral",
    "compared": "neutral",
    "none": "skip",  # not mentioned → skip
}


def compute_nss(rows: list[SnapshotRow]) -> float:
    """Net Sentiment Score = (% positive − % negative) among mentioned rows.

    Scale: −100 to +100.
    Uses `sentiment_label` when available, falls back to `mention_type` mapping
    for legacy rows where pipeline didn't populate the field.
    """
    positive = 0
    negative = 0
    total_scored = 0

    for r in rows:
        # Determine effective sentiment
        if r.sentiment_label:
            label = r.sentiment_label.lower()
        else:
            # Fallback: map mention_type → sentiment
            label = _MENTION_TYPE_TO_SENTIMENT.get(r.mention_type, "skip")

        if label == "skip" or label == "none":
            continue

        total_scored += 1
        if label == "positive":
            positive += 1
        elif label == "negative":
            negative += 1
        # neutral / mixed → counted in total but don't affect pos/neg

    if total_scored == 0:
        return 0.0

    nss = ((positive / total_scored) - (negative / total_scored)) * 100.0
    return round(nss, 2)


# ---------------------------------------------------------------------------
# Pure functions: Top-1 Win Rate
# ---------------------------------------------------------------------------

_STRUCTURED_TYPES = {"numbered_list", "table", "mixed"}


def compute_top1_win_rate(rows: list[SnapshotRow]) -> float | None:
    """Top-1 Win Rate = % of structured responses where brand is ranked #1.

    Only considers rows where:
    - structure_type indicates a ranked list (numbered_list, table, mixed)
    - position_rank is not None
    Returns None if no qualifying rows exist (need re-collect with pipeline).
    """
    qualifying = [r for r in rows if r.structure_type in _STRUCTURED_TYPES and r.position_rank is not None]

    if not qualifying:
        return None

    wins = sum(1 for r in qualifying if r.position_rank == 1)
    return round((wins / len(qualifying)) * 100.0, 2)


# ---------------------------------------------------------------------------
# Pure functions: Feature Hallucination Rate
# ---------------------------------------------------------------------------


def compute_hallucination_rate(rows: list[SnapshotRow]) -> float | None:
    """Hallucination Rate = % of brand-mentioned responses flagged as hallucination.

    Only considers rows where:
    - brand_mentioned is True
    - is_hallucination is not None (pipeline ran)
    Returns None if no qualifying rows exist.
    """
    qualifying = [r for r in rows if r.brand_mentioned and r.is_hallucination is not None]

    if not qualifying:
        return None

    hallucinated = sum(1 for r in qualifying if r.is_hallucination)
    return round((hallucinated / len(qualifying)) * 100.0, 2)


# ---------------------------------------------------------------------------
# Pure functions: Intent Penetration
# ---------------------------------------------------------------------------


def compute_intent_penetration(rows: list[SnapshotRow]) -> dict[str, float]:
    """AIVS broken down per intent type (query_type).

    Returns e.g. {"comparison": 45.2, "recommendation": 82.0, "brand_check": 60.0}
    Only includes intent types that have at least one row.
    """
    groups: dict[str, list[SnapshotRow]] = defaultdict(list)
    for r in rows:
        groups[r.query_type].append(r)

    result: dict[str, float] = {}
    for intent_type, group_rows in sorted(groups.items()):
        result[intent_type] = compute_aivs(group_rows)

    return result


# ---------------------------------------------------------------------------
# Pure functions: Global Metrics
# ---------------------------------------------------------------------------


def compute_global_metrics(rows: list[SnapshotRow]) -> GlobalMetrics:
    """Compute all global metrics from snapshot rows."""
    aivs = compute_aivs(rows)
    brand_count, total_mentions, _ = _count_mentions(rows)
    som = compute_som(brand_count, total_mentions)
    resilience = compute_resilience(rows)
    total_responses = len(rows)
    mention_rate = round(brand_count / total_responses, 4) if total_responses > 0 else 0.0

    # GEO Level 2 metrics
    nss = compute_nss(rows)
    top1_win_rate = compute_top1_win_rate(rows)
    hallucination_rate = compute_hallucination_rate(rows)

    # GEO Level 3 metrics
    intent_penetration = compute_intent_penetration(rows)

    return GlobalMetrics(
        aivs=aivs,
        som=som,
        resilience_score=resilience,
        total_responses=total_responses,
        mention_rate=mention_rate,
        nss=nss,
        top1_win_rate=top1_win_rate,
        hallucination_rate=hallucination_rate,
        intent_penetration=intent_penetration,
    )


# ---------------------------------------------------------------------------
# Pure functions: Competitive Matrix
# ---------------------------------------------------------------------------


def compute_competitive_matrix(rows: list[SnapshotRow], brand_name: str = "") -> CompetitiveMatrix:
    """Build target + competitor comparison table."""
    brand_count, total_mentions, competitor_counts = _count_mentions(rows)
    total_responses = len(rows)

    # Target row
    brand_rvs_values = [rvs_for_mention_type(r.mention_type) for r in rows if r.brand_mentioned]
    target_avg_rvs = round(sum(brand_rvs_values) / len(brand_rvs_values), 4) if brand_rvs_values else 0.0

    target = CompetitorRow(
        name=brand_name or "target",
        som=compute_som(brand_count, total_mentions),
        mention_rate=round(brand_count / total_responses, 4) if total_responses > 0 else 0.0,
        avg_rvs=target_avg_rvs,
    )

    # Competitor rows
    competitors: list[CompetitorRow] = []
    for comp_name, comp_count in sorted(competitor_counts.items(), key=lambda x: x[1], reverse=True):
        competitors.append(
            CompetitorRow(
                name=comp_name,
                som=compute_som(comp_count, total_mentions),
                mention_rate=round(comp_count / total_responses, 4) if total_responses > 0 else 0.0,
                avg_rvs=0.0,  # We don't have per-competitor mention_type
            )
        )

    return CompetitiveMatrix(target=target, competitors=competitors)


# ---------------------------------------------------------------------------
# Pure functions: Heatmap (Intent x Vendor)
# ---------------------------------------------------------------------------


def _classify_zone(aivs: float) -> str:
    """Classify AIVS into traffic-light zone."""
    if aivs < 20:
        return "red"
    if aivs < 50:
        return "yellow"
    return "green"


_UNCATEGORIZED = "Uncategorized"


def compute_heatmap(rows: list[SnapshotRow]) -> HeatmapResponse:
    """Build Scenario x Vendor heatmap with AIVS per cell."""
    if not rows:
        return HeatmapResponse(cells=[], scenarios=[], vendors=[])

    # Group rows by (category/scenario, llm_provider)
    groups: dict[tuple[str, str], list[SnapshotRow]] = defaultdict(list)
    for r in rows:
        scenario = r.category or _UNCATEGORIZED
        groups[(scenario, r.llm_provider)].append(r)

    scenarios_set: set[str] = set()
    vendors_set: set[str] = set()
    cells: list[HeatmapCell] = []

    for (scenario, vendor), group_rows in groups.items():
        scenarios_set.add(scenario)
        vendors_set.add(vendor)

        cell_aivs = compute_aivs(group_rows)
        mention_count = sum(1 for r in group_rows if r.brand_mentioned)

        cells.append(
            HeatmapCell(
                scenario=scenario,
                vendor=vendor,
                aivs=cell_aivs,
                mention_count=mention_count,
                total_count=len(group_rows),
                zone=_classify_zone(cell_aivs),
            )
        )

    return HeatmapResponse(
        cells=cells,
        scenarios=sorted(scenarios_set),
        vendors=sorted(vendors_set),
    )


# ---------------------------------------------------------------------------
# Pure functions: Citation Trust Graph
# ---------------------------------------------------------------------------


def _extract_domain(url: str) -> str:
    """Extract domain from URL, stripping www."""
    try:
        netloc = urlparse(url).netloc
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def compute_citation_graph(rows: list[SnapshotRow]) -> CitationTrustGraph:
    """Analyze domains cited across responses, flagging competitor-win sources."""
    # Track all citations
    domain_data: dict[str, dict] = {}  # domain → {count, providers, competitor_win}

    for r in rows:
        if not r.cited_urls:
            continue

        # Determine if this is a "competitor win" response:
        # brand NOT mentioned but at least one competitor IS mentioned
        is_competitor_win = False
        if not r.brand_mentioned and r.competitor_mentions:
            is_competitor_win = any(r.competitor_mentions.values())

        for url in r.cited_urls:
            domain = _extract_domain(url)
            if not domain:
                continue

            if domain not in domain_data:
                domain_data[domain] = {"count": 0, "providers": set(), "competitor_win": False}

            domain_data[domain]["count"] += 1
            domain_data[domain]["providers"].add(r.llm_provider)
            if is_competitor_win:
                domain_data[domain]["competitor_win"] = True

    domains = sorted(domain_data.items(), key=lambda x: x[1]["count"], reverse=True)
    total_citations = sum(d["count"] for d in domain_data.values())

    return CitationTrustGraph(
        domains=[
            CitationDomain(
                domain=domain,
                count=data["count"],
                providers=sorted(data["providers"]),
                appears_in_competitor_wins=data["competitor_win"],
            )
            for domain, data in domains
        ],
        total_citations=total_citations,
    )


# ---------------------------------------------------------------------------
# Pure functions: GEO-Advisor (Rule Engine)
# ---------------------------------------------------------------------------


def generate_geo_insights(
    metrics: GlobalMetrics,
    heatmap: HeatmapResponse,
    citation_graph: CitationTrustGraph,
) -> GeoActionPlan:
    """Generate actionable GEO recommendations from computed metrics."""
    insights: list[GeoInsight] = []

    # Rule 1: INVISIBLE_PROBLEM
    if metrics.som == 0 or metrics.aivs < 5:
        insights.append(
            GeoInsight(
                rule_id="INVISIBLE_PROBLEM",
                severity="critical",
                title_ru="Бренд невидим для ИИ",
                description_ru=(
                    f"AIVS = {metrics.aivs}, Share of Model = {metrics.som}%. "
                    "Бренд практически не упоминается в ответах LLM."
                ),
                recommendation_ru=(
                    "Необходимо создать контент, оптимизированный для LLM: "
                    "экспертные статьи, FAQ, сравнительные обзоры на авторитетных площадках."
                ),
            )
        )

    # Rule 2: TOXIC_TRAIL
    if metrics.total_responses > 0:
        # We need to check this from heatmap data indirectly —
        # or we can pass rows but keeping pure. Use mention_rate proxy:
        # If brand is mentioned but AIVS is very low, it might be negative mentions.
        # For precise check, we'd need mention_type distribution.
        # We'll check if AIVS < 30 but mention_rate > 0.3 (mentioned often but poorly)
        if metrics.mention_rate > 0.3 and metrics.aivs < 30:
            insights.append(
                GeoInsight(
                    rule_id="TOXIC_TRAIL",
                    severity="critical",
                    title_ru="Негативный шлейф бренда",
                    description_ru=(
                        f"Бренд упоминается в {metrics.mention_rate * 100:.0f}% ответов, "
                        f"но AIVS всего {metrics.aivs}. Это указывает на негативный контекст упоминаний."
                    ),
                    recommendation_ru=(
                        "Проанализируйте контекст упоминаний. Создайте позитивный контент: "
                        "кейсы, отзывы, экспертные рекомендации на авторитетных источниках."
                    ),
                )
            )

    # Rule 3: HEATMAP_RED_ZONE
    red_cells = [c for c in heatmap.cells if c.zone == "red"]
    if red_cells:
        red_descriptions = [f"{c.scenario}+{c.vendor} (AIVS={c.aivs})" for c in red_cells[:5]]
        insights.append(
            GeoInsight(
                rule_id="HEATMAP_RED_ZONE",
                severity="warning",
                title_ru="Красные зоны на тепловой карте",
                description_ru=(f"Обнаружено {len(red_cells)} красных зон (AIVS < 20): " + ", ".join(red_descriptions)),
                recommendation_ru=(
                    "Создайте целевой контент для проблемных комбинаций сценарий+провайдер. "
                    "Приоритет — комбинации с наименьшим AIVS."
                ),
            )
        )

    # Rule 4: COMPARATIVE_GAP — scenarios where ALL providers show AIVS < 20
    if heatmap.scenarios and len(heatmap.vendors) > 0:
        scenario_cells: dict[str, list[HeatmapCell]] = defaultdict(list)
        for c in heatmap.cells:
            scenario_cells[c.scenario].append(c)
        fully_red = [s for s, cells in scenario_cells.items() if all(c.aivs < 20 for c in cells)]
        if fully_red:
            insights.append(
                GeoInsight(
                    rule_id="COMPARATIVE_GAP",
                    severity="warning",
                    title_ru="Полностью невидимые сценарии",
                    description_ru=(
                        f"В {len(fully_red)} сценариях бренд невидим у всех провайдеров "
                        f"(AIVS < 20): {', '.join(fully_red[:5])}."
                    ),
                    recommendation_ru=(
                        "Создайте целевой контент для этих сценариев: "
                        "экспертные статьи, обзоры, FAQ на авторитетных площадках."
                    ),
                )
            )

    # Rule 5: LOCALITY_CURSE
    if len(heatmap.vendors) >= 3:
        providers_with_mentions = set()
        for c in heatmap.cells:
            if c.mention_count > 0:
                providers_with_mentions.add(c.vendor)
        if 0 < len(providers_with_mentions) <= 1:
            the_provider = next(iter(providers_with_mentions))
            insights.append(
                GeoInsight(
                    rule_id="LOCALITY_CURSE",
                    severity="warning",
                    title_ru="Локальная зависимость от одного провайдера",
                    description_ru=(
                        f"Бренд упоминается только в {the_provider}, "
                        f"но отсутствует в {len(heatmap.vendors) - 1} других провайдерах."
                    ),
                    recommendation_ru=(
                        "Диверсифицируйте источники: публикуйте контент на площадках, "
                        "которые индексируются разными LLM-провайдерами."
                    ),
                )
            )

    # Rule 6: CITATION_COMPETITOR_DOMINANCE
    if citation_graph.domains:
        competitor_win_domains = [d for d in citation_graph.domains if d.appears_in_competitor_wins]
        ratio = len(competitor_win_domains) / len(citation_graph.domains) if citation_graph.domains else 0
        if ratio > 0.7:
            insights.append(
                GeoInsight(
                    rule_id="CITATION_COMPETITOR_DOMINANCE",
                    severity="warning",
                    title_ru="Доминирование конкурентов в цитируемых источниках",
                    description_ru=(
                        f"{len(competitor_win_domains)} из {len(citation_graph.domains)} "
                        "цитируемых доменов появляются в ответах, где рекомендуют конкурентов."
                    ),
                    recommendation_ru=(
                        "Добейтесь присутствия бренда на этих площадках. "
                        "Приоритет: домены с наибольшим количеством цитирований."
                    ),
                )
            )

    # Rule 7: LOW_RESILIENCE
    if metrics.resilience_score < 0.3 and metrics.mention_rate > 0:
        insights.append(
            GeoInsight(
                rule_id="LOW_RESILIENCE",
                severity="warning",
                title_ru="Низкая стабильность упоминаний",
                description_ru=(
                    f"Resilience Score = {metrics.resilience_score:.2f}. "
                    "Бренд упоминается нестабильно — результаты сильно варьируются между запусками."
                ),
                recommendation_ru=(
                    "Увеличьте объём и разнообразие контента о бренде, "
                    "чтобы LLM стабильно находили релевантную информацию."
                ),
            )
        )

    # Rule 8: HALLUCINATION_RISK
    if metrics.resilience_score < 0.2 and metrics.aivs > 50:
        insights.append(
            GeoInsight(
                rule_id="HALLUCINATION_RISK",
                severity="info",
                title_ru="Риск галлюцинаций",
                description_ru=(
                    f"Высокий AIVS ({metrics.aivs}) при низкой стабильности "
                    f"({metrics.resilience_score:.2f}) может указывать на галлюцинации LLM."
                ),
                recommendation_ru=(
                    "Проверьте, соответствуют ли упоминания бренда реальности. "
                    "Создайте верифицируемый контент на авторитетных источниках."
                ),
            )
        )

    return GeoActionPlan(
        insights=insights,
        generated_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Pure function: Full BI dashboard computation
# ---------------------------------------------------------------------------


def compute_bi_dashboard(rows: list[SnapshotRow], brand_name: str = "") -> BiDashboardResponse:
    """Compute the complete BI dashboard from snapshot rows."""
    metrics = compute_global_metrics(rows)
    matrix = compute_competitive_matrix(rows, brand_name=brand_name)
    heatmap = compute_heatmap(rows)
    citation_graph = compute_citation_graph(rows)
    geo_plan = generate_geo_insights(metrics, heatmap, citation_graph)

    return BiDashboardResponse(
        global_metrics=metrics,
        competitive_matrix=matrix,
        heatmap=heatmap,
        citation_graph=citation_graph,
        geo_action_plan=geo_plan,
    )


# ---------------------------------------------------------------------------
# Async DB layer (thin fetch + delegate to pure functions)
# ---------------------------------------------------------------------------


async def _fetch_snapshot_rows(
    project_id: int,
    tenant_id: uuid.UUID,
    db,  # AsyncSession — imported lazily to keep module importable without DB
    days: int = 30,
    category: str | None = None,
    provider: str | None = None,
) -> list[SnapshotRow]:
    """Fetch snapshot rows joined with query metadata."""
    from sqlalchemy import and_, select

    from app.models.llm_query import LlmQuery
    from app.models.llm_snapshot import LlmSnapshot

    date_from = date.today() - timedelta(days=days)

    conditions = [
        LlmQuery.project_id == project_id,
        LlmSnapshot.tenant_id == tenant_id,
        LlmSnapshot.date >= date_from,
    ]
    if category:
        conditions.append(LlmQuery.category == category)
    if provider:
        conditions.append(LlmSnapshot.llm_provider == provider)

    stmt = (
        select(
            LlmSnapshot.llm_query_id,
            LlmQuery.query_type,
            LlmSnapshot.date,
            LlmSnapshot.llm_provider,
            LlmSnapshot.brand_mentioned,
            LlmSnapshot.mention_type,
            LlmSnapshot.competitor_mentions,
            LlmSnapshot.cited_urls,
            # GEO metrics (nullable — None for legacy rows)
            LlmSnapshot.sentiment_label,
            LlmSnapshot.position_rank,
            LlmSnapshot.structure_type,
            LlmSnapshot.is_hallucination,
            LlmQuery.category,
        )
        .join(LlmQuery, LlmSnapshot.llm_query_id == LlmQuery.id)
        .where(and_(*conditions))
    )

    result = await db.execute(stmt)
    return [
        SnapshotRow(
            llm_query_id=row.llm_query_id,
            query_type=row.query_type,
            date=row.date,
            llm_provider=row.llm_provider,
            brand_mentioned=row.brand_mentioned,
            mention_type=row.mention_type,
            competitor_mentions=row.competitor_mentions,
            cited_urls=row.cited_urls,
            sentiment_label=row.sentiment_label,
            position_rank=row.position_rank,
            structure_type=row.structure_type,
            is_hallucination=row.is_hallucination,
            category=row.category,
        )
        for row in result
    ]


async def get_bi_dashboard(
    project_id: int,
    tenant_id: uuid.UUID,
    db,
    days: int = 30,
    brand_name: str = "",
    category: str | None = None,
    provider: str | None = None,
) -> BiDashboardResponse:
    """Full BI dashboard: metrics + matrix + heatmap + citations + GEO advice."""
    rows = await _fetch_snapshot_rows(project_id, tenant_id, db, days, category=category, provider=provider)
    return compute_bi_dashboard(rows, brand_name=brand_name)


async def get_heatmap(
    project_id: int,
    tenant_id: uuid.UUID,
    db,
    days: int = 30,
) -> HeatmapResponse:
    """Scenario x Vendor heatmap only."""
    rows = await _fetch_snapshot_rows(project_id, tenant_id, db, days)
    return compute_heatmap(rows)


async def get_geo_advisor(
    project_id: int,
    tenant_id: uuid.UUID,
    db,
    days: int = 30,
) -> GeoActionPlan:
    """GEO recommendations only."""
    rows = await _fetch_snapshot_rows(project_id, tenant_id, db, days)
    metrics = compute_global_metrics(rows)
    heatmap = compute_heatmap(rows)
    citation_graph = compute_citation_graph(rows)
    return generate_geo_insights(metrics, heatmap, citation_graph)


# ---------------------------------------------------------------------------
# Debug trace: full computation with per-snapshot breakdown
# ---------------------------------------------------------------------------


def compute_debug_trace(rows: list[DebugSnapshotRow]) -> LlmDebugResponse:
    """Compute all metrics WITH full debug trace for every step."""

    # --- AIVS debug ---
    weighted_sum = 0.0
    weight_sum = 0.0
    per_snapshot_items: list[DebugSnapshotItem] = []

    for r in rows:
        rvs = rvs_for_mention_type(r.mention_type)
        w = intent_weight(r.query_type)
        contribution = rvs * w
        weighted_sum += contribution
        weight_sum += w

        bip = _brand_in_prompt(r.query_text, r.target_brand)

        item = DebugSnapshotItem(
            id=r.id,
            query_id=r.llm_query_id,
            query_text=r.query_text,
            query_type=r.query_type,
            date=r.date,
            provider=r.llm_provider,
            model=r.llm_model,
            brand_mentioned=r.brand_mentioned,
            mention_type=r.mention_type,
            mention_context=r.mention_context,
            competitor_mentions=r.competitor_mentions,
            cited_urls=r.cited_urls,
            tokens=r.response_tokens,
            cost=r.cost_usd,
            rvs=round(rvs, 4),
            rvs_source=f"RVS_LOOKUP['{r.mention_type}'] = {rvs}",
            intent_weight=round(w, 4),
            intent_source=f"INTENT_WEIGHTS['{r.query_type}'] = {w}",
            contribution=round(contribution, 4),
            brand_in_prompt=bip,
        )
        per_snapshot_items.append(item)

    raw_ratio = (weighted_sum / weight_sum) if weight_sum > 0 else 0.0
    aivs_final = min(raw_ratio * 100.0, 100.0) if weight_sum > 0 else 0.0
    capped = raw_ratio * 100.0 > 100.0

    aivs_debug = AivsDebug(
        final_score=round(aivs_final, 2),
        weighted_sum=round(weighted_sum, 4),
        weight_sum=round(weight_sum, 4),
        raw_ratio=round(raw_ratio, 4),
        capped=capped,
        formula=f"AIVS = ({weighted_sum:.4f} / {weight_sum:.4f}) × 100 = {aivs_final:.2f}",
        per_snapshot=per_snapshot_items,
    )

    # --- SoM debug ---
    brand_count = sum(1 for r in rows if r.brand_mentioned)
    comp_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.competitor_mentions:
            for comp, mentioned in r.competitor_mentions.items():
                if mentioned:
                    # Skip sub-brand :: keys (roll-up: parent already counted)
                    if "::" in comp:
                        continue
                    comp_counts[comp] += 1
    total_mentions = brand_count + sum(comp_counts.values())
    som_val = round((brand_count / total_mentions) * 100.0, 2) if total_mentions > 0 else 0.0

    comp_sum = sum(comp_counts.values())
    som_debug = SomDebug(
        final_score=som_val,
        brand_count=brand_count,
        total_mentions=total_mentions,
        per_competitor=dict(comp_counts),
        formula=f"SoM = {brand_count} / ({brand_count}+{comp_sum}) × 100 = {som_val:.2f}%",
    )

    # --- Resilience debug ---
    groups: dict[tuple[int, str], dict] = defaultdict(lambda: {"mentioned": 0, "total": 0, "query_text": ""})
    for r in rows:
        key = (r.llm_query_id, r.llm_provider)
        groups[key]["total"] += 1
        groups[key]["query_text"] = r.query_text
        if r.brand_mentioned:
            groups[key]["mentioned"] += 1

    group_items: list[ResilienceGroupDebug] = []
    resiliences: list[float] = []
    for (qid, prov), g in sorted(groups.items()):
        res = g["mentioned"] / g["total"] if g["total"] > 0 else 0.0
        resiliences.append(res)
        group_items.append(
            ResilienceGroupDebug(
                query_id=qid,
                query_text=g["query_text"],
                provider=prov,
                total=g["total"],
                mentioned=g["mentioned"],
                resilience=round(res, 4),
            )
        )

    resilience_avg = round(sum(resiliences) / len(resiliences), 4) if resiliences else 0.0
    res_vals_str = "+".join(f"{r:.2f}" for r in resiliences)
    resilience_debug = ResilienceDebug(
        final_score=resilience_avg,
        groups=group_items,
        formula=f"Resilience = avg({res_vals_str}) / {len(resiliences)} = {resilience_avg}",
    )

    # --- Mention Rate debug ---
    total_rows = len(rows)
    mention_rate = round(brand_count / total_rows, 4) if total_rows > 0 else 0.0

    # Split by brand_in_prompt for organic vs prompted
    organic_total = sum(1 for item in per_snapshot_items if not item.brand_in_prompt)
    organic_mentions = sum(1 for item in per_snapshot_items if not item.brand_in_prompt and item.brand_mentioned)
    organic_rate = round(organic_mentions / organic_total, 4) if organic_total > 0 else 0.0

    prompted_total = sum(1 for item in per_snapshot_items if item.brand_in_prompt)
    prompted_mentions = sum(1 for item in per_snapshot_items if item.brand_in_prompt and item.brand_mentioned)
    prompted_rate = round(prompted_mentions / prompted_total, 4) if prompted_total > 0 else 0.0

    mention_rate_debug = MentionRateDebug(
        final_rate=mention_rate,
        brand_mentions=brand_count,
        total=total_rows,
        formula=f"MentionRate (all) = {brand_count} / {total_rows} = {mention_rate}",
        organic_rate=organic_rate,
        organic_mentions=organic_mentions,
        organic_total=organic_total,
        organic_formula=f"Organic = {organic_mentions} / {organic_total} = {organic_rate}"
        if organic_total > 0
        else "No organic queries",
        prompted_rate=prompted_rate,
        prompted_mentions=prompted_mentions,
        prompted_total=prompted_total,
        prompted_formula=f"Prompted = {prompted_mentions} / {prompted_total} = {prompted_rate}"
        if prompted_total > 0
        else "No prompted queries",
    )

    return LlmDebugResponse(
        project_id=rows[0].id if rows else 0,  # overridden by caller
        days=0,  # overridden by caller
        snapshot_count=total_rows,
        aivs_debug=aivs_debug,
        som_debug=som_debug,
        resilience_debug=resilience_debug,
        mention_rate_debug=mention_rate_debug,
        snapshots=per_snapshot_items,
    )


async def _fetch_debug_rows(
    project_id: int,
    tenant_id: uuid.UUID,
    db,
    days: int = 30,
) -> list[DebugSnapshotRow]:
    """Fetch extended snapshot rows for the Debug Console."""
    from sqlalchemy import and_, select

    from app.models.llm_query import LlmQuery
    from app.models.llm_snapshot import LlmSnapshot

    date_from = date.today() - timedelta(days=days)

    stmt = (
        select(
            LlmSnapshot.id,
            LlmSnapshot.llm_query_id,
            LlmQuery.query_text,
            LlmQuery.query_type,
            LlmQuery.target_brand,
            LlmSnapshot.date,
            LlmSnapshot.llm_provider,
            LlmSnapshot.llm_model,
            LlmSnapshot.brand_mentioned,
            LlmSnapshot.mention_type,
            LlmSnapshot.mention_context,
            LlmSnapshot.competitor_mentions,
            LlmSnapshot.cited_urls,
            LlmSnapshot.response_tokens,
            LlmSnapshot.cost_usd,
        )
        .join(LlmQuery, LlmSnapshot.llm_query_id == LlmQuery.id)
        .where(
            and_(
                LlmQuery.project_id == project_id,
                LlmSnapshot.tenant_id == tenant_id,
                LlmSnapshot.date >= date_from,
            )
        )
        .order_by(LlmSnapshot.date.desc(), LlmSnapshot.llm_provider)
    )

    result = await db.execute(stmt)
    return [
        DebugSnapshotRow(
            id=row.id,
            llm_query_id=row.llm_query_id,
            query_text=row.query_text,
            query_type=row.query_type,
            date=row.date,
            llm_provider=row.llm_provider,
            llm_model=row.llm_model,
            brand_mentioned=row.brand_mentioned,
            mention_type=row.mention_type,
            mention_context=row.mention_context,
            competitor_mentions=row.competitor_mentions,
            cited_urls=row.cited_urls,
            response_tokens=row.response_tokens,
            cost_usd=row.cost_usd,
            target_brand=row.target_brand,
        )
        for row in result
    ]


async def get_debug_trace(
    project_id: int,
    tenant_id: uuid.UUID,
    db,
    days: int = 30,
) -> LlmDebugResponse:
    """Full debug trace for all metrics."""
    rows = await _fetch_debug_rows(project_id, tenant_id, db, days)
    trace = compute_debug_trace(rows)
    trace.project_id = project_id
    trace.days = days
    return trace
