"""Pydantic response models for BI Dashboard, Aggregation & Insights."""

from datetime import datetime

from pydantic import BaseModel, Field


class GlobalMetrics(BaseModel):
    aivs: float = Field(ge=0, le=100, description="AI Visibility Score (0-100)")
    som: float = Field(ge=0, le=100, description="Share of Model (%)")
    resilience_score: float = Field(ge=0, le=1, description="Consistency across repeated runs")
    total_responses: int = Field(ge=0)
    mention_rate: float = Field(ge=0, le=1, description="Fraction of responses mentioning brand")
    # GEO Level 2 metrics
    nss: float = Field(default=0.0, ge=-100, le=100, description="Net Sentiment Score (-100 to +100)")
    top1_win_rate: float | None = Field(default=None, description="Top-1 Win Rate (%) or None if no data")
    hallucination_rate: float | None = Field(default=None, description="Feature Hallucination Rate (%) or None")
    # GEO Level 3 metrics
    intent_penetration: dict[str, float] = Field(default_factory=dict, description="AIVS per intent type")


class CompetitorRow(BaseModel):
    name: str
    som: float = Field(ge=0, le=100)
    mention_rate: float = Field(ge=0, le=1)
    avg_rvs: float = Field(ge=0)


class CompetitiveMatrix(BaseModel):
    target: CompetitorRow
    competitors: list[CompetitorRow]


class HeatmapCell(BaseModel):
    scenario: str
    vendor: str
    aivs: float = Field(ge=0, le=100)
    mention_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    zone: str = Field(pattern=r"^(red|yellow|green)$")


class HeatmapResponse(BaseModel):
    cells: list[HeatmapCell]
    scenarios: list[str]
    vendors: list[str]


class CitationDomain(BaseModel):
    domain: str
    count: int = Field(ge=0)
    providers: list[str]
    appears_in_competitor_wins: bool = False


class CitationTrustGraph(BaseModel):
    domains: list[CitationDomain]
    total_citations: int = Field(ge=0)


class GeoInsight(BaseModel):
    rule_id: str
    severity: str = Field(pattern=r"^(critical|warning|info)$")
    title_ru: str
    description_ru: str
    recommendation_ru: str


class GeoActionPlan(BaseModel):
    insights: list[GeoInsight]
    generated_at: datetime


class BiDashboardResponse(BaseModel):
    global_metrics: GlobalMetrics
    competitive_matrix: CompetitiveMatrix
    heatmap: HeatmapResponse
    citation_graph: CitationTrustGraph
    geo_action_plan: GeoActionPlan
