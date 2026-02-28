from datetime import date, datetime

from pydantic import BaseModel


class LlmSnapshotResponse(BaseModel):
    id: int
    llm_query_id: int
    date: date
    llm_provider: str
    llm_model: str
    brand_mentioned: bool
    mention_type: str
    mention_context: str | None
    competitor_mentions: dict | None
    cited_urls: list[str] | None
    response_tokens: int | None
    cost_usd: float | None
    collected_at: datetime

    model_config = {"from_attributes": True}


class LlmSnapshotDetail(LlmSnapshotResponse):
    raw_response: str | None


class LlmChartDataset(BaseModel):
    label: str  # provider name
    data: list[float | None]  # mention rate per day (0.0 - 1.0)


class LlmChartDataResponse(BaseModel):
    labels: list[str]  # dates
    datasets: list[LlmChartDataset]


class LlmDashboardStats(BaseModel):
    total_queries: int
    total_checks: int
    brand_mention_rate: float  # 0.0 - 1.0 across all providers
    mention_rate_by_provider: dict[str, float]  # {"chatgpt": 0.75, "gemini": 0.6}
    sov: float  # share of voice (brand mentions / total brand+competitor mentions)
    sov_by_provider: dict[str, float]
    top_cited_urls: list[dict]  # [{"url": "...", "count": 5}]
    total_cost_usd: float
    competitor_sov: dict[str, float]  # {"SEMrush": 0.3, "Moz": 0.1}
    competitor_sov_by_provider: dict[str, dict[str, float]] = {}  # {"chatgpt": {"SEMrush": 0.3}}


class LlmCitationItem(BaseModel):
    url: str
    domain: str
    count: int
    providers: list[str]


class LlmCitationsResponse(BaseModel):
    citations: list[LlmCitationItem]
    total: int
