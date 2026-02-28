from datetime import datetime

from pydantic import BaseModel


class AiOverviewSnapshotResponse(BaseModel):
    id: int
    keyword_id: int
    date: str
    has_ai_overview: bool
    brand_in_aio: bool
    aio_position: int | None
    aio_sources: list[dict] | None
    aio_snippet: str | None
    collected_at: datetime

    model_config = {"from_attributes": True}


class AiOverviewDashboardStats(BaseModel):
    total_keywords: int
    keywords_with_aio: int
    aio_coverage_rate: float  # % of keywords that have AI Overview
    brand_in_aio_rate: float  # % of AIO blocks where our brand appears
    avg_aio_position: float | None


class AiOverviewChartDataset(BaseModel):
    label: str
    data: list[float | None]


class AiOverviewChartResponse(BaseModel):
    labels: list[str]  # dates
    datasets: list[AiOverviewChartDataset]
