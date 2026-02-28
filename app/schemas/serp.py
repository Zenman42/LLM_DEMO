from pydantic import BaseModel


class ChartDataset(BaseModel):
    label: str
    data: list[float | None]
    type: str = "serp"  # "serp" or "gsc"


class ChartDataResponse(BaseModel):
    labels: list[str]
    datasets: list[ChartDataset]


class SerpDetailItem(BaseModel):
    position: int
    url: str
    domain: str
    title: str
    snippet: str


class SerpDetailsResponse(BaseModel):
    keyword: str
    date: str
    search_engine: str
    details: list[SerpDetailItem]
