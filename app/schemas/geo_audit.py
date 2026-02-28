"""Pydantic response models for GEO Audit — site crawl & content analysis."""

from pydantic import BaseModel, Field


# ── Crawl Access ─────────────────────────────────────────────────

class BotAccessRow(BaseModel):
    bot: str = Field(description="Bot name, e.g. Googlebot, GPTBot, ClaudeBot")
    allowed: bool | None = Field(default=None, description="True=allowed, False=blocked, None=not mentioned")
    crawl_delay: int | None = Field(default=None, ge=0, description="Crawl-delay in seconds if specified")


class MetaRobotsRow(BaseModel):
    url: str
    meta_robots: str | None = Field(default=None, description="Content of <meta name=robots>")
    x_robots_tag: str | None = Field(default=None, description="X-Robots-Tag response header")


class CrawlAccessSection(BaseModel):
    robots_bots: list[BotAccessRow] = Field(default_factory=list)
    meta_robots: list[MetaRobotsRow] = Field(default_factory=list)
    sitemap_found: bool = False
    sitemap_url: str | None = None


# ── Schema.org Audit ─────────────────────────────────────────────

class SchemaTypeFound(BaseModel):
    type_name: str = Field(description="JSON-LD @type, e.g. Organization, Product")
    page_url: str
    properties_count: int = Field(ge=0, description="Number of top-level properties in the object")


class SchemaAuditSection(BaseModel):
    types_found: list[SchemaTypeFound] = Field(default_factory=list)
    recommended_missing: list[str] = Field(
        default_factory=list,
        description="Schema types recommended for GEO but not found on any scanned page",
    )


# ── Content Audit ────────────────────────────────────────────────

class AnswerFirstPageRow(BaseModel):
    url: str
    title: str = ""
    first_p_length: int = Field(ge=0, description="Character count of first <p> text")
    has_summary_pattern: bool = Field(default=False, description="Contains TL;DR / key takeaway / in short")
    score: float = Field(ge=0, le=100, description="Answer-First composite score")


class EeatPageRow(BaseModel):
    url: str
    has_author: bool = False
    has_date_published: bool = False
    has_references: bool = Field(default=False, description="Has outbound citation-style links")
    has_about_page_link: bool = False
    schema_author: bool = Field(default=False, description="Author present in JSON-LD structured data")


class PageFreshnessRow(BaseModel):
    url: str
    last_modified_header: str | None = None
    json_ld_date_modified: str | None = None
    meta_article_modified: str | None = None
    freshness_days: int | None = Field(default=None, ge=0, description="Days since most recent date signal")


class ContentAuditSection(BaseModel):
    answer_first: list[AnswerFirstPageRow] = Field(default_factory=list)
    eeat: list[EeatPageRow] = Field(default_factory=list)
    freshness: list[PageFreshnessRow] = Field(default_factory=list)


# ── Top-level response ──────────────────────────────────────────

class SiteAuditResponse(BaseModel):
    project_id: int
    domain: str
    pages_scanned: int = Field(ge=0)
    crawl_access: CrawlAccessSection = Field(default_factory=CrawlAccessSection)
    schema_audit: SchemaAuditSection = Field(default_factory=SchemaAuditSection)
    content_audit: ContentAuditSection = Field(default_factory=ContentAuditSection)
