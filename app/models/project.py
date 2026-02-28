import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str | None] = mapped_column(String(255), nullable=True)
    search_engine: Mapped[str] = mapped_column(String(10), default="both")  # google / yandex / both
    region_google: Mapped[str | None] = mapped_column(String(200), nullable=True)
    region_yandex: Mapped[int] = mapped_column(Integer, default=213)  # 213 = Moscow
    ywm_host_id: Mapped[str | None] = mapped_column(String(200), nullable=True)  # e.g. "https:domain.com:443"
    gsc_site_url: Mapped[str | None] = mapped_column(String(200), nullable=True)  # e.g. "sc-domain:domain.com"
    # LLM tracking
    track_llm: Mapped[bool] = mapped_column(Boolean, default=False)
    track_ai_overview: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_providers: Mapped[list | None] = mapped_column(
        JSONB, default=list
    )  # ["chatgpt", "deepseek", "perplexity", "yandexgpt"]
    brand_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    brand_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # LLM Onboarding profile (audit profile fields stored on project)
    brand_aliases: Mapped[list | None] = mapped_column(JSONB, default=list)
    competitors: Mapped[list | None] = mapped_column(JSONB, default=list)
    categories: Mapped[list | None] = mapped_column(JSONB, default=list)  # product/service use-cases
    golden_facts: Mapped[list | None] = mapped_column(JSONB, default=list)  # anti-hallucination facts
    personas: Mapped[list | None] = mapped_column(JSONB, default=list)  # ["beginner", "cto", ...]
    # Sub-brand hierarchy
    brand_sub_brands: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    # {"ЗИЛАРТ": ["ЖК Зиларт", "Zilart"], "Сердце Столицы": ["ЖК Сердце Столицы"]}
    competitor_sub_brands: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    # {"ПИК": {"Саларьево Парк": ["ЖК Саларьево"]}, "Самолёт": {}}
    market: Mapped[str | None] = mapped_column(String(10), nullable=True)  # "ru" or "en"
    geo: Mapped[str | None] = mapped_column(String(255), nullable=True)  # "Russia", "Moscow"

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="projects")  # noqa: F821
    keywords: Mapped[list["Keyword"]] = relationship("Keyword", back_populates="project", cascade="all, delete-orphan")  # noqa: F821
    llm_queries: Mapped[list["LlmQuery"]] = relationship(  # noqa: F821
        "LlmQuery", back_populates="project", cascade="all, delete-orphan"
    )
    query_tags: Mapped[list["QueryTag"]] = relationship(  # noqa: F821
        "QueryTag", back_populates="project", cascade="all, delete-orphan"
    )
    user_projects: Mapped[list["UserProject"]] = relationship(  # noqa: F821
        "UserProject", back_populates="project", cascade="all, delete-orphan"
    )
