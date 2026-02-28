import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class QueryTag(Base):
    """A tag that can be applied to LLM queries.

    Two types:
      - 'class': branded / thematic (replaces query_class column)
      - 'scenario': user-defined scenario (many-to-many, replaces category column)
    """

    __tablename__ = "query_tags"
    __table_args__ = (UniqueConstraint("project_id", "tag_name", "tag_type", name="uq_query_tag_per_project"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tag_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # 'class' | 'scenario'
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="query_tags")  # noqa: F821
    query_links: Mapped[list["QueryTagLink"]] = relationship(
        "QueryTagLink", back_populates="tag", cascade="all, delete-orphan"
    )


class QueryTagLink(Base):
    """Junction table linking LLM queries to tags (many-to-many)."""

    __tablename__ = "query_tag_links"
    __table_args__ = (UniqueConstraint("query_id", "tag_id", name="uq_query_tag_link"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("llm_queries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("query_tags.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Relationships
    query: Mapped["LlmQuery"] = relationship("LlmQuery", back_populates="tag_links")  # noqa: F821
    tag: Mapped["QueryTag"] = relationship("QueryTag", back_populates="query_links")
