import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DiscoveredEntity(Base):
    """Entity discovered in LLM responses but not yet matched to a known competitor."""

    __tablename__ = "discovered_entities"
    __table_args__ = (
        UniqueConstraint("project_id", "entity_name", name="uq_project_entity"),
        Index("ix_discovered_entity_project", "project_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    entity_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen: Mapped[date] = mapped_column(Date, nullable=False)
    last_seen: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending | confirmed | rejected | promoted | alias
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verified_by: Mapped[str | None] = mapped_column(String(20), nullable=True)  # "perplexity" | "openai" | "manual"
    suggested_parent: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # e.g. "ПИК" if LLM thinks this is a sub-brand
    llm_explanation: Mapped[str | None] = mapped_column(String(1000), nullable=True)  # LLM verification explanation

    # Entity classification (populated by ProLEA profiler)
    entity_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # company | product | person | location | service | other

    # Link to semantic profile used for vector-based clustering
    profile_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("entity_profiles.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Entity-to-entity alias: when set, this entity is an alternative name for the target entity.
    # Aliased entities are hidden from discovered list and their mentions aggregate to the target.
    alias_of_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("discovered_entities.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Self-referential relationship: entity.aliases = list of entities aliased to this one
    aliases: Mapped[list["DiscoveredEntity"]] = relationship(
        "DiscoveredEntity",
        back_populates="alias_target",
        foreign_keys="DiscoveredEntity.alias_of_id",
    )
    alias_target: Mapped["DiscoveredEntity | None"] = relationship(
        "DiscoveredEntity",
        back_populates="aliases",
        remote_side="DiscoveredEntity.id",
        foreign_keys="DiscoveredEntity.alias_of_id",
    )
