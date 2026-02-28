import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, LargeBinary, SmallInteger, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    plan: Mapped[str] = mapped_column(String(20), default="free")
    max_projects: Mapped[int] = mapped_column(Integer, default=5)
    max_keywords: Mapped[int] = mapped_column(Integer, default=500)
    max_llm_queries: Mapped[int] = mapped_column(Integer, default=50)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Encrypted API credentials
    justmagic_api_key: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    ywm_token: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    ywm_host_id: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    gsc_credentials_json: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    telegram_bot_token: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    telegram_chat_id: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # LLM API credentials (encrypted)
    openai_api_key: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    deepseek_api_key: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    perplexity_api_key: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    yandexgpt_api_key: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    yandexgpt_folder_id: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    gemini_api_key: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    gigachat_api_key: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # Collection schedule
    collection_hour: Mapped[int] = mapped_column(SmallInteger, default=6)
    collection_minute: Mapped[int] = mapped_column(SmallInteger, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    users: Mapped[list["User"]] = relationship("User", back_populates="tenant", cascade="all, delete-orphan")  # noqa: F821
    projects: Mapped[list["Project"]] = relationship("Project", back_populates="tenant", cascade="all, delete-orphan")  # noqa: F821
