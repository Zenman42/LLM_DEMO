"""add DeepSeek and YandexGPT provider keys, remove unused google_ai/anthropic

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-17 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add new LLM provider columns
    op.add_column("tenants", sa.Column("deepseek_api_key", sa.LargeBinary(), nullable=True))
    op.add_column("tenants", sa.Column("yandexgpt_api_key", sa.LargeBinary(), nullable=True))
    op.add_column("tenants", sa.Column("yandexgpt_folder_id", sa.LargeBinary(), nullable=True))

    # Remove unused provider columns
    op.drop_column("tenants", "google_ai_api_key")
    op.drop_column("tenants", "anthropic_api_key")


def downgrade() -> None:
    # Re-add removed columns
    op.add_column("tenants", sa.Column("anthropic_api_key", sa.LargeBinary(), nullable=True))
    op.add_column("tenants", sa.Column("google_ai_api_key", sa.LargeBinary(), nullable=True))

    # Remove new columns
    op.drop_column("tenants", "yandexgpt_folder_id")
    op.drop_column("tenants", "yandexgpt_api_key")
    op.drop_column("tenants", "deepseek_api_key")
