from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "pt_user"
    postgres_password: str = "changeme"
    postgres_db: str = "position_tracker"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_url_sync(self) -> str:
        """For Alembic migrations (sync driver)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: str = ""
    clickhouse_db: str = "position_tracker"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Auth
    jwt_secret_key: str = "change-this-to-a-random-string"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 30
    jwt_algorithm: str = "HS256"

    # Encryption for tenant credentials
    fernet_key: str = ""

    # JustMagic API
    justmagic_api_url: str = "https://api.just-magic.org/api_v1.php"

    # Yandex Webmaster API
    ywm_api_url: str = "https://api.webmaster.yandex.net/v4"
    ywm_user_id: str = ""

    # App
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Data retention
    retention_months: int = 24  # Drop partitions older than this many months

    # CORS
    allowed_origins: str = "*"  # comma-separated, e.g. "https://app.example.com,https://admin.example.com"

    # Demo mode — always on, auth bypassed, all pages accessible without login
    demo_mode: bool = True
    demo_user_email: str = "demo@llmtracker.com"

    # Logging
    log_level: str = "INFO"
    log_json: bool = False  # set True in production for structured JSON logs

    # Sentry
    sentry_dsn: str = ""  # leave empty to disable


settings = Settings()


def validate_settings_for_production() -> None:
    """Validate critical settings. Called on startup in non-test environments."""
    errors: list[str] = []

    if not settings.demo_mode:
        if settings.jwt_secret_key in ("change-this-to-a-random-string", ""):
            errors.append("JWT_SECRET_KEY must be set to a secure random value")

        if len(settings.jwt_secret_key) < 32:
            errors.append("JWT_SECRET_KEY must be at least 32 characters")

    if not settings.fernet_key:
        errors.append(
            'FERNET_KEY must be set (generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")'
        )

    if settings.app_env == "production":
        if settings.allowed_origins == "*":
            errors.append("ALLOWED_ORIGINS must not be '*' in production")
        if settings.app_debug:
            errors.append("APP_DEBUG must be false in production")

    if errors:
        raise SystemExit("Configuration errors:\n  - " + "\n  - ".join(errors))
