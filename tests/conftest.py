import asyncio
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings

# Override settings for tests
settings.jwt_secret_key = "test-secret-key-that-is-at-least-32-bytes-long"
settings.fernet_key = "KxJCocbnA3KD20pkgSN3uUZybasKP1X9lAJDX4oLxoQ="  # test-only Fernet key
settings.app_env = "development"

from app.core.security import create_access_token  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.postgres import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models.tenant import Tenant  # noqa: E402
from app.models.user import User  # noqa: E402

# Use a test database — NullPool avoids asyncpg connection conflicts between tests
TEST_DB_URL = settings.postgres_url.replace("/position_tracker", "/position_tracker_test")

test_engine = create_async_engine(TEST_DB_URL, echo=False, poolclass=NullPool)
test_session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def setup_db():
    """Create all tables before each test and drop after."""
    async with test_engine.begin() as conn:
        # Enable pgvector if available; tests that need vector columns
        # require a pgvector-enabled PostgreSQL (e.g. pgvector/pgvector:pg16).
        # Uses a SAVEPOINT so that a failure doesn't abort the outer transaction
        # (PostgreSQL marks the whole transaction as failed after any error).
        pgvector_ok = False
        try:
            async with conn.begin_nested():
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            pgvector_ok = True
        except Exception:
            pass

        if not pgvector_ok:
            # pgvector extension unavailable — swap the Vector(1536) column
            # to LargeBinary so entity_profiles table can still be created
            # (discovered_entities has an FK to it).  Tests needing real
            # vector operations will fail individually.
            from sqlalchemy import LargeBinary

            ep_table = Base.metadata.tables.get("entity_profiles")
            if ep_table is not None and "embedding" in ep_table.c:
                ep_table.c.embedding.type = LargeBinary()

        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with test_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    async with test_session_factory() as session:
        yield session


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def tenant_and_user(db: AsyncSession) -> tuple[Tenant, User]:
    """Create a test tenant and user."""
    from app.core.security import hash_password

    tenant = Tenant(name="Test Corp", slug="test-corp")
    db.add(tenant)
    await db.flush()

    user = User(
        tenant_id=tenant.id,
        email="test@example.com",
        password_hash=hash_password("testpassword123"),
        role="owner",
    )
    db.add(user)
    await db.commit()
    return tenant, user


@pytest.fixture
async def auth_headers(tenant_and_user: tuple[Tenant, User]) -> dict[str, str]:
    """Get auth headers with a valid access token."""
    tenant, user = tenant_and_user
    token = create_access_token(user.id, tenant.id)
    return {"Authorization": f"Bearer {token}"}
