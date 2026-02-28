#!/bin/sh
set -e

# Defaults for supervisor env vars
export PORT="${PORT:-8000}"
export WEB_WORKERS="${WEB_WORKERS:-2}"
export CELERY_CONCURRENCY="${CELERY_CONCURRENCY:-2}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Run database migrations before starting services.
# Applies each pending migration individually so that a failure in one
# does not silently skip later migrations via `stamp head`.
echo "Running database migrations..."
alembic upgrade head || {
    echo "WARNING: alembic upgrade head failed. Attempting individual migrations..."
    # Get current revision and head revision
    CURRENT=$(alembic current 2>/dev/null | grep -oP '^\w+' | head -1)
    if [ -z "$CURRENT" ]; then
        echo "No current revision found — stamping base and retrying..."
        alembic stamp base 2>/dev/null || true
    fi
    # Retry upgrade — let errors be visible this time
    alembic upgrade head || echo "ERROR: Migration failed. Some features may not work."
}

# Ensure brand_description is TEXT regardless of migration state.
# (The VARCHAR→TEXT migration may have been skipped by stamp.)
python -c "
from app.core.config import settings
from sqlalchemy import create_engine, text
e = create_engine(settings.postgres_url_sync)
with e.begin() as c:
    r = c.execute(text(\"\"\"
        SELECT data_type FROM information_schema.columns
        WHERE table_name='projects' AND column_name='brand_description'
    \"\"\")).fetchone()
    if r and r[0] != 'text':
        c.execute(text('ALTER TABLE projects ALTER COLUMN brand_description TYPE TEXT'))
        print('brand_description: upgraded to TEXT')
    else:
        print('brand_description: already TEXT (ok)')
"

# Ensure alias_of_id column exists (may have been skipped by stamp head).
python -c "
from app.core.config import settings
from sqlalchemy import create_engine, text
e = create_engine(settings.postgres_url_sync)
with e.begin() as c:
    r = c.execute(text(\"\"\"
        SELECT column_name FROM information_schema.columns
        WHERE table_name='discovered_entities' AND column_name='alias_of_id'
    \"\"\")).fetchone()
    if not r:
        print('alias_of_id column missing — applying schema fix...')
        c.execute(text('''
            ALTER TABLE discovered_entities
            ADD COLUMN alias_of_id INTEGER REFERENCES discovered_entities(id) ON DELETE SET NULL
        '''))
        c.execute(text('''
            CREATE INDEX IF NOT EXISTS ix_discovered_entity_alias_of
            ON discovered_entities (alias_of_id)
        '''))
        print('alias_of_id: column added')
    else:
        print('alias_of_id: already exists (ok)')
"

# Ensure entity_type and profile_id columns exist on discovered_entities.
# These were added to the model in PR #31 but the ALTER TABLE migration was missing.
python -c "
from app.core.config import settings
from sqlalchemy import create_engine, text
e = create_engine(settings.postgres_url_sync)
with e.begin() as c:
    # entity_type column
    r = c.execute(text(\"\"\"
        SELECT column_name FROM information_schema.columns
        WHERE table_name='discovered_entities' AND column_name='entity_type'
    \"\"\")).fetchone()
    if not r:
        print('entity_type column missing — applying schema fix...')
        c.execute(text('''
            ALTER TABLE discovered_entities
            ADD COLUMN entity_type VARCHAR(50)
        '''))
        print('entity_type: column added')
    else:
        print('entity_type: already exists (ok)')

    # profile_id column (FK added only if entity_profiles table exists)
    r = c.execute(text(\"\"\"
        SELECT column_name FROM information_schema.columns
        WHERE table_name='discovered_entities' AND column_name='profile_id'
    \"\"\")).fetchone()
    if not r:
        print('profile_id column missing — applying schema fix...')
        has_profiles_table = c.execute(text(\"\"\"
            SELECT 1 FROM information_schema.tables
            WHERE table_name='entity_profiles'
        \"\"\")).fetchone()
        if has_profiles_table:
            c.execute(text('''
                ALTER TABLE discovered_entities
                ADD COLUMN profile_id INTEGER REFERENCES entity_profiles(id) ON DELETE SET NULL
            '''))
        else:
            c.execute(text('''
                ALTER TABLE discovered_entities
                ADD COLUMN profile_id INTEGER
            '''))
        print('profile_id: column added')
    else:
        print('profile_id: already exists (ok)')
"

# Bump project limit for all tenants still at the old default of 5.
python -c "
from app.core.config import settings
from sqlalchemy import create_engine, text
e = create_engine(settings.postgres_url_sync)
with e.begin() as c:
    r = c.execute(text(\"UPDATE tenants SET max_projects = 10 WHERE max_projects < 10\"))
    print(f'tenant max_projects bumped for {r.rowcount} row(s)')
"

echo "Migrations complete."

exec supervisord -c /app/supervisord.conf
