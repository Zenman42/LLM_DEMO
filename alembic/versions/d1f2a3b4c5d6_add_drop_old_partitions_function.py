"""add drop_old_partitions function

Revision ID: d1f2a3b4c5d6
Revises: c3a1f7d84e20
Create Date: 2026-02-16 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "d1f2a3b4c5d6"
down_revision: str | None = "c3a1f7d84e20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION drop_old_partitions(
            p_table TEXT,
            p_months_retain INT DEFAULT 24
        ) RETURNS void AS $$
        DECLARE
            v_cutoff DATE;
            v_part RECORD;
            v_part_date DATE;
        BEGIN
            v_cutoff := date_trunc('month', CURRENT_DATE)
                        - (p_months_retain || ' months')::INTERVAL;

            FOR v_part IN
                SELECT inhrelid::regclass::text AS partition_name
                FROM pg_inherits
                WHERE inhparent = p_table::regclass
            LOOP
                IF v_part.partition_name ~ '_\\d{4}_\\d{2}$' THEN
                    BEGIN
                        v_part_date := to_date(
                            regexp_replace(
                                v_part.partition_name,
                                '.*_(\\d{4}_\\d{2})$', '\\1'
                            ),
                            'YYYY_MM'
                        );
                        IF v_part_date < v_cutoff THEN
                            EXECUTE format(
                                'DROP TABLE IF EXISTS %I',
                                v_part.partition_name
                            );
                            RAISE NOTICE 'Dropped old partition: %',
                                         v_part.partition_name;
                        END IF;
                    EXCEPTION WHEN others THEN
                        RAISE WARNING 'Could not process partition %: %',
                                       v_part.partition_name, SQLERRM;
                    END;
                END IF;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS drop_old_partitions(TEXT, INT)")
