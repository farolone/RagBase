"""URL-based deduplication via PostgreSQL."""

import psycopg
from rag.config import settings


def is_already_ingested(source_url: str) -> bool:
    """Check if a URL has already been ingested."""
    conninfo = (
        f"host={settings.postgres_host} "
        f"port={settings.postgres_port} "
        f"dbname={settings.postgres_db} "
        f"user={settings.postgres_user} "
        f"password={settings.postgres_password}"
    )
    try:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM documents WHERE source_url = %s LIMIT 1",
                    (source_url,),
                )
                return cur.fetchone() is not None
    except Exception:
        return False
