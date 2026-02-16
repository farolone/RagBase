import json
from datetime import datetime

import psycopg
from psycopg.rows import dict_row

from rag.config import settings
from rag.models import Document, Platform


class PostgresStore:
    def __init__(self):
        self.conninfo = (
            f"host={settings.postgres_host} "
            f"port={settings.postgres_port} "
            f"dbname={settings.postgres_db} "
            f"user={settings.postgres_user} "
            f"password={settings.postgres_password}"
        )
        self._test_ids: list[str] = []

    def _connect(self):
        return psycopg.connect(self.conninfo, row_factory=dict_row)

    def save_document(self, doc: Document):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (id, title, source_url, platform, author, language, created_at, ingested_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        source_url = EXCLUDED.source_url,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        doc.id,
                        doc.title,
                        doc.source_url,
                        doc.platform.value,
                        doc.author,
                        doc.language,
                        doc.created_at,
                        doc.ingested_at,
                        json.dumps(doc.metadata),
                    ),
                )
            conn.commit()
        self._test_ids.append(doc.id)

    def get_document(self, doc_id: str) -> Document | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_document(row)

    def search_documents(
        self,
        platform: Platform | None = None,
        author: str | None = None,
        limit: int = 100,
    ) -> list[Document]:
        conditions = []
        params: list = []

        if platform:
            conditions.append("platform = %s")
            params.append(platform.value)
        if author:
            conditions.append("author = %s")
            params.append(author)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM documents {where} ORDER BY ingested_at DESC LIMIT %s",
                    params,
                )
                return [self._row_to_document(row) for row in cur.fetchall()]

    def cleanup_test_data(self):
        if not self._test_ids:
            return
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE id = ANY(%s)",
                    (self._test_ids,),
                )
            conn.commit()
        self._test_ids.clear()

    @staticmethod
    def _row_to_document(row: dict) -> Document:
        return Document(
            id=str(row["id"]),
            title=row["title"],
            source_url=row["source_url"],
            platform=Platform(row["platform"]),
            author=row["author"],
            language=row["language"],
            created_at=row["created_at"],
            ingested_at=row["ingested_at"],
            metadata=row["metadata"] if isinstance(row["metadata"], dict) else {},
        )
