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

    # --- Document listing ---

    def list_documents(
        self,
        platform: str | None = None,
        author: str | None = None,
        collection_id: str | None = None,
        flagged: bool | None = None,
        search: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[dict], int]:
        conditions = []
        params: list = []

        if platform:
            conditions.append("d.platform = %s")
            params.append(platform)
        if author:
            conditions.append("d.author = %s")
            params.append(author)
        if flagged is not None:
            conditions.append("d.flagged = %s")
            params.append(flagged)
        if search:
            conditions.append("d.title ILIKE %s")
            params.append(f"%{search}%")
        if collection_id:
            conditions.append("EXISTS (SELECT 1 FROM document_collections dc WHERE dc.document_id = d.id AND dc.collection_id = %s)")
            params.append(collection_id)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) as cnt FROM documents d {where}", params)
                total = cur.fetchone()["cnt"]

                cur.execute(
                    f"""SELECT d.*,
                        COALESCE(d.chunk_count, 0) as chunk_count,
                        COALESCE(d.entity_count, 0) as entity_count
                    FROM documents d {where}
                    ORDER BY d.ingested_at DESC
                    LIMIT %s OFFSET %s""",
                    params + [limit, offset],
                )
                rows = cur.fetchall()
                return rows, total

    def get_document_detail(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
                doc = cur.fetchone()
                if not doc:
                    return None

                cur.execute(
                    """SELECT t.* FROM tags t
                    JOIN document_tags dt ON dt.tag_id = t.id
                    WHERE dt.document_id = %s""",
                    (doc_id,),
                )
                tags = cur.fetchall()

                cur.execute(
                    """SELECT c.* FROM collections c
                    JOIN document_collections dc ON dc.collection_id = c.id
                    WHERE dc.document_id = %s""",
                    (doc_id,),
                )
                collections = cur.fetchall()

                doc["tags"] = tags
                doc["collections"] = collections
                return doc

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted

    def update_document_counts(self, doc_id: str, chunk_count: int, entity_count: int):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET chunk_count = %s, entity_count = %s WHERE id = %s",
                    (chunk_count, entity_count, doc_id),
                )
            conn.commit()

    def update_document_quality(self, doc_id: str, quality_score: int):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET quality_score = %s WHERE id = %s",
                    (quality_score, doc_id),
                )
            conn.commit()

    def flag_document(self, doc_id: str, flagged: bool, reason: str | None = None):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET flagged = %s WHERE id = %s",
                    (flagged, doc_id),
                )
                if flagged and reason:
                    cur.execute(
                        """INSERT INTO source_ratings (document_id, flagged, flag_reason)
                        VALUES (%s, TRUE, %s)""",
                        (doc_id, reason),
                    )
            conn.commit()

    # --- Transcripts ---

    def save_transcript(self, document_id: str, segments: list[dict], source: str = "api"):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO document_transcripts (document_id, segments, source)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (document_id) DO UPDATE SET
                        segments = EXCLUDED.segments,
                        source = EXCLUDED.source""",
                    (document_id, json.dumps(segments), source),
                )
            conn.commit()

    def get_transcript(self, document_id: str) -> dict | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT segments, source FROM document_transcripts WHERE document_id = %s",
                    (document_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return {"segments": row["segments"], "source": row["source"]}

    def update_published_at(self, document_id: str, published_at: datetime):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET published_at = %s WHERE id = %s",
                    (published_at, document_id),
                )
            conn.commit()

    def backfill_published_at(self) -> int:
        """Backfill published_at from metadata.upload_date (YouTube) and metadata.date (Web)."""
        count = 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                # YouTube: upload_date as YYYYMMDD
                cur.execute(
                    """UPDATE documents
                    SET published_at = to_timestamp(metadata->>'upload_date', 'YYYYMMDD')
                    WHERE published_at IS NULL
                      AND metadata->>'upload_date' IS NOT NULL
                      AND length(metadata->>'upload_date') = 8"""
                )
                count += cur.rowcount

                # Web: date as YYYY-MM-DD
                cur.execute(
                    """UPDATE documents
                    SET published_at = (metadata->>'date')::timestamptz
                    WHERE published_at IS NULL
                      AND metadata->>'date' IS NOT NULL
                      AND metadata->>'date' ~ '^\d{4}-\d{2}-\d{2}'"""
                )
                count += cur.rowcount
            conn.commit()
        return count

    # --- Collections ---

    def list_collections(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT c.*, COUNT(dc.document_id) as doc_count
                    FROM collections c
                    LEFT JOIN document_collections dc ON dc.collection_id = c.id
                    GROUP BY c.id ORDER BY c.name"""
                )
                return cur.fetchall()

    def create_collection(self, name: str, description: str = "", color: str = "#3B82F6") -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO collections (name, description, color) VALUES (%s, %s, %s) RETURNING *",
                    (name, description, color),
                )
                row = cur.fetchone()
            conn.commit()
            return row

    def get_collection(self, collection_id: str) -> dict | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT c.*, COUNT(dc.document_id) as doc_count
                    FROM collections c
                    LEFT JOIN document_collections dc ON dc.collection_id = c.id
                    WHERE c.id = %s GROUP BY c.id""",
                    (collection_id,),
                )
                return cur.fetchone()

    def update_collection(self, collection_id: str, name: str, description: str, color: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE collections SET name = %s, description = %s, color = %s WHERE id = %s",
                    (name, description, color, collection_id),
                )
                updated = cur.rowcount > 0
            conn.commit()
            return updated

    def delete_collection(self, collection_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM collections WHERE id = %s", (collection_id,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted

    def add_document_to_collection(self, document_id: str, collection_id: str):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO document_collections (document_id, collection_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (document_id, collection_id),
                )
            conn.commit()

    def remove_document_from_collection(self, document_id: str, collection_id: str):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_collections WHERE document_id = %s AND collection_id = %s",
                    (document_id, collection_id),
                )
            conn.commit()

    def get_collection_document_ids(self, collection_id: str) -> list[str]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT document_id FROM document_collections WHERE collection_id = %s",
                    (collection_id,),
                )
                return [row["document_id"] for row in cur.fetchall()]

    # --- Tags ---

    def list_tags(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT t.*, COUNT(dt.document_id) as doc_count
                    FROM tags t
                    LEFT JOIN document_tags dt ON dt.tag_id = t.id
                    GROUP BY t.id ORDER BY t.name"""
                )
                return cur.fetchall()

    def create_tag(self, name: str, tag_type: str = "manual", color: str = "#6B7280") -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO tags (name, tag_type, color) VALUES (%s, %s, %s) ON CONFLICT (name) DO UPDATE SET tag_type=EXCLUDED.tag_type RETURNING *",
                    (name, tag_type, color),
                )
                row = cur.fetchone()
            conn.commit()
            return row

    def delete_tag(self, tag_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_tags WHERE tag_id = %s", (tag_id,))
                cur.execute("DELETE FROM tags WHERE id = %s", (tag_id,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted

    def set_document_tags(self, document_id: str, tag_ids: list[str]):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_tags WHERE document_id = %s", (document_id,))
                for tag_id in tag_ids:
                    cur.execute(
                        "INSERT INTO document_tags (document_id, tag_id) VALUES (%s, %s)",
                        (document_id, tag_id),
                    )
            conn.commit()

    # --- Chat sessions ---

    def create_chat_session(self, title: str = "New Chat", collection_id: str | None = None) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_sessions (title, collection_id) VALUES (%s, %s) RETURNING *",
                    (title, collection_id),
                )
                row = cur.fetchone()
            conn.commit()
            return row

    def list_chat_sessions(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT cs.*, COUNT(cm.id) as message_count
                    FROM chat_sessions cs
                    LEFT JOIN chat_messages cm ON cm.session_id = cs.id
                    GROUP BY cs.id ORDER BY cs.created_at DESC"""
                )
                return cur.fetchall()

    def get_chat_messages(self, session_id: str) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM chat_messages WHERE session_id = %s ORDER BY created_at",
                    (session_id,),
                )
                return cur.fetchall()

    def save_chat_message(self, session_id: str, role: str, content: str, source_chunks: list | None = None) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_messages (session_id, role, content, source_chunks) VALUES (%s, %s, %s, %s) RETURNING *",
                    (session_id, role, content, json.dumps(source_chunks or [])),
                )
                row = cur.fetchone()
            conn.commit()
            return row

    def delete_chat_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted

    # --- Feedback ---

    def save_feedback(self, session_id: str | None, question: str, answer: str, rating: int, comment: str = ""):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO answer_feedback (session_id, question, answer, rating, comment) VALUES (%s, %s, %s, %s, %s)",
                    (session_id, question, answer, rating, comment),
                )
            conn.commit()

    # --- Dedup ---

    def document_exists_by_url(self, source_url: str) -> bool:
        """Check if a document with this source_url already exists."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM documents WHERE source_url = %s LIMIT 1",
                    (source_url,),
                )
                return cur.fetchone() is not None

    # --- Stats ---

    def get_platform_stats(self) -> dict:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT platform, COUNT(*) as count FROM documents GROUP BY platform")
                return {row["platform"]: row["count"] for row in cur.fetchall()}

    def get_ingestion_timeline(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT DATE(ingested_at) as date, COUNT(*) as count
                    FROM documents
                    WHERE ingested_at > NOW() - INTERVAL '30 days'
                    GROUP BY DATE(ingested_at) ORDER BY date"""
                )
                return [{"date": str(row["date"]), "count": row["count"]} for row in cur.fetchall()]

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
