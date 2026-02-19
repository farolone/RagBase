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

    # --- New methods for frontend ---

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
        """List documents with pagination, filters, and total count."""
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
        """Get document with tags, collections, and ratings."""
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
        """Delete document from PostgreSQL (cascades to junction tables)."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted

    def update_document_counts(self, doc_id: str, chunk_count: int, entity_count: int):
        """Update chunk and entity counts for a document."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET chunk_count = %s, entity_count = %s WHERE id = %s",
                    (chunk_count, entity_count, doc_id),
                )
            conn.commit()

    def update_document_quality(self, doc_id: str, quality_score: int):
        """Set quality score (1-5) for a document."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET quality_score = %s WHERE id = %s",
                    (quality_score, doc_id),
                )
            conn.commit()

    def flag_document(self, doc_id: str, flagged: bool, reason: str | None = None):
        """Flag/unflag a document."""
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

    # --- Tags ---

    def list_tags(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM tags ORDER BY name")
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

    # --- Stats ---

    def get_platform_stats(self) -> dict:
        """Get document counts per platform."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT platform, COUNT(*) as count FROM documents GROUP BY platform")
                return {row["platform"]: row["count"] for row in cur.fetchall()}

    def get_ingestion_timeline(self) -> list[dict]:
        """Get ingestion counts per day for the last 30 days."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT DATE(ingested_at) as date, COUNT(*) as count
                    FROM documents
                    WHERE ingested_at > NOW() - INTERVAL '30 days'
                    GROUP BY DATE(ingested_at) ORDER BY date"""
                )
                return [{"date": str(row["date"]), "count": row["count"]} for row in cur.fetchall()]

    # --- Source Configs ---

    def ensure_source_configs_table(self):
        """Create source_configs table if it doesn't exist."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS source_configs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        source_type TEXT NOT NULL,
                        name TEXT NOT NULL,
                        config JSONB NOT NULL DEFAULT '{}'::jsonb,
                        collection_id UUID REFERENCES collections(id) ON DELETE SET NULL,
                        enabled BOOLEAN DEFAULT TRUE,
                        last_run_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_source_configs_type ON source_configs(source_type)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_source_configs_collection ON source_configs(collection_id)")
            conn.commit()

    def list_source_configs(self, source_type: str | None = None) -> list[dict]:
        """List all source configs with optional type filter, joined with collection name."""
        self.ensure_source_configs_table()
        conditions = []
        params: list = []
        if source_type:
            conditions.append("sc.source_type = %s")
            params.append(source_type)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT sc.*, c.name as collection_name, c.color as collection_color
                    FROM source_configs sc
                    LEFT JOIN collections c ON c.id = sc.collection_id
                    {where}
                    ORDER BY sc.source_type, sc.name""",
                    params,
                )
                return cur.fetchall()

    def create_source_config(
        self, source_type: str, name: str, config: dict,
        collection_id: str | None = None, enabled: bool = True,
    ) -> dict:
        self.ensure_source_configs_table()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO source_configs (source_type, name, config, collection_id, enabled)
                    VALUES (%s, %s, %s, %s, %s) RETURNING *""",
                    (source_type, name, json.dumps(config), collection_id, enabled),
                )
                row = cur.fetchone()
            conn.commit()
            return row

    def get_source_config(self, config_id: str) -> dict | None:
        self.ensure_source_configs_table()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT sc.*, c.name as collection_name, c.color as collection_color
                    FROM source_configs sc
                    LEFT JOIN collections c ON c.id = sc.collection_id
                    WHERE sc.id = %s""",
                    (config_id,),
                )
                return cur.fetchone()

    def update_source_config(self, config_id: str, **fields) -> dict | None:
        self.ensure_source_configs_table()
        allowed = {"source_type", "name", "config", "collection_id", "enabled"}
        updates = []
        params: list = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "config":
                value = json.dumps(value)
            updates.append(f"{key} = %s")
            params.append(value)
        if not updates:
            return self.get_source_config(config_id)
        updates.append("updated_at = NOW()")
        params.append(config_id)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE source_configs SET {', '.join(updates)} WHERE id = %s RETURNING *",
                    params,
                )
                row = cur.fetchone()
            conn.commit()
            return row

    def delete_source_config(self, config_id: str) -> bool:
        self.ensure_source_configs_table()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM source_configs WHERE id = %s", (config_id,))
                deleted = cur.rowcount > 0
            conn.commit()
            return deleted

    def get_source_configs_by_type(self, source_type: str) -> list[dict]:
        """Get enabled source configs for a specific type (used by pipeline)."""
        self.ensure_source_configs_table()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT sc.*, c.name as collection_name
                    FROM source_configs sc
                    LEFT JOIN collections c ON c.id = sc.collection_id
                    WHERE sc.source_type = %s AND sc.enabled = TRUE
                    ORDER BY sc.name""",
                    (source_type,),
                )
                return cur.fetchall()

    def update_source_last_run(self, config_id: str):
        """Update last_run_at timestamp after pipeline processes a source."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE source_configs SET last_run_at = NOW() WHERE id = %s",
                    (config_id,),
                )
            conn.commit()

    def import_sources_from_yaml(self, yaml_path: str) -> int:
        """Import sources from a YAML file into source_configs table. Returns count imported."""
        from pathlib import Path
        import yaml as _yaml

        path = Path(yaml_path)
        if not path.exists():
            return 0

        with open(path) as f:
            data = _yaml.safe_load(f) or {}

        count = 0

        # Web URLs
        for item in data.get("web", []) or []:
            if isinstance(item, dict):
                self.create_source_config(
                    source_type="web_url",
                    name=item.get("url", "")[:80],
                    config={"url": item["url"]},
                    enabled=item.get("enabled", True),
                )
                count += 1

        # YouTube
        yt = data.get("youtube", {}) or {}
        if yt.get("watch_later"):
            self.create_source_config(
                source_type="youtube_watch_later",
                name="Watch Later",
                config={},
            )
            count += 1
        for pl_id in yt.get("playlists", []) or []:
            self.create_source_config(
                source_type="youtube_playlist",
                name=f"Playlist {pl_id[:20]}",
                config={"playlist_id": pl_id},
            )
            count += 1
        for ch_id in yt.get("channels", []) or []:
            self.create_source_config(
                source_type="youtube_channel",
                name=f"Channel {ch_id[:20]}",
                config={"channel_id": ch_id, "max_videos": yt.get("channel_max_videos", 5)},
            )
            count += 1

        # Reddit
        rd = data.get("reddit", {}) or {}
        if rd.get("saved_posts"):
            self.create_source_config(
                source_type="reddit_saved",
                name="Reddit Saved Posts",
                config={"limit": rd.get("limit", 50)},
            )
            count += 1

        # Twitter
        tw = data.get("twitter", {}) or {}
        if tw.get("bookmarks"):
            self.create_source_config(
                source_type="twitter_bookmarks",
                name="Twitter Bookmarks",
                config={"limit": tw.get("limit", 50)},
            )
            count += 1

        # Folders
        folders = data.get("folders", {}) or {}
        if folders.get("enabled"):
            for src in folders.get("sources", []) or []:
                self.create_source_config(
                    source_type="folder",
                    name=src.get("path", "").split("/")[-1] or "Folder",
                    config={
                        "path": src["path"],
                        "extensions": src.get("extensions", [".pdf"]),
                        "max_file_size_mb": src.get("max_file_size_mb", 50),
                    },
                )
                count += 1

        return count

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
