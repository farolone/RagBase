# RAG Wissensmanagement-System -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Self-hosted RAG-System fuer 50-100K Dokumente (YouTube, Twitter, Reddit, Web, PDF) mit minimalen Informationsverlusten, automatischem Tagging und Knowledge Graph. Dual-Model Setup mit intelligentem Routing.

**Architecture:** Modularer Python-Monorepo mit klarer Schichtenarchitektur: Ingestion -> Processing -> Storage -> Retrieval -> Generation. Jede Schicht ist unabhaengig testbar. Infrastruktur (Qdrant, Neo4j, PostgreSQL, Prefect) + Processing (Embedding, NER, Topics) laeuft in einem LXC Container auf Proxmox (192.168.178.4, x86, keine GPU). LLM-Inference (Dual-Model: Qwen 2.5-72B + MiniMax M2.5) + Reranking laeuft auf dem Mac Studio Ultra (512GB RAM), erreichbar via Ollama REST API im LAN.

**Tech Stack:** Python 3.12+, uv, LlamaIndex, Qdrant, Neo4j, PostgreSQL, BGE-M3, Qwen 2.5-72B (Kern-RAG), MiniMax M2.5 (Agentic Router), Qwen3-Reranker-8B, GLiNER, BERTopic, Prefect, Docker Compose

---

## Projekt-Struktur (Ziel)

```
rag/
  pyproject.toml
  uv.lock
  docker-compose.yml          # Qdrant + Neo4j + PostgreSQL + Prefect
  .env.example
  src/
    rag/
      __init__.py
      config.py                # Zentrale Konfiguration (Pydantic Settings)
      models.py                # Datenmodelle (Document, Chunk, Entity, Topic)
      storage/
        __init__.py
        qdrant.py              # Qdrant Client (Hybrid Search)
        neo4j.py               # Neo4j Client (Knowledge Graph)
        postgres.py            # PostgreSQL Client (Metadaten)
      ingestion/
        __init__.py
        base.py                # BaseIngestor ABC
        pdf.py                 # Docling + PyMuPDF4LLM
        youtube.py             # youtube-transcript-api + Whisper
        twitter.py             # Twikit
        reddit.py              # PRAW
        web.py                 # Trafilatura + Crawl4AI
      processing/
        __init__.py
        chunking.py            # Hierarchisches + Semantisches Chunking
        embedding.py           # BGE-M3 via sentence-transformers
        ner.py                 # GLiNER + spaCy
        topics.py              # BERTopic
      retrieval/
        __init__.py
        hybrid.py              # Hybrid Search (Dense + Sparse)
        reranker.py            # Qwen3-Reranker
        graph_retrieval.py     # Neo4j + LightRAG
      generation/
        __init__.py
        llm.py                 # Ollama Client (Qwen + MiniMax)
        router.py              # Dual-Model Query Router (einfach->Qwen, komplex->MiniMax->Qwen)
        citation.py            # Citation-aware Answer Generation
      pipeline/
        __init__.py
        flows.py               # Prefect Flows
      cli.py                   # CLI Interface (Click/Typer)
      api.py                   # FastAPI REST API
  tests/
    conftest.py                # Shared Fixtures
    test_models.py
    storage/
    ingestion/
    processing/
    retrieval/
    generation/
```

---

## Phase 1: Fundament (Projekt-Setup + Infrastruktur + Datenmodelle)

### Task 1: Projekt-Repository initialisieren

**Files:**
- Create: `pyproject.toml`
- Create: `src/rag/__init__.py`
- Create: `.gitignore`
- Create: `.python-version`

**Step 1: Git-Repository initialisieren**

Run:
```bash
cd /Users/matthias/Rag
git init
```

**Step 2: Python-Projekt mit uv erstellen**

Run:
```bash
uv init --lib --name rag --python 3.12
```

**Step 3: pyproject.toml anpassen**

Die von `uv init` erstellte Datei editieren -- Kern-Dependencies hinzufuegen:

```toml
[project]
name = "rag"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
storage = [
    "qdrant-client>=1.12",
    "neo4j>=5.0",
    "psycopg[binary]>=3.0",
    "sqlalchemy>=2.0",
]
ingestion = [
    "docling>=2.0",
    "pymupdf4llm>=0.0.10",
    "youtube-transcript-api>=0.6",
    "trafilatura>=1.0",
    "praw>=7.0",
]
processing = [
    "sentence-transformers>=3.0",
    "FlagEmbedding>=1.0",
    "spacy>=3.7",
    "gliner>=1.0",
    "bertopic>=0.16",
]
retrieval = [
    "llama-index>=0.11",
]
generation = [
    "ollama>=0.4",
]
pipeline = [
    "prefect>=3.0",
]
api = [
    "fastapi>=0.115",
    "uvicorn>=0.32",
]
cli = [
    "typer>=0.12",
    "rich>=13.0",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.8",
]
all = [
    "rag[storage,ingestion,processing,retrieval,generation,pipeline,api,cli,dev]",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Step 4: .gitignore erstellen**

```
__pycache__/
*.pyc
.venv/
.env
*.egg-info/
dist/
.pytest_cache/
.ruff_cache/
data/
*.gguf
```

**Step 5: .env.example erstellen**

```
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag
POSTGRES_USER=rag
POSTGRES_PASSWORD=changeme

# Ollama (Mac Studio Ultra im LAN)
OLLAMA_HOST=192.168.178.y
OLLAMA_PORT=11434
OLLAMA_MODEL_RAG=qwen2.5:72b-instruct-q8_0
OLLAMA_MODEL_AGENT=minimax-m2.5:q4_K_M
OLLAMA_MODEL_RERANKER=qwen3-reranker:8b

# Reddit
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=rag-bot/0.1

# Embedding
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=mps
```

**Step 6: Kern-Dependencies installieren**

Run:
```bash
uv sync
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: initialize project with uv, pyproject.toml, and .env.example"
```

---

### Task 2: Docker Compose fuer Infrastruktur

**Files:**
- Create: `docker-compose.yml`
- Create: `docker/qdrant/config.yaml`
- Create: `docker/neo4j/neo4j.conf` (optional)
- Create: `docker/postgres/init.sql`

**Step 1: docker-compose.yml erstellen**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/changeme
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_server_memory_heap_initial__size=512m
      - NEO4J_server_memory_heap_max__size=2g
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      - POSTGRES_DB=rag
      - POSTGRES_USER=rag
      - POSTGRES_PASSWORD=changeme
    restart: unless-stopped

volumes:
  qdrant_data:
  neo4j_data:
  postgres_data:
```

**Step 2: PostgreSQL Init-Script erstellen**

Datei `docker/postgres/init.sql`:

```sql
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source_url TEXT,
    platform TEXT NOT NULL,
    author TEXT,
    language TEXT,
    created_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_documents_platform ON documents(platform);
CREATE INDEX idx_documents_author ON documents(author);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);
```

**Step 3: Docker Compose starten und verifizieren**

Run:
```bash
docker compose up -d
```

Verifizieren:
```bash
docker compose ps
curl -s http://localhost:6333/healthz  # Qdrant
curl -s http://localhost:7474          # Neo4j
pg_isready -h localhost -p 5432 -U rag # PostgreSQL
```
Expected: Alle 3 Services "healthy"/"running"

**Step 4: Commit**

```bash
git add docker-compose.yml docker/
git commit -m "infra: add docker-compose with Qdrant, Neo4j, PostgreSQL"
```

---

### Task 3: Zentrale Konfiguration und Datenmodelle

**Files:**
- Create: `src/rag/config.py`
- Create: `src/rag/models.py`
- Create: `tests/test_models.py`

**Step 1: Failing Tests schreiben**

Datei `tests/test_models.py`:

```python
from rag.models import Document, Chunk, Entity, Platform


def test_document_creation():
    doc = Document(
        title="Test Article",
        source_url="https://example.com/article",
        platform=Platform.WEB,
        author="John Doe",
        language="en",
    )
    assert doc.title == "Test Article"
    assert doc.platform == Platform.WEB
    assert doc.id is not None


def test_chunk_creation():
    chunk = Chunk(
        document_id="doc-123",
        content="This is a test chunk.",
        chunk_index=0,
        token_count=6,
        metadata={"section": "intro"},
    )
    assert chunk.content == "This is a test chunk."
    assert chunk.parent_chunk_id is None


def test_chunk_hierarchy():
    parent = Chunk(
        document_id="doc-123",
        content="Parent content with more context.",
        chunk_index=0,
        token_count=10,
    )
    child = Chunk(
        document_id="doc-123",
        content="Child content.",
        chunk_index=0,
        token_count=4,
        parent_chunk_id=parent.id,
    )
    assert child.parent_chunk_id == parent.id


def test_entity_creation():
    entity = Entity(
        name="Berlin",
        entity_type="LOCATION",
        source_document_id="doc-123",
        source_chunk_id="chunk-456",
        confidence=0.95,
    )
    assert entity.entity_type == "LOCATION"
    assert entity.confidence == 0.95


def test_platform_enum():
    assert Platform.YOUTUBE.value == "youtube"
    assert Platform.TWITTER.value == "twitter"
    assert Platform.REDDIT.value == "reddit"
    assert Platform.WEB.value == "web"
    assert Platform.PDF.value == "pdf"
```

**Step 2: Tests laufen lassen -- muessen fehlschlagen**

Run:
```bash
uv run pytest tests/test_models.py -v
```
Expected: FAIL -- `ModuleNotFoundError` oder `ImportError`

**Step 3: config.py implementieren**

Datei `src/rag/config.py`:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag"
    postgres_user: str = "rag"
    postgres_password: str = "changeme"

    # Ollama (Mac Studio Ultra im LAN)
    ollama_host: str = "192.168.178.y"
    ollama_port: int = 11434
    ollama_model_rag: str = "qwen2.5:72b-instruct-q8_0"
    ollama_model_agent: str = "minimax-m2.5:q4_K_M"
    ollama_model_reranker: str = "qwen3-reranker:8b"

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "mps"

    # Chunking
    chunk_size_leaf: int = 512
    chunk_size_parent: int = 1024
    chunk_size_grandparent: int = 2048
    chunk_overlap: int = 50

    model_config = {"env_file": ".env", "env_prefix": ""}


settings = Settings()
```

**Step 4: models.py implementieren**

Datei `src/rag/models.py`:

```python
from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class Platform(str, Enum):
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    REDDIT = "reddit"
    WEB = "web"
    PDF = "pdf"


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_url: str | None = None
    platform: Platform
    author: str | None = None
    language: str | None = None
    created_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    parent_chunk_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    entity_type: str
    source_document_id: str
    source_chunk_id: str | None = None
    confidence: float = 1.0
    metadata: dict = Field(default_factory=dict)


class Topic(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    hierarchy_path: str | None = None
    bertopic_id: int | None = None
    metadata: dict = Field(default_factory=dict)
```

**Step 5: Tests erneut laufen lassen -- muessen bestehen**

Run:
```bash
uv run pytest tests/test_models.py -v
```
Expected: All PASS

**Step 6: Commit**

```bash
git add src/rag/config.py src/rag/models.py tests/test_models.py
git commit -m "feat: add config (pydantic-settings) and core data models"
```

---

## Phase 2: Storage Layer (Qdrant + PostgreSQL + Neo4j Clients)

### Task 4: Qdrant Client mit Hybrid Search

**Files:**
- Create: `src/rag/storage/__init__.py`
- Create: `src/rag/storage/qdrant.py`
- Create: `tests/storage/__init__.py`
- Create: `tests/storage/test_qdrant.py`

**Step 1: Failing Tests schreiben**

Datei `tests/storage/test_qdrant.py`:

```python
import pytest
from rag.storage.qdrant import QdrantStore
from rag.models import Chunk, Platform


@pytest.fixture
def store():
    """Creates a test collection and cleans up after."""
    s = QdrantStore(collection_name="test_documents")
    s.ensure_collection(dense_dim=1024)
    yield s
    s.delete_collection()


def test_ensure_collection(store):
    info = store.get_collection_info()
    assert info is not None


def test_upsert_and_search(store):
    chunk = Chunk(
        document_id="doc-1",
        content="Berlin is the capital of Germany.",
        chunk_index=0,
        token_count=7,
        metadata={"platform": "web", "author": "test"},
    )
    # Use a fake embedding (1024 dims)
    dense = [0.1] * 1024
    sparse_indices = [0, 5, 10]
    sparse_values = [0.8, 0.5, 0.3]

    store.upsert(
        chunk=chunk,
        dense_vector=dense,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
    )

    results = store.search(
        dense_vector=dense,
        limit=5,
    )
    assert len(results) >= 1
    assert results[0].chunk_id == chunk.id


def test_search_with_filter(store):
    chunk = Chunk(
        document_id="doc-1",
        content="Test content",
        chunk_index=0,
        token_count=3,
        metadata={"platform": "youtube", "author": "alice"},
    )
    dense = [0.2] * 1024
    store.upsert(chunk=chunk, dense_vector=dense)

    results = store.search(
        dense_vector=dense,
        filter_platform="youtube",
        limit=5,
    )
    assert len(results) >= 1
    assert results[0].metadata["platform"] == "youtube"
```

**Step 2: Tests muessen fehlschlagen**

Run:
```bash
uv run pytest tests/storage/test_qdrant.py -v
```
Expected: FAIL (ImportError)

**Step 3: Qdrant Client implementieren**

Datei `src/rag/storage/qdrant.py`:

```python
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from rag.config import settings


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict


class QdrantStore:
    def __init__(self, collection_name: str | None = None):
        self.client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        self.collection_name = collection_name or settings.qdrant_collection

    def ensure_collection(self, dense_dim: int = 1024):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_dim, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            # Create payload indexes for filtering
            for field, schema in [
                ("platform", PayloadSchemaType.KEYWORD),
                ("author", PayloadSchemaType.KEYWORD),
                ("document_id", PayloadSchemaType.KEYWORD),
                ("language", PayloadSchemaType.KEYWORD),
            ]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )

    def get_collection_info(self):
        return self.client.get_collection(self.collection_name)

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)

    def upsert(
        self,
        chunk,
        dense_vector: list[float],
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
    ):
        vectors = {"dense": dense_vector}
        if sparse_indices and sparse_values:
            vectors["sparse"] = SparseVector(
                indices=sparse_indices, values=sparse_values
            )

        payload = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            **chunk.metadata,
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=hash(chunk.id) % (2**63),
                    vector=vectors,
                    payload=payload,
                )
            ],
        )

    def search(
        self,
        dense_vector: list[float],
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        filter_platform: str | None = None,
        filter_author: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        conditions = []
        if filter_platform:
            conditions.append(
                FieldCondition(
                    key="platform", match=MatchValue(value=filter_platform)
                )
            )
        if filter_author:
            conditions.append(
                FieldCondition(
                    key="author", match=MatchValue(value=filter_author)
                )
            )

        query_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            SearchResult(
                chunk_id=hit.payload["chunk_id"],
                document_id=hit.payload["document_id"],
                content=hit.payload["content"],
                score=hit.score,
                metadata={
                    k: v
                    for k, v in hit.payload.items()
                    if k not in ("chunk_id", "document_id", "content", "chunk_index")
                },
            )
            for hit in results.points
        ]
```

**Step 4: Tests erneut laufen lassen**

Run:
```bash
uv run pytest tests/storage/test_qdrant.py -v
```
Expected: All PASS (benoetigt laufenden Qdrant Docker Container)

**Step 5: Commit**

```bash
git add src/rag/storage/ tests/storage/
git commit -m "feat: add Qdrant client with hybrid search and metadata filtering"
```

---

### Task 5: PostgreSQL Client fuer Dokument-Metadaten

**Files:**
- Create: `src/rag/storage/postgres.py`
- Create: `tests/storage/test_postgres.py`

**Step 1: Failing Tests schreiben**

```python
# tests/storage/test_postgres.py
import pytest
from rag.storage.postgres import PostgresStore
from rag.models import Document, Platform


@pytest.fixture
def store():
    s = PostgresStore()
    yield s
    s.cleanup_test_data()


def test_save_and_get_document(store):
    doc = Document(
        title="Test Doc",
        source_url="https://example.com",
        platform=Platform.WEB,
        author="Alice",
        language="en",
    )
    store.save_document(doc)
    retrieved = store.get_document(doc.id)
    assert retrieved is not None
    assert retrieved.title == "Test Doc"


def test_search_by_platform(store):
    doc = Document(
        title="YouTube Video",
        source_url="https://youtube.com/watch?v=123",
        platform=Platform.YOUTUBE,
    )
    store.save_document(doc)
    results = store.search_documents(platform=Platform.YOUTUBE)
    assert any(d.id == doc.id for d in results)


def test_search_by_author(store):
    doc = Document(
        title="By Bob",
        platform=Platform.REDDIT,
        author="bob",
    )
    store.save_document(doc)
    results = store.search_documents(author="bob")
    assert any(d.id == doc.id for d in results)
```

**Step 2: Implementierung** -- SQLAlchemy-basierter Client der Document-Objekte in PostgreSQL speichert und abfragt. CRUD + Filter nach platform, author, date range.

**Step 3: Tests bestehen lassen, Commit**

```bash
git add src/rag/storage/postgres.py tests/storage/test_postgres.py
git commit -m "feat: add PostgreSQL client for document metadata storage"
```

---

### Task 6: Neo4j Client fuer Knowledge Graph

**Files:**
- Create: `src/rag/storage/neo4j.py`
- Create: `tests/storage/test_neo4j.py`

**Step 1: Failing Tests schreiben**

Tests fuer: Entity erstellen, Beziehung erstellen (Document->Entity), Entity-Suche, Multi-Hop-Query ("Alle Dokumente von Person X ueber Thema Y").

**Step 2: Implementierung** -- Neo4j Python Driver Client mit Cypher Queries. Schema:
- `(:Document)` nodes mit properties
- `(:Person)`, `(:Organization)`, `(:Location)`, `(:Topic)` nodes
- `[:MENTIONS]`, `[:HAS_TOPIC]`, `[:FROM_PLATFORM]` relationships
- Schema-Constraints fuer Unique-IDs

**Step 3: Tests bestehen lassen, Commit**

```bash
git add src/rag/storage/neo4j.py tests/storage/test_neo4j.py
git commit -m "feat: add Neo4j client for knowledge graph with entity relationships"
```

---

## Phase 3: Embedding + Chunking

### Task 7: BGE-M3 Embedding Client

**Files:**
- Create: `src/rag/processing/__init__.py`
- Create: `src/rag/processing/embedding.py`
- Create: `tests/processing/__init__.py`
- Create: `tests/processing/test_embedding.py`

**Step 1: Failing Tests schreiben**

```python
# tests/processing/test_embedding.py
import pytest
from rag.processing.embedding import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_embed_single(embedder):
    result = embedder.embed("Hello World")
    assert result.dense is not None
    assert len(result.dense) == 1024
    assert result.sparse_indices is not None
    assert result.sparse_values is not None


def test_embed_batch(embedder):
    texts = ["Hello", "World", "Berlin ist die Hauptstadt"]
    results = embedder.embed_batch(texts)
    assert len(results) == 3
    for r in results:
        assert len(r.dense) == 1024


def test_embed_german_english_similarity(embedder):
    de = embedder.embed("Hund")
    en = embedder.embed("dog")
    from numpy import dot
    from numpy.linalg import norm
    similarity = dot(de.dense, en.dense) / (norm(de.dense) * norm(en.dense))
    assert similarity > 0.5  # Cross-lingual similarity should be meaningful
```

**Step 2: Implementierung** -- `Embedder` Klasse die BGE-M3 via `FlagEmbedding` oder `sentence-transformers` laedt. Liefert Dense + Sparse Vektoren. Device-Konfiguration fuer MPS.

```python
# src/rag/processing/embedding.py
from dataclasses import dataclass

from FlagEmbedding import BGEM3FlagModel

from rag.config import settings


@dataclass
class EmbeddingResult:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class Embedder:
    def __init__(self):
        self.model = BGEM3FlagModel(
            settings.embedding_model, use_fp16=True
        )

    def embed(self, text: str) -> EmbeddingResult:
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
        )
        results = []
        for i in range(len(texts)):
            dense = output["dense_vecs"][i].tolist()
            sparse = output["lexical_weights"][i]
            indices = list(sparse.keys())
            values = list(sparse.values())
            results.append(
                EmbeddingResult(
                    dense=dense,
                    sparse_indices=indices,
                    sparse_values=values,
                )
            )
        return results
```

**Step 3: Tests bestehen lassen, Commit**

```bash
git add src/rag/processing/ tests/processing/
git commit -m "feat: add BGE-M3 embedding with dense + sparse vector output"
```

---

### Task 8: Hierarchisches Chunking

**Files:**
- Create: `src/rag/processing/chunking.py`
- Create: `tests/processing/test_chunking.py`

**Step 1: Failing Tests schreiben**

Tests fuer:
- `chunk_text()` mit Leaf/Parent/Grandparent Hierarchie
- Chunk-Overlap korrekt
- Metadaten werden an Chunks weitergegeben
- Verschiedene Medientypen: `chunk_youtube()` (bei Kapitelgrenzen), `chunk_reddit()` (Post + Comments), `chunk_tweet()` (Thread)

**Step 2: Implementierung** -- Chunking-Modul mit verschiedenen Strategien:
- `HierarchicalChunker`: Leaf (512) -> Parent (1024) -> Grandparent (2048), verbunden via `parent_chunk_id`
- `SemanticChunker`: Embedding-Similarity Breakpoints (fuer Web-Artikel)
- `MediaChunker`: Medientyp-spezifisch (YouTube: Kapitel, Reddit: Post+Comments, Twitter: Threads)

**Step 3: Tests bestehen lassen, Commit**

```bash
git add src/rag/processing/chunking.py tests/processing/test_chunking.py
git commit -m "feat: add hierarchical and media-specific chunking strategies"
```

---

## Phase 4: Ingestion Pipelines

### Task 9: Base Ingestor + PDF Pipeline

**Files:**
- Create: `src/rag/ingestion/__init__.py`
- Create: `src/rag/ingestion/base.py`
- Create: `src/rag/ingestion/pdf.py`
- Create: `tests/ingestion/__init__.py`
- Create: `tests/ingestion/test_pdf.py`
- Create: `tests/fixtures/sample.pdf` (kleines Test-PDF)

**Step 1: Failing Tests**

Tests fuer:
- `PDFIngestor.ingest(path)` gibt `Document` + `list[Chunk]` zurueck
- Metadaten werden extrahiert (Titel, Seitenzahlen)
- Tabellen werden erkannt (Docling)
- Einfache PDFs werden schnell geparsed (PyMuPDF4LLM Fallback)

**Step 2: Implementierung**

```python
# src/rag/ingestion/base.py
from abc import ABC, abstractmethod
from rag.models import Document, Chunk


class BaseIngestor(ABC):
    @abstractmethod
    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        """Ingest a source and return a Document with its Chunks."""
        ...
```

```python
# src/rag/ingestion/pdf.py
from pathlib import Path
from docling.document_converter import DocumentConverter
from rag.ingestion.base import BaseIngestor
from rag.models import Document, Chunk, Platform
from rag.processing.chunking import HierarchicalChunker


class PDFIngestor(BaseIngestor):
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        path = Path(source)
        result = self.converter.convert(str(path))
        text = result.document.export_to_markdown()

        doc = Document(
            title=path.stem,
            source_url=str(path),
            platform=Platform.PDF,
            metadata={"pages": len(result.document.pages) if hasattr(result.document, 'pages') else None},
        )

        chunks = self.chunker.chunk(
            text=text,
            document_id=doc.id,
            metadata={"platform": "pdf", "source": str(path)},
        )
        return doc, chunks
```

**Step 3: Tests bestehen lassen, Commit**

```bash
git add src/rag/ingestion/ tests/ingestion/ tests/fixtures/
git commit -m "feat: add PDF ingestion with Docling + PyMuPDF4LLM fallback"
```

---

### Task 10: YouTube Pipeline

**Files:**
- Create: `src/rag/ingestion/youtube.py`
- Create: `tests/ingestion/test_youtube.py`

**Implementierung:** `YouTubeIngestor` der:
1. Video-ID aus URL extrahiert
2. `youtube_transcript_api` fuer Transcript mit Timestamps nutzt
3. Bei Kapitel-Grenzen chunked (falls vorhanden), sonst 60-Sek-Fenster
4. Metadata: Channel, Video-ID, Timestamps, Kapitel, Source-URL mit Timestamp-Parameter

**Commit:**
```bash
git commit -m "feat: add YouTube transcript ingestion with chapter-based chunking"
```

---

### Task 11: Web Pipeline

**Files:**
- Create: `src/rag/ingestion/web.py`
- Create: `tests/ingestion/test_web.py`

**Implementierung:** `WebIngestor` der:
1. Trafilatura fuer Artikel-Extraktion nutzt (F1: 0.958)
2. Metadata: Titel, Autor, Datum, Domain, URL
3. Semantisches Chunking fuer Artikel-Text
4. Fallback: Crawl4AI fuer JS-heavy Seiten

**Commit:**
```bash
git commit -m "feat: add web article ingestion with Trafilatura + Crawl4AI fallback"
```

---

### Task 12: Reddit Pipeline

**Files:**
- Create: `src/rag/ingestion/reddit.py`
- Create: `tests/ingestion/test_reddit.py`

**Implementierung:** `RedditIngestor` der:
1. PRAW mit OAuth nutzt (100 req/min)
2. Post-Title + Body als ein Chunk
3. Top-Level Comments als einzelne Chunks, verlinkt zum Parent
4. Metadata: Subreddit, Author, Score, Comment-Tree, URL

**Commit:**
```bash
git commit -m "feat: add Reddit ingestion via PRAW with hierarchical comment chunks"
```

---

### Task 13: Twitter/X Pipeline

**Files:**
- Create: `src/rag/ingestion/twitter.py`
- Create: `tests/ingestion/test_twitter.py`

**Implementierung:** `TwitterIngestor` der:
1. Twikit mit Account-Pool nutzt
2. Threads erkennt und als Parent-Dokument zusammenfasst
3. Einzelne Tweets als Chunks
4. Metadata: Author, Tweet-ID, Thread-ID, Likes, Date
5. Retry-Logik + Rate-Limit Handling

**Hinweis:** Dies ist die fragile Komponente. Gutes Error-Handling und Monitoring essentiell.

**Commit:**
```bash
git commit -m "feat: add Twitter/X ingestion via Twikit with thread support"
```

---

## Phase 5: NER + Topic Modeling + Knowledge Graph

### Task 14: GLiNER NER Pipeline

**Files:**
- Create: `src/rag/processing/ner.py`
- Create: `tests/processing/test_ner.py`

**Step 1: Failing Tests**

```python
# tests/processing/test_ner.py
from rag.processing.ner import EntityExtractor


def test_extract_person():
    extractor = EntityExtractor()
    entities = extractor.extract("Angela Merkel visited Berlin yesterday.")
    names = [e.name for e in entities]
    assert "Angela Merkel" in names


def test_extract_location():
    extractor = EntityExtractor()
    entities = extractor.extract("Berlin is the capital of Germany.")
    types = {e.name: e.entity_type for e in entities}
    assert types.get("Berlin") == "LOCATION" or types.get("Germany") == "LOCATION"


def test_extract_german():
    extractor = EntityExtractor()
    entities = extractor.extract("Die Bundeskanzlerin besuchte das Brandenburger Tor in Berlin.")
    names = [e.name for e in entities]
    assert "Berlin" in names or "Brandenburger Tor" in names
```

**Step 2: Implementierung** -- `EntityExtractor` mit GLiNER (zero-shot), integriert in spaCy-Pipeline via `gliner-spacy`. Entity-Typen: PERSON, ORGANIZATION, LOCATION, TOPIC (konfigurierbar).

**Commit:**
```bash
git commit -m "feat: add GLiNER-based multilingual NER extraction"
```

---

### Task 15: BERTopic Topic Modeling

**Files:**
- Create: `src/rag/processing/topics.py`
- Create: `tests/processing/test_topics.py`

**Implementierung:** `TopicModeler` der:
1. BERTopic mit `paraphrase-multilingual-MiniLM-L12-v2` Embeddings nutzt
2. Hierarchische Topics via `.hierarchical_topics()` erzeugt
3. Inkrementelle Updates via `.merge_models()` unterstuetzt
4. Topic-IDs und Labels zurueckgibt

**Commit:**
```bash
git commit -m "feat: add BERTopic hierarchical topic modeling"
```

---

### Task 16: Knowledge Graph Population

**Files:**
- Create: `src/rag/processing/graph_builder.py`
- Create: `tests/processing/test_graph_builder.py`

**Implementierung:** `GraphBuilder` der:
1. Entities aus NER nimmt und als Nodes in Neo4j erstellt
2. Topics aus BERTopic als Topic-Nodes erstellt
3. Beziehungen erstellt: Document->MENTIONS->Entity, Document->HAS_TOPIC->Topic
4. Bidirektionale Provenance-Links mit Confidence-Scores und Chunk-Referenzen

**Commit:**
```bash
git commit -m "feat: add knowledge graph population from NER + BERTopic results"
```

---

## Phase 6: Retrieval + Generation

### Task 17: Hybrid Retrieval Pipeline

**Files:**
- Create: `src/rag/retrieval/__init__.py`
- Create: `src/rag/retrieval/hybrid.py`
- Create: `tests/retrieval/__init__.py`
- Create: `tests/retrieval/test_hybrid.py`

**Implementierung:** `HybridRetriever` der:
1. Query embedded mit BGE-M3 (Dense + Sparse)
2. Qdrant Hybrid Search (Dense + Sparse mit RRF Fusion)
3. Metadata-Filter (Platform, Author, Date Range)
4. Top-K Kandidaten zurueckgibt

**Commit:**
```bash
git commit -m "feat: add hybrid retrieval with dense + sparse search via Qdrant"
```

---

### Task 18: Reranker Integration

**Files:**
- Create: `src/rag/retrieval/reranker.py`
- Create: `tests/retrieval/test_reranker.py`

**Implementierung:** `Reranker` der:
1. Qwen3-Reranker-8B via Ollama oder sentence-transformers laedt
2. Top 50 Retrieval-Ergebnisse nimmt
3. Cross-Encoder Scoring durchfuehrt
4. Top 5-10 reranked Ergebnisse zurueckgibt

**Commit:**
```bash
git commit -m "feat: add Qwen3-Reranker for retrieval result reranking"
```

---

### Task 19: Dual-Model LLM Client + Query Router

**Files:**
- Create: `src/rag/generation/__init__.py`
- Create: `src/rag/generation/llm.py`
- Create: `src/rag/generation/router.py`
- Create: `tests/generation/__init__.py`
- Create: `tests/generation/test_llm.py`
- Create: `tests/generation/test_router.py`

**Implementierung:**
1. `OllamaClient` -- Generischer Wrapper um Ollama REST API (Mac Studio Ultra im LAN)
   - `generate(model, prompt, ...)` -- Unterstuetzt beide Modelle
   - `embed(model, text)` -- Fuer Bulk-Embedding Fallback
   - Konfigurierbar: Host, Port, Timeout, Retry
2. `QueryRouter` -- Entscheidet welches Modell die Query bearbeitet:
   - Regelbasierte Klassifikation (Keywords, Query-Laenge, explizite Flags)
   - Einfache Fragen -> Qwen 2.5-72B direkt
   - Multi-Step / Tool-Use -> MiniMax M2.5 orchestriert, Qwen synthetisiert
   - Code-Fragen -> MiniMax M2.5
   - >128K Kontext -> MiniMax M2.5
3. `CitationGenerator` -- Baut Prompt mit Retrieved Chunks, instruiert Qwen fuer inline Citations [1], [2] etc., parst Antwort und mappt Citations auf Source-URLs

**Commit:**
```bash
git commit -m "feat: add dual-model LLM client with query router and citation generation"
```

---

### Task 19b: Agentic Multi-Step Retrieval (MiniMax M2.5)

**Files:**
- Create: `src/rag/generation/agent.py`
- Create: `tests/generation/test_agent.py`

**Implementierung:**
1. `AgenticRetriever` -- Nutzt MiniMax M2.5's Function Calling (76.8% BFCL):
   - Definiert Tools: `search_qdrant`, `query_neo4j`, `search_by_author`, `search_by_topic`
   - M2.5 entscheidet welche Tools aufzurufen und in welcher Reihenfolge
   - Sammelt Ergebnisse aus mehreren Tool-Calls
   - Uebergibt gesammelte Ergebnisse an Qwen fuer finale Synthese
2. Nur fuer komplexe Queries aktiviert (via QueryRouter)

**Commit:**
```bash
git commit -m "feat: add agentic multi-step retrieval via MiniMax M2.5 function calling"
```

---

## Phase 7: End-to-End Pipeline + Orchestrierung

### Task 20: Ingestion Pipeline (Prefect Flow)

**Files:**
- Create: `src/rag/pipeline/__init__.py`
- Create: `src/rag/pipeline/flows.py`
- Create: `tests/pipeline/test_flows.py`

**Implementierung:** Prefect Flows die:
1. Jede Quelle als eigenen Flow kapseln (YouTube, Reddit, Twitter, Web, PDF)
2. Unified Processing: Chunking -> Embedding -> NER -> Topics -> Graph -> Upsert
3. Error Handling: Ein fehlgeschlagener Flow stoppt nicht die anderen
4. Scheduling: Taeglicher Cron via `flow.serve()`

```python
# Grob-Struktur
from prefect import flow, task

@task
def ingest_document(source: str, platform: str):
    ...

@task
def process_chunks(doc, chunks):
    ...  # embed, NER, topics

@task
def store_results(doc, chunks, entities, topics):
    ...  # Qdrant, Neo4j, PostgreSQL

@flow
def daily_ingestion():
    youtube_flow()
    reddit_flow()
    web_flow()
    # twitter_flow()  # optional, fragil
```

**Commit:**
```bash
git commit -m "feat: add Prefect orchestration for daily ingestion pipeline"
```

---

### Task 21: CLI Interface

**Files:**
- Create: `src/rag/cli.py`

**Implementierung:** Typer-basiertes CLI:
- `rag ingest <source> --type pdf|youtube|web|reddit|twitter` -- Einzelnes Dokument ingesten
- `rag search <query> [--platform X] [--author Y]` -- Hybrid Search
- `rag ask <question>` -- Q&A mit Citations
- `rag stats` -- Statistiken (Dokument-Counts, Entity-Counts)
- `rag pipeline run` -- Manueller Pipeline-Trigger

**Commit:**
```bash
git commit -m "feat: add Typer CLI for ingest, search, ask, and stats commands"
```

---

### Task 22: FastAPI REST API

**Files:**
- Create: `src/rag/api.py`

**Implementierung:** FastAPI Endpoints:
- `POST /ingest` -- Dokument ingesten
- `GET /search?q=...&platform=...&author=...` -- Hybrid Search
- `POST /ask` -- Q&A mit Citations
- `GET /documents/{id}` -- Dokument-Details mit Entities und Topics
- `GET /entities?type=PERSON&name=...` -- Entity-Suche
- `GET /topics` -- Topic-Hierarchie
- `GET /stats` -- System-Statistiken

**Commit:**
```bash
git commit -m "feat: add FastAPI REST API for search, ask, and exploration"
```

---

## Phase 8: Integration + Evaluation

### Task 23: End-to-End Integration Test

**Files:**
- Create: `tests/test_e2e.py`

**Test:** Kompletter Durchlauf:
1. PDF ingesten -> Chunks + Entities + Topics in Qdrant, Neo4j, PostgreSQL
2. Hybrid Search Query -> Relevante Chunks zurueck
3. Q&A mit Citations -> Antwort mit korrekten Source-Links
4. Neo4j Query -> Entities und Beziehungen korrekt

**Commit:**
```bash
git commit -m "test: add end-to-end integration test for full RAG pipeline"
```

---

### Task 24: Evaluation Set erstellen

**Files:**
- Create: `tests/eval/eval_queries.json`
- Create: `tests/eval/evaluate.py`

**Implementierung:**
1. 20-30 Query-Document-Paare (DE + EN gemischt) manuell erstellen
2. nDCG@10 und Faithfulness messen
3. Vergleich: mit/ohne Reranking, mit/ohne hierarchisches Chunking

**Commit:**
```bash
git commit -m "test: add evaluation framework with manual query-document pairs"
```

---

## Zusammenfassung: Alle Tasks

| Phase | Task | Beschreibung | Geschaetzte Zeit |
|---|---|---|---|
| 0 | 0 | LXC Container erstellen + Docker installieren | 30 min |
| 0 | 0b | Ollama auf Mac Studio: Qwen + MiniMax + Reranker | 30 min |
| 0 | 0c | Netzwerk LXC <-> Mac Studio testen | 15 min |
| 1 | 1 | Projekt-Repository initialisieren | 15 min |
| 1 | 2 | Docker Compose Infrastruktur (im LXC) | 20 min |
| 1 | 3 | Config + Datenmodelle | 20 min |
| 2 | 4 | Qdrant Client (Hybrid Search) | 30 min |
| 2 | 5 | PostgreSQL Client (Metadaten) | 25 min |
| 2 | 6 | Neo4j Client (Knowledge Graph) | 30 min |
| 3 | 7 | BGE-M3 Embedding (CPU im LXC) | 25 min |
| 3 | 8 | Hierarchisches Chunking | 30 min |
| 4 | 9 | PDF Ingestion (Docling) | 30 min |
| 4 | 10 | YouTube Ingestion | 25 min |
| 4 | 11 | Web Ingestion (Trafilatura) | 25 min |
| 4 | 12 | Reddit Ingestion (PRAW) | 25 min |
| 4 | 13 | Twitter/X Ingestion (Twikit) | 30 min |
| 5 | 14 | GLiNER NER Pipeline (CPU im LXC) | 25 min |
| 5 | 15 | BERTopic Topic Modeling (CPU im LXC) | 25 min |
| 5 | 16 | Knowledge Graph Population | 25 min |
| 6 | 17 | Hybrid Retrieval Pipeline | 25 min |
| 6 | 18 | Reranker Integration (via Mac Studio API) | 20 min |
| 6 | 19 | Dual-Model LLM Client + Query Router | 35 min |
| 6 | 19b | Agentic Multi-Step Retrieval (MiniMax M2.5) | 30 min |
| 7 | 20 | Prefect Pipeline Orchestrierung | 30 min |
| 7 | 21 | CLI Interface | 25 min |
| 7 | 22 | FastAPI REST API | 25 min |
| 8 | 23 | End-to-End Integration Test | 30 min |
| 8 | 24 | Evaluation Framework | 25 min |

**Gesamt: ~27 Tasks in 9 Phasen (inkl. Phase 0: Infrastruktur)**
