# RagBase

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Neo4j](https://img.shields.io/badge/Neo4j-4581C3?logo=neo4j&logoColor=white)](https://neo4j.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white)](https://postgresql.org)
[![HTMX](https://img.shields.io/badge/HTMX-3366CC?logo=htmx&logoColor=white)](https://htmx.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![Alpine.js](https://img.shields.io/badge/Alpine.js-8BC0D0?logo=alpinedotjs&logoColor=white)](https://alpinejs.dev)
[![Prefect](https://img.shields.io/badge/Prefect-024DFD?logo=prefect&logoColor=white)](https://prefect.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-hosted, modular RAG (Retrieval-Augmented Generation) system for building a personal knowledge base from multiple sources. Designed for 50-100K multilingual documents with hybrid search, a knowledge graph, and a full web UI.

## Features

- **Multi-source ingestion** - PDFs (Docling), web articles (Trafilatura), YouTube transcripts, Reddit threads, Twitter/X bookmarks, local folders
- **Hybrid search** - Dense + sparse vectors via BGE-M3 embeddings (1024d) and Qdrant
- **Knowledge graph** - Entity extraction (GLiNER) and topic modeling (BERTopic) stored in Neo4j
- **Hierarchical chunking** - Leaf (512), parent (1024), and grandparent (2048) token chunks with media-specific strategies
- **LLM-powered answers** - Streaming chat with citations via any OpenAI-compatible API (LM Studio, mlx-server, vLLM, etc.)
- **Automated pipeline** - Prefect-based daily ingestion from configured sources with deduplication
- **Web frontend** - Dashboard, search, chat, document management, collections, graph visualization, pipeline monitoring
- **CLI** - Ingest, search, ask, and view stats from the terminal

## Architecture

```
                        +---------------------------+
                        |      Web UI / API         |
                        | FastAPI + HTMX + Alpine.js|
                        +---------------------------+
                                    |
              +---------------------+---------------------+
              |                     |                     |
        +----------+         +-----------+        +------------+
        |  Qdrant  |         |   Neo4j   |        | PostgreSQL |
        | (vectors)|         |  (graph)  |        | (metadata) |
        +----------+         +-----------+        +------------+
              |                     |                     |
              +---------------------+---------------------+
                                    |
                        +---------------------------+
                        |   Processing Pipeline     |
                        | BGE-M3 . GLiNER . BERTopic|
                        +---------------------------+
                                    |
                        +---------------------------+
                        |    Ingestion Layer        |
                        | PDF Web YT Reddit Twitter |
                        +---------------------------+
                                    |
                        +---------------------------+
                        |  LLM (OpenAI-compatible)  |
                        | LM Studio / mlx / vLLM    |
                        +---------------------------+
```

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- An OpenAI-compatible LLM server (e.g. [LM Studio](https://lmstudio.ai/), [mlx-server](https://github.com/ml-explore/mlx-examples))

### 1. Clone and install

```bash
git clone https://github.com/farolone/RagBase.git
cd RagBase
pip install uv
uv sync --extra all
```

### 2. Start infrastructure

```bash
docker compose up -d
```

This starts Qdrant (vector DB), Neo4j (knowledge graph), and PostgreSQL (metadata).

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` with your LLM endpoint and credentials:

```ini
LLM_BASE_URL=http://your-llm-host:port/v1
LLM_MODEL_RAG=your-model-name

EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu          # or "cuda" if GPU available
```

### 4. Ingest content

```bash
# Single document
python -m rag.cli ingest https://en.wikipedia.org/wiki/Retrieval-augmented_generation
python -m rag.cli ingest paper.pdf
python -m rag.cli ingest "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Search
python -m rag.cli search "hybrid retrieval strategies"

# Ask a question
python -m rag.cli ask "What are the main approaches to RAG?"
```

### 5. Start the web UI

```bash
uvicorn rag.api:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

## Project Structure

```
src/rag/
├── api.py                  # FastAPI application
├── api_routes/             # Modular API endpoints
│   ├── chat.py             #   Chat sessions with SSE streaming
│   ├── collections.py      #   Document collections
│   ├── documents.py        #   Document CRUD + cascade delete
│   ├── graph.py            #   Knowledge graph queries
│   ├── pipeline.py         #   Pipeline status and control
│   ├── ratings.py          #   Source quality ratings
│   ├── search.py           #   Hybrid search
│   └── tags.py             #   Document tagging
├── cli.py                  # Typer CLI
├── config.py               # Pydantic settings
├── models.py               # Core data models
├── frontend/               # Static assets (CSS, JS)
├── templates/              # Jinja2 templates (HTMX)
├── generation/
│   ├── llm.py              # OpenAI-compatible LLM client
│   ├── citation.py         # Citation extraction
│   └── router.py           # Query routing
├── ingestion/
│   ├── pdf.py              # Docling + PyMuPDF4LLM
│   ├── web.py              # Trafilatura
│   ├── youtube.py          # YouTube transcripts
│   ├── reddit.py           # PRAW
│   ├── twitter.py          # Twikit
│   ├── folder_scanner.py   # Local folder monitoring
│   └── document.py         # Document management
├── pipeline/
│   ├── flows.py            # Prefect flows
│   ├── sources.py          # Source config (sources.yaml)
│   ├── dedup.py            # URL-based deduplication
│   ├── tasks.py            # Prefect tasks
│   └── deploy.py           # Deployment and scheduling
├── processing/
│   ├── chunking.py         # Hierarchical chunking
│   ├── embedding.py        # BGE-M3 dense + sparse
│   ├── ner.py              # GLiNER entity extraction
│   ├── topics.py           # BERTopic modeling
│   └── graph_builder.py    # Entity to Neo4j graph
├── retrieval/
│   ├── hybrid.py           # Dense + sparse fusion
│   └── reranker.py         # Optional LLM reranking
└── storage/
    ├── qdrant.py            # Vector store
    ├── neo4j_store.py       # Knowledge graph
    └── postgres.py          # Metadata + relations
```

## Automated Pipeline

Configure ingestion sources in `sources.yaml`:

```yaml
schedule:
  cron: "0 6 * * *"

web:
  - url: https://example.com/article
    enabled: true

youtube:
  watch_later: true
  playlists:
    - PLxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

folders:
  enabled: true
  sources:
    - path: /data/papers
      extensions: [".pdf", ".epub"]
```

Deploy with Prefect:

```bash
# Start Prefect server
prefect server start

# Deploy the pipeline
python -m rag.pipeline.deploy
```

## API

| Endpoint | Description |
|---|---|
| `GET /api/documents` | List documents (paginated, filterable) |
| `DELETE /api/documents/{id}` | Cascade delete (Qdrant + Neo4j + PostgreSQL) |
| `GET /api/search?q=...` | Hybrid search with filters |
| `GET /api/ask/stream` | SSE streaming answers with citations |
| `GET /api/chat/sessions` | Chat session management |
| `GET /api/collections` | Document collections |
| `GET /api/tags` | Tags (manual + BERTopic) |
| `GET /api/graph/entities` | Knowledge graph entities |
| `GET /api/graph/neighborhood/{name}` | Entity neighborhood subgraph |
| `GET /api/pipeline/status` | Pipeline run status |
| `GET /health` | Health check |

## Web UI

| Page | Path | Description |
|---|---|---|
| Dashboard | `/` | Overview with stats, charts, recent documents |
| Search | `/search` | Hybrid search with filters and highlights |
| Chat | `/chat` | Multi-session chat with streaming and citations |
| Documents | `/documents` | Document list with detail view and management |
| Collections | `/collections` | Organize documents into collections |
| Graph | `/graph` | Interactive knowledge graph visualization |
| Pipeline | `/pipeline` | Ingestion pipeline status and control |
| Settings | `/settings` | System configuration |

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| API framework | FastAPI |
| Frontend | Jinja2 + HTMX + Alpine.js + TailwindCSS |
| Vector DB | Qdrant (hybrid dense + sparse) |
| Knowledge graph | Neo4j 5 Community |
| Metadata DB | PostgreSQL 16 |
| Embeddings | BGE-M3 (1024d dense + sparse) |
| NER | GLiNER |
| Topics | BERTopic |
| LLM | Any OpenAI-compatible API |
| Pipeline | Prefect 3 |
| Visualization | vis-network (graph), Chart.js (stats) |

## Development

```bash
# Install dev dependencies
uv sync --extra all

# Run tests
pytest

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## License

MIT
