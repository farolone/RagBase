from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rag.api_routes import documents, search, chat, collections, tags, ratings, graph, pipeline

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="RAG Wissensdatenbank", version="0.2.0")

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include API routers
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(collections.router)
app.include_router(tags.router)
app.include_router(ratings.router)
app.include_router(graph.router)
app.include_router(pipeline.router)


# --- Pydantic models for existing endpoints ---

class IngestRequest(BaseModel):
    source: str
    type: str = "auto"


class FolderIngestRequest(BaseModel):
    path: str
    extensions: list[str] | None = None
    exclude_patterns: list[str] | None = None
    max_file_size_mb: float = 100


class AskRequest(BaseModel):
    question: str
    platform: str | None = None
    limit: int = 10


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    citation_count: int


# File extensions that trigger the DocumentIngestor
_DOCUMENT_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".epub", ".txt", ".md", ".html",
    ".xlsx", ".csv", ".json", ".yaml", ".yml", ".log", ".adoc",
}


def _detect_source_type(source: str) -> str:
    """Auto-detect source type from URL or file path."""
    if "youtube.com" in source or "youtu.be" in source:
        return "youtube"
    if "reddit.com" in source:
        return "reddit"

    # Check if it looks like a local file path
    p = Path(source)
    if p.suffix.lower() in _DOCUMENT_EXTENSIONS:
        return "document"

    # Default to web
    return "web"


# --- Page routes ---

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("pages/stats.html", {"request": request, "page": "stats"})


@app.get("/search")
def search_page(request: Request):
    return templates.TemplateResponse("pages/search.html", {"request": request, "page": "search"})


@app.get("/chat")
def chat_page(request: Request):
    return templates.TemplateResponse("pages/chat.html", {"request": request, "page": "chat"})


@app.get("/documents")
def documents_page(request: Request):
    return templates.TemplateResponse("pages/documents.html", {"request": request, "page": "documents"})


@app.get("/documents/{doc_id}")
def document_detail_page(request: Request, doc_id: str):
    return templates.TemplateResponse("pages/document_detail.html", {"request": request, "page": "documents", "doc_id": doc_id})


@app.get("/collections")
def collections_page(request: Request):
    return templates.TemplateResponse("pages/collections.html", {"request": request, "page": "collections"})


@app.get("/collections/{collection_id}")
def collection_detail_page(request: Request, collection_id: str):
    return templates.TemplateResponse("pages/collection_detail.html", {"request": request, "page": "collections", "collection_id": collection_id})


@app.get("/graph")
def graph_page(request: Request):
    return templates.TemplateResponse("pages/graph.html", {"request": request, "page": "graph"})


@app.get("/pipeline")
def pipeline_page(request: Request):
    return templates.TemplateResponse("pages/pipeline.html", {"request": request, "page": "pipeline"})


@app.get("/settings")
def settings_page(request: Request):
    return templates.TemplateResponse("pages/settings.html", {"request": request, "page": "settings"})


# --- Existing API endpoints (kept for backward compatibility) ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    from rag.ingestion.pdf import PDFIngestor
    from rag.ingestion.youtube import YouTubeIngestor
    from rag.ingestion.web import WebIngestor
    from rag.ingestion.document import DocumentIngestor
    from rag.ingestion.reddit import RedditIngestor
    from rag.processing.embedding import Embedder
    from rag.processing.ner import EntityExtractor
    from rag.processing.graph_builder import GraphBuilder
    from rag.storage.qdrant import QdrantStore
    from rag.storage.postgres import PostgresStore

    source_type = req.type
    if source_type == "auto":
        source_type = _detect_source_type(req.source)

    ingestors = {
        "pdf": PDFIngestor,
        "youtube": YouTubeIngestor,
        "web": WebIngestor,
        "document": DocumentIngestor,
        "reddit": RedditIngestor,
    }

    if source_type not in ingestors:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {source_type}")

    ingestor = ingestors[source_type]()
    doc, chunks = ingestor.ingest(req.source)

    embedder = Embedder()
    qdrant = QdrantStore()
    qdrant.ensure_collection()
    postgres = PostgresStore()

    for chunk in chunks:
        emb = embedder.embed(chunk.content)
        qdrant.upsert(
            chunk=chunk,
            dense_vector=emb.dense,
            sparse_indices=emb.sparse_indices,
            sparse_values=emb.sparse_values,
        )

    postgres.save_document(doc)

    ner = EntityExtractor()
    graph_builder = GraphBuilder()
    all_entities = []
    for chunk in chunks:
        entities = ner.extract(chunk.content, document_id=doc.id, chunk_id=chunk.id)
        all_entities.extend(entities)
    graph_builder.process_document(doc, all_entities)
    graph_builder.close()

    # Update counts
    postgres.update_document_counts(doc.id, len(chunks), len(all_entities))

    return {
        "document_id": doc.id,
        "title": doc.title,
        "chunks": len(chunks),
        "entities": len(all_entities),
    }


@app.post("/api/ingest/folder")
def ingest_folder(req: FolderIngestRequest):
    """Scan a folder and ingest all supported files."""
    from rag.ingestion.folder_scanner import FolderScanner
    from rag.ingestion.document import DocumentIngestor
    from rag.processing.embedding import Embedder
    from rag.processing.ner import EntityExtractor
    from rag.processing.graph_builder import GraphBuilder
    from rag.storage.qdrant import QdrantStore
    from rag.storage.postgres import PostgresStore

    scanner = FolderScanner(
        extensions=req.extensions,
        exclude_patterns=req.exclude_patterns,
        max_file_size_mb=req.max_file_size_mb,
    )

    try:
        scan_result = scanner.scan(req.path)
    except (FileNotFoundError, NotADirectoryError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    postgres = PostgresStore()
    embedder = Embedder()
    qdrant = QdrantStore()
    qdrant.ensure_collection()
    ner = EntityExtractor()
    doc_ingestor = DocumentIngestor()

    ingested = 0
    skipped = 0
    errors = []

    for f in scan_result.files:
        resolved = str(f.path.resolve())

        # Dedup check
        if postgres.document_exists_by_url(resolved):
            skipped += 1
            continue

        try:
            doc, chunks = doc_ingestor.ingest(resolved)

            for chunk in chunks:
                emb = embedder.embed(chunk.content)
                qdrant.upsert(
                    chunk=chunk,
                    dense_vector=emb.dense,
                    sparse_indices=emb.sparse_indices,
                    sparse_values=emb.sparse_values,
                )

            postgres.save_document(doc)

            graph = GraphBuilder()
            all_entities = []
            for chunk in chunks:
                entities = ner.extract(chunk.content, document_id=doc.id, chunk_id=chunk.id)
                all_entities.extend(entities)
            graph.process_document(doc, all_entities)
            graph.close()

            postgres.update_document_counts(doc.id, len(chunks), len(all_entities))
            ingested += 1

        except Exception as e:
            errors.append({"file": f.relative_path, "error": str(e)})

    return {
        "ingested": ingested,
        "skipped": skipped,
        "errors": errors,
        "total_scanned": len(scan_result.files),
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    from rag.retrieval.hybrid import HybridRetriever
    from rag.generation.router import QueryRouter
    from rag.generation.citation import CitationGenerator

    retriever = HybridRetriever()
    results = retriever.retrieve(req.question, limit=req.limit, filter_platform=req.platform)

    if not results:
        return AskResponse(answer="No relevant documents found.", sources=[], citation_count=0)

    citation_gen = CitationGenerator()
    prompt, citation_map = citation_gen.build_prompt(req.question, results)

    router = QueryRouter()
    answer = router.generate(req.question, context=prompt, system=citation_gen.SYSTEM_PROMPT)

    parsed = citation_gen.parse_citations(answer, citation_map)
    return AskResponse(**parsed)


@app.get("/stats")
def get_stats():
    from rag.storage.qdrant import QdrantStore
    from rag.storage.postgres import PostgresStore
    from rag.storage.neo4j_store import Neo4jStore
    from collections import Counter

    qdrant = QdrantStore()
    postgres = PostgresStore()

    try:
        info = qdrant.get_collection_info()
        vector_count = info.points_count
    except Exception:
        vector_count = 0

    docs = postgres.search_documents(limit=10000)
    platforms = Counter(d.platform.value for d in docs)
    timeline = postgres.get_ingestion_timeline()

    try:
        neo4j = Neo4jStore()
        entity_count = neo4j.get_entity_count()
        neo4j.close()
    except Exception:
        entity_count = 0

    return {
        "vectors": vector_count,
        "documents": len(docs),
        "platforms": dict(platforms),
        "entities": entity_count,
        "timeline": timeline,
    }
