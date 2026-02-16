from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="RAG API", version="0.1.0")


class IngestRequest(BaseModel):
    source: str
    type: str = "auto"


class AskRequest(BaseModel):
    question: str
    platform: str | None = None
    limit: int = 10


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    citation_count: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    from rag.ingestion.pdf import PDFIngestor
    from rag.ingestion.youtube import YouTubeIngestor
    from rag.ingestion.web import WebIngestor
    from rag.processing.embedding import Embedder
    from rag.processing.ner import EntityExtractor
    from rag.processing.graph_builder import GraphBuilder
    from rag.storage.qdrant import QdrantStore
    from rag.storage.postgres import PostgresStore

    source_type = req.type
    if source_type == "auto":
        if req.source.endswith(".pdf"):
            source_type = "pdf"
        elif "youtube.com" in req.source or "youtu.be" in req.source:
            source_type = "youtube"
        elif "reddit.com" in req.source:
            source_type = "reddit"
        else:
            source_type = "web"

    ingestors = {
        "pdf": PDFIngestor,
        "youtube": YouTubeIngestor,
        "web": WebIngestor,
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
    graph = GraphBuilder()
    all_entities = []
    for chunk in chunks:
        entities = ner.extract(chunk.content, document_id=doc.id, chunk_id=chunk.id)
        all_entities.extend(entities)
    graph.process_document(doc, all_entities)
    graph.close()

    return {
        "document_id": doc.id,
        "title": doc.title,
        "chunks": len(chunks),
        "entities": len(all_entities),
    }


@app.get("/search")
def search_docs(
    q: str = Query(..., description="Search query"),
    platform: str | None = Query(None),
    author: str | None = Query(None),
    limit: int = Query(10, le=100),
):
    from rag.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()
    results = retriever.retrieve(q, limit=limit, filter_platform=platform, filter_author=author)

    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ],
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

    return {
        "vectors": vector_count,
        "documents": len(docs),
        "platforms": dict(platforms),
    }
