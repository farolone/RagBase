import json
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter(prefix="/api/documents", tags=["documents"])

DOCUMENTS_BASE = Path("/root/documents")


class QualityUpdate(BaseModel):
    quality_score: int


class FlagUpdate(BaseModel):
    flagged: bool
    reason: str | None = None


@router.get("")
def list_documents(
    platform: str | None = Query(None),
    author: str | None = Query(None),
    collection_id: str | None = Query(None),
    flagged: bool | None = Query(None),
    search: str | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    rows, total = pg.list_documents(
        platform=platform, author=author, collection_id=collection_id,
        flagged=flagged, search=search, offset=offset, limit=limit,
    )
    # Convert rows to serializable dicts
    docs = []
    for row in rows:
        doc = dict(row)
        for k, v in doc.items():
            if hasattr(v, 'isoformat'):
                doc[k] = v.isoformat()
            elif isinstance(v, dict):
                pass  # JSON already
            elif hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, type(None))):
                doc[k] = str(v)
        docs.append(doc)
    return {"documents": docs, "total": total, "offset": offset, "limit": limit}


@router.get("/{doc_id}")
def get_document(doc_id: str):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    doc = pg.get_document_detail(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    # Serialize
    result = dict(doc)
    for k, v in result.items():
        if hasattr(v, 'isoformat'):
            result[k] = v.isoformat()
        elif hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, type(None), list, dict)):
            result[k] = str(v)
    # Serialize nested tags/collections
    if "tags" in result:
        result["tags"] = [_serialize_row(t) for t in result["tags"]]
    if "collections" in result:
        result["collections"] = [_serialize_row(c) for c in result["collections"]]
    return result


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    """Cascade delete: removes from Qdrant, Neo4j, and PostgreSQL."""
    from rag.storage.postgres import PostgresStore
    from rag.storage.qdrant import QdrantStore
    from rag.storage.neo4j_store import Neo4jStore

    pg = PostgresStore()
    qdrant = QdrantStore()
    neo4j = Neo4jStore()

    # Check exists
    doc = pg.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from all stores
    vectors_deleted = qdrant.delete_by_document_id(doc_id)
    entities_deleted = neo4j.delete_document_cascade(doc_id)
    neo4j.close()
    pg.delete_document(doc_id)

    return {
        "deleted": True,
        "document_id": doc_id,
        "vectors_deleted": vectors_deleted,
        "entities_deleted": entities_deleted,
    }


@router.post("/{doc_id}/re-ingest")
def re_ingest_document(doc_id: str):
    """Re-ingest a document from its original source."""
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    doc = pg.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.source_url:
        raise HTTPException(status_code=400, detail="No source URL for this document")

    # Delete old data first
    from rag.storage.qdrant import QdrantStore
    from rag.storage.neo4j_store import Neo4jStore
    qdrant = QdrantStore()
    neo4j = Neo4jStore()
    qdrant.delete_by_document_id(doc_id)
    neo4j.delete_document_cascade(doc_id)
    neo4j.close()
    pg.delete_document(doc_id)

    # Re-ingest
    from rag.ingestion.web import WebIngestor
    from rag.ingestion.youtube import YouTubeIngestor
    from rag.ingestion.pdf import PDFIngestor
    from rag.processing.embedding import Embedder
    from rag.processing.ner import EntityExtractor
    from rag.processing.graph_builder import GraphBuilder

    source = doc.source_url
    if "youtube.com" in source or "youtu.be" in source:
        ingestor = YouTubeIngestor()
    elif source.endswith(".pdf"):
        ingestor = PDFIngestor()
    else:
        ingestor = WebIngestor()

    new_doc, chunks = ingestor.ingest(source)
    embedder = Embedder()
    qdrant.ensure_collection()
    for chunk in chunks:
        emb = embedder.embed(chunk.content)
        qdrant.upsert(chunk=chunk, dense_vector=emb.dense, sparse_indices=emb.sparse_indices, sparse_values=emb.sparse_values)

    pg.save_document(new_doc)
    pg.update_document_counts(new_doc.id, len(chunks), 0)

    ner = EntityExtractor()
    graph = GraphBuilder()
    all_entities = []
    for chunk in chunks:
        entities = ner.extract(chunk.content, document_id=new_doc.id, chunk_id=chunk.id)
        all_entities.extend(entities)
    graph.process_document(new_doc, all_entities)
    graph.close()
    pg.update_document_counts(new_doc.id, len(chunks), len(all_entities))

    return {"document_id": new_doc.id, "title": new_doc.title, "chunks": len(chunks), "entities": len(all_entities)}


@router.get("/{doc_id}/chunks")
def get_document_chunks(doc_id: str):
    from rag.storage.qdrant import QdrantStore
    qdrant = QdrantStore()
    chunks = qdrant.get_chunks_for_document(doc_id)
    return {"chunks": chunks, "count": len(chunks)}


@router.get("/{doc_id}/entities")
def get_document_entities(doc_id: str):
    from rag.storage.neo4j_store import Neo4jStore
    neo4j = Neo4jStore()
    entities = neo4j.get_entities_for_document(doc_id)
    neo4j.close()
    return {"entities": entities, "count": len(entities)}


@router.put("/{doc_id}/quality")
def update_quality(doc_id: str, body: QualityUpdate):
    if not 1 <= body.quality_score <= 5:
        raise HTTPException(status_code=400, detail="Quality score must be 1-5")
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    pg.update_document_quality(doc_id, body.quality_score)
    return {"ok": True}


@router.post("/{doc_id}/flag")
def flag_document(doc_id: str, body: FlagUpdate):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    pg.flag_document(doc_id, body.flagged, body.reason)
    return {"ok": True}


@router.put("/{doc_id}/tags")
def update_document_tags(doc_id: str, tag_ids: list[str]):
    from rag.storage.postgres import PostgresStore
    pg = PostgresStore()
    pg.set_document_tags(doc_id, tag_ids)
    return {"ok": True}


@router.post("/upload")
def upload_document(
    file: UploadFile = File(...),
    collection_id: str | None = Form(None),
):
    """Upload a file, save to collection folder, ingest, and assign to collection."""
    from rag.storage.postgres import PostgresStore

    pg = PostgresStore()

    # Determine target folder
    folder_name = "Unsorted"
    if collection_id:
        coll = pg.get_collection(collection_id)
        if coll:
            folder_name = coll["name"]

    target_dir = DOCUMENTS_BASE / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    target_path = target_dir / file.filename
    with open(target_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest
    try:
        from rag.pipeline.tasks import _run_full_ingest

        ext = Path(file.filename).suffix.lower()
        if ext in (".pdf",):
            source_type = "pdf"
        else:
            source_type = "web"  # fallback for txt/md/etc

        result = _run_full_ingest(str(target_path), source_type)

        # Assign to collection
        if collection_id and result:
            pg.add_document_to_collection(result["doc_id"], collection_id)

        return {
            "uploaded": True,
            "filename": file.filename,
            "path": str(target_path),
            "document_id": result["doc_id"] if result else None,
            "chunks": result["chunks"] if result else 0,
            "entities": result["entities"] if result else 0,
        }
    except Exception as e:
        return {
            "uploaded": True,
            "filename": file.filename,
            "path": str(target_path),
            "error": str(e),
        }


def _serialize_row(row):
    result = dict(row)
    for k, v in result.items():
        if hasattr(v, 'isoformat'):
            result[k] = v.isoformat()
        elif hasattr(v, '__str__') and not isinstance(v, (str, int, float, bool, type(None), list, dict)):
            result[k] = str(v)
    return result
