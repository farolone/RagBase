"""Check existing chunk data for timestamps."""
from rag.storage.qdrant import QdrantStore
from rag.storage.postgres import PostgresStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

pg = PostgresStore()
docs, _ = pg.list_documents(limit=20)
yt_doc = [d for d in docs if d.get("platform") == "youtube"][0]
print("Doc:", yt_doc["id"], yt_doc["title"][:60])

qs = QdrantStore()
results = qs.client.scroll(
    collection_name="documents",
    scroll_filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=str(yt_doc["id"])))]),
    limit=5,
    with_payload=True,
)
for pt in results[0]:
    meta = pt.payload.get("metadata", {})
    ci = meta.get("chunk_index", "?")
    st = meta.get("start_time", "N/A")
    txt = pt.payload.get("content", "")[:100]
    print(f"  Chunk {ci} | start={st}s | {txt}")
