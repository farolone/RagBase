"""Check existing chunk data for timestamps."""
from rag.storage.qdrant import QdrantStore
from rag.storage.postgres import PostgresStore

pg = PostgresStore()
qs = QdrantStore()

docs, _ = pg.list_documents(limit=500)
yt_docs = [d for d in docs if d.get("platform") == "youtube"]
print(f"{len(yt_docs)} YouTube docs total\n")

# Check 3 docs
for yt_doc in yt_docs[:3]:
    chunks = qs.get_chunks_for_document(str(yt_doc["id"]), limit=5)
    print(f"Doc: {yt_doc['title'][:60]}")
    for c in chunks[:3]:
        meta = c["metadata"]
        st = meta.get("start_time", "N/A")
        src = meta.get("source_url", "N/A")
        vid = meta.get("video_id", "N/A")
        print(f"  idx={c['chunk_index']} | start_time={st} | video_id={vid}")
        print(f"    source_url={src[:80]}")
        print(f"    text: {c['content'][:80]}")
    print()
