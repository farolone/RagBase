"""Manually assign the last 10 unassigned documents."""
from rag.storage.postgres import PostgresStore

pg = PostgresStore()

# Get collection IDs
colls = {c["name"]: str(c["id"]) for c in pg.list_collections()}

# title substring -> collection name
MANUAL = {
    "Animated Miniature Worlds In Blender": "OpenClaw",
    "MASSIVE STRASSEN-SCHLACHTEN AN SILVESTER": "Politik",
    "unterschätzt Ihr alle die Polizei": "Politik",
    "Starlink kill SiriusXM": "OpenClaw",
    "Raus aus Afghanistan": "Auswandern",
    "Verbrannte Pizza Du VS KI": "Unterhaltung",
    "20 December 2025": "OpenClaw",
    "Reinhard Flötotto": "Unterhaltung",
    "LiDAR Scanning in Blender": "OpenClaw",
    "FOLLOW THROUGH": "Unterhaltung",
}

docs, total = pg.list_documents(limit=500)
assigned = 0

for doc in docs:
    title = doc.get("title", "") or ""
    for substr, coll_name in MANUAL.items():
        if substr.lower() in title.lower():
            pg.add_document_to_collection(str(doc["id"]), colls[coll_name])
            print(f"  {coll_name:<20} <- {title[:70]}")
            assigned += 1
            break

print(f"\nAssigned {assigned}/10 remaining documents.")
