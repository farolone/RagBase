from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/entities")
def list_entities(
    entity_type: str | None = Query(None),
    name: str | None = Query(None),
    limit: int = Query(100, le=500),
):
    from rag.storage.neo4j_store import Neo4jStore
    neo4j = Neo4jStore()
    if name:
        entities = neo4j.search_entities(entity_type=entity_type, name=name, limit=limit)
    else:
        entities = neo4j.get_all_entities_with_counts(entity_type=entity_type, limit=limit)
    entity_types = neo4j.get_entity_types()
    neo4j.close()
    return {"entities": entities, "entity_types": entity_types, "count": len(entities)}


@router.get("/document/{doc_id}")
def get_document_graph(doc_id: str, limit: int = Query(50, le=200)):
    """Get the entity graph for a specific document."""
    from rag.storage.neo4j_store import Neo4jStore
    neo4j = Neo4jStore()
    data = neo4j.get_document_graph(doc_id, limit=limit)
    neo4j.close()
    return data


@router.get("/neighborhood/{entity_name}")
def get_neighborhood(entity_name: str):
    from rag.storage.neo4j_store import Neo4jStore
    neo4j = Neo4jStore()
    data = neo4j.get_entity_neighborhood(entity_name)
    neo4j.close()
    return data


@router.get("/stats")
def graph_stats():
    from rag.storage.neo4j_store import Neo4jStore
    neo4j = Neo4jStore()
    entity_count = neo4j.get_entity_count()
    entity_types = neo4j.get_entity_types()
    neo4j.close()
    return {"entity_count": entity_count, "entity_types": entity_types}
