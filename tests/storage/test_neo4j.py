import pytest
from rag.storage.neo4j_store import Neo4jStore
from rag.models import Entity, Document, Platform


@pytest.fixture
def store():
    s = Neo4jStore()
    yield s
    s.cleanup_test_data()
    s.close()


def test_create_document_node(store):
    doc = Document(
        title="Test Article",
        source_url="https://example.com",
        platform=Platform.WEB,
        author="Alice",
    )
    store.create_document_node(doc)
    result = store.get_document_node(doc.id)
    assert result is not None
    assert result["title"] == "Test Article"


def test_create_entity_and_relationship(store):
    doc = Document(
        title="Berlin Article",
        platform=Platform.WEB,
    )
    entity = Entity(
        name="Berlin",
        entity_type="LOCATION",
        source_document_id=doc.id,
        confidence=0.95,
    )
    store.create_document_node(doc)
    store.create_entity_node(entity)
    store.create_mentions_relationship(doc.id, entity.id, confidence=0.95)

    entities = store.get_entities_for_document(doc.id)
    assert len(entities) >= 1
    assert any(e["name"] == "Berlin" for e in entities)


def test_search_entities_by_type(store):
    doc = Document(title="Test", platform=Platform.PDF)
    entity = Entity(
        name="Munich",
        entity_type="LOCATION",
        source_document_id=doc.id,
    )
    store.create_document_node(doc)
    store.create_entity_node(entity)
    store.create_mentions_relationship(doc.id, entity.id)

    results = store.search_entities(entity_type="LOCATION")
    assert any(e["name"] == "Munich" for e in results)
