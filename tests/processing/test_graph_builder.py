import pytest
from rag.processing.graph_builder import GraphBuilder
from rag.storage.neo4j_store import Neo4jStore
from rag.models import Document, Entity, Topic, Platform


@pytest.fixture
def builder():
    store = Neo4jStore()
    b = GraphBuilder(store=store)
    yield b
    store.cleanup_test_data()
    store.close()


def test_process_document_with_entities(builder):
    doc = Document(title="Test Graph Doc", platform=Platform.WEB)
    entities = [
        Entity(name="Berlin", entity_type="LOCATION", source_document_id=doc.id, confidence=0.95),
        Entity(name="Angela Merkel", entity_type="PERSON", source_document_id=doc.id, confidence=0.9),
    ]

    builder.process_document(doc, entities)

    found = builder.store.get_entities_for_document(doc.id)
    names = [e["name"] for e in found]
    assert "Berlin" in names
    assert "Angela Merkel" in names


def test_process_document_with_topics(builder):
    doc = Document(title="Topic Test", platform=Platform.PDF)
    entities = [
        Entity(name="Python", entity_type="TOPIC", source_document_id=doc.id),
    ]
    topics = [
        Topic(name="programming", bertopic_id=0),
        Topic(name="machine-learning", bertopic_id=1),
    ]

    builder.process_document(doc, entities, topics)

    found = builder.store.get_entities_for_document(doc.id)
    assert len(found) >= 1


def test_process_batch(builder):
    docs = []
    for i in range(3):
        doc = Document(title=f"Batch Doc {i}", platform=Platform.WEB)
        entities = [
            Entity(name=f"Entity{i}", entity_type="ORGANIZATION", source_document_id=doc.id),
        ]
        docs.append((doc, entities, None))

    count = builder.process_batch(docs)
    assert count == 3
