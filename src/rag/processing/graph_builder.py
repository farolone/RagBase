from rag.models import Document, Entity, Topic
from rag.storage.neo4j_store import Neo4jStore


class GraphBuilder:
    """Populates Neo4j knowledge graph from NER + BERTopic results."""

    def __init__(self, store: Neo4jStore | None = None):
        self.store = store or Neo4jStore()

    def process_document(
        self,
        doc: Document,
        entities: list[Entity],
        topics: list[Topic] | None = None,
    ) -> None:
        self.store.create_document_node(doc)

        for entity in entities:
            self.store.create_entity_node(entity)
            self.store.create_mentions_relationship(
                doc.id, entity.id, confidence=entity.confidence
            )

        if topics:
            for topic in topics:
                self.store.create_topic_relationship(doc.id, topic.name)

    def process_batch(
        self,
        documents: list[tuple[Document, list[Entity], list[Topic] | None]],
    ) -> int:
        count = 0
        for doc, entities, topics in documents:
            self.process_document(doc, entities, topics)
            count += 1
        return count

    def close(self):
        self.store.close()
