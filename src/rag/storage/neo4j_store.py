from neo4j import GraphDatabase

from rag.config import settings
from rag.models import Document, Entity


class Neo4jStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self._test_doc_ids: list[str] = []
        self._test_entity_ids: list[str] = []
        self._ensure_constraints()

    def _ensure_constraints(self):
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
            )

    def close(self):
        self.driver.close()

    def create_document_node(self, doc: Document):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.title = $title,
                    d.source_url = $source_url,
                    d.platform = $platform,
                    d.author = $author
                """,
                doc_id=doc.id,
                title=doc.title,
                source_url=doc.source_url,
                platform=doc.platform.value,
                author=doc.author,
            )
        self._test_doc_ids.append(doc.id)

    def get_document_node(self, doc_id: str) -> dict | None:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (d:Document {doc_id: $doc_id}) RETURN d",
                doc_id=doc_id,
            )
            record = result.single()
            if record is None:
                return None
            return dict(record["d"])

    def create_entity_node(self, entity: Entity):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.name = $name,
                    e.entity_type = $entity_type,
                    e.confidence = $confidence
                """,
                entity_id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
            )
        self._test_entity_ids.append(entity.id)

    def create_mentions_relationship(
        self, doc_id: str, entity_id: str, confidence: float = 1.0
    ):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                MATCH (e:Entity {entity_id: $entity_id})
                MERGE (d)-[r:MENTIONS]->(e)
                SET r.confidence = $confidence
                """,
                doc_id=doc_id,
                entity_id=entity_id,
                confidence=confidence,
            )

    def create_topic_relationship(self, doc_id: str, topic_name: str):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (t:Topic {name: $topic_name})
                MERGE (d)-[:HAS_TOPIC]->(t)
                """,
                doc_id=doc_id,
                topic_name=topic_name,
            )

    def get_entities_for_document(self, doc_id: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[:MENTIONS]->(e:Entity)
                RETURN e.name AS name, e.entity_type AS entity_type, e.confidence AS confidence
                """,
                doc_id=doc_id,
            )
            return [dict(record) for record in result]

    def search_entities(
        self, entity_type: str | None = None, name: str | None = None, limit: int = 50
    ) -> list[dict]:
        conditions = []
        params: dict = {"limit": limit}

        if entity_type:
            conditions.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type
        if name:
            conditions.append("e.name CONTAINS $name")
            params["name"] = name

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        with self.driver.session() as session:
            result = session.run(
                f"MATCH (e:Entity) {where} RETURN e.name AS name, e.entity_type AS entity_type LIMIT $limit",
                **params,
            )
            return [dict(record) for record in result]

    def get_documents_for_entity(self, entity_name: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)-[:MENTIONS]->(e:Entity {name: $name})
                RETURN d.doc_id AS doc_id, d.title AS title, d.platform AS platform
                """,
                name=entity_name,
            )
            return [dict(record) for record in result]

    def cleanup_test_data(self):
        with self.driver.session() as session:
            for doc_id in self._test_doc_ids:
                session.run(
                    "MATCH (d:Document {doc_id: $doc_id}) DETACH DELETE d",
                    doc_id=doc_id,
                )
            for entity_id in self._test_entity_ids:
                session.run(
                    "MATCH (e:Entity {entity_id: $entity_id}) DETACH DELETE e",
                    entity_id=entity_id,
                )
        self._test_doc_ids.clear()
        self._test_entity_ids.clear()
