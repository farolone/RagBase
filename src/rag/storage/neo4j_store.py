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
                f"MATCH (e:Entity) {where} RETURN e.name AS name, e.entity_type AS entity_type, e.entity_id AS entity_id LIMIT $limit",
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

    def delete_document_cascade(self, doc_id: str) -> int:
        """Delete a document node and all its relationships from Neo4j.
        Also deletes orphan entities (entities only connected to this document).
        Returns count of deleted entities.
        """
        with self.driver.session() as session:
            # First, find and delete orphan entities
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[:MENTIONS]->(e:Entity)
                WHERE NOT EXISTS {
                    MATCH (other:Document)-[:MENTIONS]->(e)
                    WHERE other.doc_id <> $doc_id
                }
                WITH collect(e) as orphans, count(e) as cnt
                UNWIND orphans as orphan
                DETACH DELETE orphan
                RETURN cnt
                """,
                doc_id=doc_id,
            )
            record = result.single()
            deleted_entities = record["cnt"] if record else 0

            # Then delete the document node and remaining relationships
            session.run(
                "MATCH (d:Document {doc_id: $doc_id}) DETACH DELETE d",
                doc_id=doc_id,
            )
            return deleted_entities

    def get_document_graph(self, doc_id: str, limit: int = 50) -> dict:
        """Get the entity graph for a specific document.
        Returns nodes and edges suitable for vis-network.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[r:MENTIONS]->(e:Entity)
                RETURN d, e, r
                ORDER BY r.confidence DESC
                LIMIT $limit
                """,
                doc_id=doc_id,
                limit=limit,
            )
            nodes = []
            edges = []
            seen_nodes = set()
            doc_node_id = None

            for record in result:
                doc_node = record["d"]
                entity = record["e"]

                # Document node (once)
                d_id = f"d_{doc_node['doc_id']}"
                if d_id not in seen_nodes:
                    nodes.append({
                        "id": d_id,
                        "label": doc_node.get("title", "Dokument"),
                        "group": "Document",
                        "type": "document",
                        "doc_id": doc_node["doc_id"],
                    })
                    seen_nodes.add(d_id)
                    doc_node_id = d_id

                # Entity node
                e_id = f"e_{entity['entity_id']}"
                if e_id not in seen_nodes:
                    nodes.append({
                        "id": e_id,
                        "label": entity["name"],
                        "group": entity.get("entity_type", "UNKNOWN"),
                        "type": "entity",
                    })
                    seen_nodes.add(e_id)

                edges.append({
                    "from": doc_node_id or d_id,
                    "to": e_id,
                    "label": "MENTIONS",
                })

            return {"nodes": nodes, "edges": edges}

    def get_entity_neighborhood(self, entity_name: str, depth: int = 1) -> dict:
        """Get an entity's neighborhood for graph visualization.
        Returns nodes and edges suitable for vis-network.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (center:Entity {name: $name})
                OPTIONAL MATCH (d:Document)-[r:MENTIONS]->(center)
                OPTIONAL MATCH (d)-[r2:MENTIONS]->(neighbor:Entity)
                WHERE neighbor <> center
                WITH center, collect(DISTINCT d) as docs,
                     collect(DISTINCT {entity: neighbor, doc: d}) as neighbors
                RETURN center, docs, neighbors
                """,
                name=entity_name,
            )
            record = result.single()
            if not record:
                return {"nodes": [], "edges": []}

            nodes = []
            edges = []
            seen_nodes = set()

            # Center entity
            center = record["center"]
            center_id = f"e_{center['name']}"
            nodes.append({
                "id": center_id,
                "label": center["name"],
                "group": center.get("entity_type", "UNKNOWN"),
                "type": "entity",
            })
            seen_nodes.add(center_id)

            # Document nodes
            for doc in record["docs"]:
                if doc is None:
                    continue
                doc_id = f"d_{doc['doc_id']}"
                if doc_id not in seen_nodes:
                    nodes.append({
                        "id": doc_id,
                        "label": doc.get("title", "Unknown"),
                        "group": "Document",
                        "type": "document",
                        "doc_id": doc["doc_id"],
                    })
                    seen_nodes.add(doc_id)
                edges.append({"from": doc_id, "to": center_id, "label": "MENTIONS"})

            # Neighbor entities (via shared documents)
            for item in record["neighbors"]:
                entity = item["entity"]
                doc = item["doc"]
                if entity is None or doc is None:
                    continue
                eid = f"e_{entity['name']}"
                if eid not in seen_nodes:
                    nodes.append({
                        "id": eid,
                        "label": entity["name"],
                        "group": entity.get("entity_type", "UNKNOWN"),
                        "type": "entity",
                    })
                    seen_nodes.add(eid)
                doc_id = f"d_{doc['doc_id']}"
                edge_key = f"{doc_id}->{eid}"
                edges.append({"from": doc_id, "to": eid, "label": "MENTIONS"})

            return {"nodes": nodes, "edges": edges}

    def get_all_entities_with_counts(self, entity_type: str | None = None, limit: int = 100) -> list[dict]:
        """Get entities with document counts for the entity list view."""
        conditions = []
        params: dict = {"limit": limit}
        if entity_type:
            conditions.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type
        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (e:Entity)
                {where}
                OPTIONAL MATCH (d:Document)-[:MENTIONS]->(e)
                RETURN e.name AS name, e.entity_type AS entity_type,
                       e.entity_id AS entity_id, count(d) AS doc_count
                ORDER BY doc_count DESC
                LIMIT $limit
                """,
                **params,
            )
            return [dict(record) for record in result]

    def get_entity_types(self) -> list[str]:
        """Get distinct entity types."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN DISTINCT e.entity_type AS entity_type ORDER BY entity_type"
            )
            return [record["entity_type"] for record in result if record["entity_type"]]

    def get_entity_count(self) -> int:
        """Get total entity count."""
        with self.driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0

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
