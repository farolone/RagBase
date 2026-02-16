from gliner import GLiNER

from rag.models import Entity


class EntityExtractor:
    """Zero-shot multilingual NER using GLiNER."""

    DEFAULT_LABELS = ["PERSON", "ORGANIZATION", "LOCATION", "TOPIC", "EVENT"]

    def __init__(self, model_name: str = "urchade/gliner_multi-v2.1"):
        self.model = GLiNER.from_pretrained(model_name)

    def extract(
        self,
        text: str,
        labels: list[str] | None = None,
        threshold: float = 0.4,
        document_id: str | None = None,
        chunk_id: str | None = None,
    ) -> list[Entity]:
        labels = labels or self.DEFAULT_LABELS

        predictions = self.model.predict_entities(
            text, labels, threshold=threshold
        )

        entities = []
        seen = set()
        for pred in predictions:
            key = (pred["text"], pred["label"])
            if key in seen:
                continue
            seen.add(key)

            entities.append(
                Entity(
                    name=pred["text"],
                    entity_type=pred["label"],
                    source_document_id=document_id or "",
                    source_chunk_id=chunk_id,
                    confidence=pred["score"],
                )
            )

        return entities

    def extract_batch(
        self,
        texts: list[str],
        labels: list[str] | None = None,
        threshold: float = 0.4,
    ) -> list[list[Entity]]:
        return [
            self.extract(text, labels, threshold)
            for text in texts
        ]
