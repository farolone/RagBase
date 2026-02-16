from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from rag.models import Topic


class TopicModeler:
    """BERTopic-based hierarchical topic modeling with incremental support."""

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        min_topic_size: int = 5,
    ):
        self.sentence_model = SentenceTransformer(embedding_model)
        self.model = BERTopic(
            embedding_model=self.sentence_model,
            min_topic_size=min_topic_size,
            language="multilingual",
            calculate_probabilities=False,
        )
        self._is_fitted = False

    def fit(self, texts: list[str]) -> list[Topic]:
        topics, _ = self.model.fit_transform(texts)
        self._is_fitted = True
        return self._extract_topics()

    def transform(self, texts: list[str]) -> list[int]:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        topics, _ = self.model.transform(texts)
        return topics

    def get_topics(self) -> list[Topic]:
        if not self._is_fitted:
            return []
        return self._extract_topics()

    def get_topic_for_text(self, text: str) -> int:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet.")
        topics, _ = self.model.transform([text])
        return topics[0]

    def merge_models(self, other: "TopicModeler") -> None:
        if self._is_fitted and other._is_fitted:
            self.model = BERTopic.merge_models(
                [self.model, other.model]
            )

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = BERTopic.load(path)
        self._is_fitted = True

    def _extract_topics(self) -> list[Topic]:
        topic_info = self.model.get_topic_info()
        topics = []
        for _, row in topic_info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                continue  # Skip outlier topic
            topics.append(
                Topic(
                    name=row["Name"],
                    bertopic_id=topic_id,
                    metadata={"count": int(row["Count"])},
                )
            )
        return topics
