import pytest
from rag.processing.topics import TopicModeler


@pytest.fixture(scope="module")
def modeler():
    return TopicModeler(min_topic_size=2)


def test_fit_and_extract_topics(modeler):
    # Need enough documents with distinct topics
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing handles text data.",
        "Computer vision processes images and video.",
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
        "London is the capital of England.",
        "Tokyo is the capital of Japan.",
        "Python is a popular programming language.",
        "JavaScript runs in web browsers.",
        "Docker containers package applications.",
        "Kubernetes orchestrates container deployments.",
    ]
    topics = modeler.fit(texts)
    assert len(topics) >= 1


def test_transform_new_text(modeler):
    topic_id = modeler.get_topic_for_text("AI and machine learning are transforming tech.")
    assert isinstance(topic_id, int)


def test_not_fitted():
    m = TopicModeler(min_topic_size=2)
    with pytest.raises(RuntimeError):
        m.transform(["test"])
