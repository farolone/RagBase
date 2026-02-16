import pytest
from rag.storage.postgres import PostgresStore
from rag.models import Document, Platform


@pytest.fixture
def store():
    s = PostgresStore()
    yield s
    s.cleanup_test_data()


def test_save_and_get_document(store):
    doc = Document(
        title="Test Doc",
        source_url="https://example.com",
        platform=Platform.WEB,
        author="Alice",
        language="en",
    )
    store.save_document(doc)
    retrieved = store.get_document(doc.id)
    assert retrieved is not None
    assert retrieved.title == "Test Doc"


def test_search_by_platform(store):
    doc = Document(
        title="YouTube Video",
        source_url="https://youtube.com/watch?v=123",
        platform=Platform.YOUTUBE,
    )
    store.save_document(doc)
    results = store.search_documents(platform=Platform.YOUTUBE)
    assert any(d.id == doc.id for d in results)


def test_search_by_author(store):
    doc = Document(
        title="By Bob",
        platform=Platform.REDDIT,
        author="bob",
    )
    store.save_document(doc)
    results = store.search_documents(author="bob")
    assert any(d.id == doc.id for d in results)
