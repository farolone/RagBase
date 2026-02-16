from rag.models import Document, Chunk, Entity, Platform


def test_document_creation():
    doc = Document(
        title="Test Article",
        source_url="https://example.com/article",
        platform=Platform.WEB,
        author="John Doe",
        language="en",
    )
    assert doc.title == "Test Article"
    assert doc.platform == Platform.WEB
    assert doc.id is not None


def test_chunk_creation():
    chunk = Chunk(
        document_id="doc-123",
        content="This is a test chunk.",
        chunk_index=0,
        token_count=6,
        metadata={"section": "intro"},
    )
    assert chunk.content == "This is a test chunk."
    assert chunk.parent_chunk_id is None


def test_chunk_hierarchy():
    parent = Chunk(
        document_id="doc-123",
        content="Parent content with more context.",
        chunk_index=0,
        token_count=10,
    )
    child = Chunk(
        document_id="doc-123",
        content="Child content.",
        chunk_index=0,
        token_count=4,
        parent_chunk_id=parent.id,
    )
    assert child.parent_chunk_id == parent.id


def test_entity_creation():
    entity = Entity(
        name="Berlin",
        entity_type="LOCATION",
        source_document_id="doc-123",
        source_chunk_id="chunk-456",
        confidence=0.95,
    )
    assert entity.entity_type == "LOCATION"
    assert entity.confidence == 0.95


def test_platform_enum():
    assert Platform.YOUTUBE.value == "youtube"
    assert Platform.TWITTER.value == "twitter"
    assert Platform.REDDIT.value == "reddit"
    assert Platform.WEB.value == "web"
    assert Platform.PDF.value == "pdf"
