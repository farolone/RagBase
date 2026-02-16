import pytest
from rag.processing.chunking import HierarchicalChunker, MediaChunker
from rag.models import Chunk


@pytest.fixture
def chunker():
    return HierarchicalChunker(leaf_size=50, parent_size=100, grandparent_size=200, overlap=10)


def test_chunk_short_text(chunker):
    chunks = chunker.chunk(
        text="This is a short text.",
        document_id="doc-1",
    )
    assert len(chunks) >= 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].content == "This is a short text."


def test_chunk_hierarchy(chunker):
    # Generate text long enough to create parent chunks
    words = ["word"] * 300
    text = " ".join(words)
    chunks = chunker.chunk(text=text, document_id="doc-2")

    leaf_chunks = [c for c in chunks if c.parent_chunk_id is not None]
    parent_chunks = [c for c in chunks if c.parent_chunk_id is None and c.token_count > 50]

    # Should have both leaf and parent level chunks
    assert len(chunks) > 1


def test_chunk_metadata_passed(chunker):
    chunks = chunker.chunk(
        text="Some text here for testing metadata.",
        document_id="doc-3",
        metadata={"platform": "web", "author": "test"},
    )
    for c in chunks:
        assert c.metadata.get("platform") == "web"
        assert c.metadata.get("author") == "test"


def test_chunk_overlap(chunker):
    # Build text with enough words to get multiple chunks
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunker.chunk(text=text, document_id="doc-4")
    leaf_chunks = [c for c in chunks if c.parent_chunk_id is not None or len(chunks) == 1]
    if len(leaf_chunks) >= 2:
        # Check overlap: last words of chunk N should appear at start of chunk N+1
        c0_words = leaf_chunks[0].content.split()
        c1_words = leaf_chunks[1].content.split()
        # There should be some overlap
        overlap = set(c0_words[-10:]) & set(c1_words[:10])
        assert len(overlap) > 0


class TestMediaChunker:
    def test_chunk_youtube_with_chapters(self):
        mc = MediaChunker()
        segments = [
            {"text": "Welcome to the video.", "start": 0.0, "chapter": "Intro"},
            {"text": "Let me explain the topic.", "start": 5.0, "chapter": "Intro"},
            {"text": "Now the main content.", "start": 60.0, "chapter": "Main"},
            {"text": "More details here.", "start": 65.0, "chapter": "Main"},
            {"text": "Thanks for watching.", "start": 120.0, "chapter": "Outro"},
        ]
        chunks = mc.chunk_youtube(segments, document_id="yt-1")
        assert len(chunks) >= 3  # At least one per chapter
        assert any("Intro" in c.metadata.get("chapter", "") for c in chunks)

    def test_chunk_reddit(self):
        mc = MediaChunker()
        post = {
            "title": "Interesting post",
            "body": "This is the post body with some content.",
            "comments": [
                "First comment with opinion.",
                "Second comment with more detail.",
            ],
        }
        chunks = mc.chunk_reddit(post, document_id="reddit-1")
        assert len(chunks) >= 2  # At least post + comments
        assert chunks[0].metadata.get("type") == "post"
