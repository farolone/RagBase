import pytest
from rag.ingestion.reddit import RedditIngestor


def test_reddit_ingestor_init():
    """Test that RedditIngestor can be instantiated (doesn't require API keys for init)."""
    ingestor = RedditIngestor(client_id="fake", client_secret="fake")
    assert ingestor.reddit is not None
    assert ingestor.media_chunker is not None


@pytest.mark.network
def test_reddit_ingest_live():
    """Integration test - requires valid Reddit API credentials."""
    ingestor = RedditIngestor()
    doc, chunks = ingestor.ingest(
        "https://www.reddit.com/r/Python/comments/example"
    )
    assert doc.platform.value == "reddit"
    assert len(chunks) >= 1
