from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from rag.ingestion.reddit import RedditIngestor


def test_reddit_ingestor_init():
    """Test that RedditIngestor can be instantiated (doesn't require API keys for init)."""
    ingestor = RedditIngestor(client_id="fake", client_secret="fake")
    assert ingestor.reddit is not None
    assert ingestor.media_chunker is not None


def test_reddit_ingest_mocked():
    """Test Reddit ingestion with mocked PRAW submission."""
    ingestor = RedditIngestor(client_id="fake", client_secret="fake")

    # Mock submission
    mock_submission = MagicMock()
    mock_submission.title = "Test Post Title"
    mock_submission.selftext = "This is the body of the test post."
    mock_submission.url = "https://www.reddit.com/r/Python/comments/abc123/test_post"
    mock_submission.id = "abc123"
    mock_submission.author = MagicMock(__str__=lambda self: "test_user")
    mock_submission.subreddit = MagicMock(__str__=lambda self: "Python")
    mock_submission.score = 42
    mock_submission.num_comments = 5
    mock_submission.created_utc = 1700000000.0  # 2023-11-14

    # Mock comments
    mock_comment1 = MagicMock()
    mock_comment1.body = "Great post!"
    mock_comment2 = MagicMock()
    mock_comment2.body = "Thanks for sharing."
    mock_submission.comments = MagicMock()
    mock_submission.comments.replace_more = MagicMock()
    mock_submission.comments.__getitem__ = lambda self, key: [mock_comment1, mock_comment2][key]
    mock_submission.comments.__iter__ = lambda self: iter([mock_comment1, mock_comment2])
    # Support slicing [:20]
    mock_submission.comments.__class__ = list
    type(mock_submission.comments).__getitem__ = lambda self, key: [mock_comment1, mock_comment2].__getitem__(key)

    with patch.object(ingestor.reddit, "submission", return_value=mock_submission):
        doc, chunks = ingestor.ingest(
            "https://www.reddit.com/r/Python/comments/abc123/test_post"
        )

    assert doc.platform.value == "reddit"
    assert doc.title == "Test Post Title"
    assert doc.author == "test_user"
    assert doc.metadata["subreddit"] == "Python"
    assert doc.metadata["score"] == 42
    assert doc.created_at is not None
    assert doc.created_at.year == 2023

    # Should have at least the post chunk
    assert len(chunks) >= 1
    assert "Test Post Title" in chunks[0].content


@pytest.mark.network
def test_reddit_ingest_live():
    """Integration test - requires valid Reddit API credentials and network."""
    from rag.config import settings
    if not settings.reddit_client_id:
        pytest.skip("No Reddit credentials configured")

    ingestor = RedditIngestor()
    doc, chunks = ingestor.ingest(
        "https://www.reddit.com/r/Python/comments/1h0j0kz/what_are_your_favorite_python_libraries/"
    )
    assert doc.platform.value == "reddit"
    assert len(chunks) >= 1
    assert doc.created_at is not None
