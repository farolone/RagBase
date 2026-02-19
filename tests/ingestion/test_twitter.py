from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.ingestion.twitter import TwitterIngestor


@pytest.fixture
def ingestor():
    return TwitterIngestor(cookies_path="/tmp/fake_cookies.json")


def test_extract_tweet_id(ingestor):
    assert ingestor._extract_tweet_id("https://x.com/user/status/1234567890") == "1234567890"
    assert ingestor._extract_tweet_id("https://twitter.com/user/status/9876543210") == "9876543210"
    assert ingestor._extract_tweet_id("1234567890") == "1234567890"


def test_extract_tweet_id_invalid(ingestor):
    with pytest.raises(ValueError):
        ingestor._extract_tweet_id("not-a-tweet-url")


def test_twitter_ingestor_init():
    ingestor = TwitterIngestor()
    assert ingestor._client is None
    assert ingestor.media_chunker is not None
    assert ingestor._cookies_path == "/root/rag/twitter_cookies.json"


def test_twitter_ingestor_custom_cookies():
    ingestor = TwitterIngestor(cookies_path="/custom/path.json")
    assert ingestor._cookies_path == "/custom/path.json"


def test_twitter_ingest_single_tweet(ingestor):
    """Test single tweet ingestion with mocked twikit client."""
    mock_user = MagicMock()
    mock_user.name = "TestUser"
    mock_user.id = "12345"

    mock_tweet = MagicMock()
    mock_tweet.text = "This is a test tweet about Python programming."
    mock_tweet.id = "9999999"
    mock_tweet.user = mock_user
    mock_tweet.favorite_count = 10
    mock_tweet.retweet_count = 3
    mock_tweet.created_at = "Wed Feb 19 12:00:00 +0000 2026"
    mock_tweet.in_reply_to_tweet_id = None

    mock_client = AsyncMock()
    mock_client.get_tweet_by_id = AsyncMock(return_value=mock_tweet)

    with patch.object(ingestor, "_get_client", return_value=mock_client):
        doc, chunks = ingestor.ingest("https://x.com/TestUser/status/9999999")

    assert doc.platform.value == "twitter"
    assert doc.title == "Tweet by TestUser"
    assert doc.author == "TestUser"
    assert doc.metadata["tweet_id"] == "9999999"
    assert doc.metadata["likes"] == 10
    assert doc.created_at is not None
    assert doc.created_at.year == 2026

    assert len(chunks) == 1
    assert "Python programming" in chunks[0].content


def test_twitter_ingest_thread(ingestor):
    """Test thread ingestion (self-replies) with mocked twikit client."""
    mock_user = MagicMock()
    mock_user.name = "ThreadUser"
    mock_user.id = "12345"

    # Parent tweet (first in thread)
    mock_parent = MagicMock()
    mock_parent.text = "Thread starts here. 1/2"
    mock_parent.id = "1000"
    mock_parent.user = mock_user
    mock_parent.in_reply_to_tweet_id = None

    # Reply tweet (second in thread, by same user)
    mock_reply = MagicMock()
    mock_reply.text = "Thread continues. 2/2"
    mock_reply.id = "1001"
    mock_reply.user = mock_user
    mock_reply.favorite_count = 5
    mock_reply.retweet_count = 1
    mock_reply.created_at = "Wed Feb 19 13:00:00 +0000 2026"
    mock_reply.in_reply_to_tweet_id = "1000"

    mock_client = AsyncMock()

    async def mock_get_tweet(tweet_id):
        if tweet_id == "1001":
            return mock_reply
        elif tweet_id == "1000":
            return mock_parent
        raise ValueError(f"Unknown tweet: {tweet_id}")

    mock_client.get_tweet_by_id = AsyncMock(side_effect=mock_get_tweet)

    with patch.object(ingestor, "_get_client", return_value=mock_client):
        doc, chunks = ingestor.ingest("https://x.com/ThreadUser/status/1001")

    assert doc.platform.value == "twitter"
    assert doc.title == "Tweet by ThreadUser"

    # Thread: 1 parent chunk + 2 individual tweet chunks = 3
    assert len(chunks) == 3
    assert chunks[0].metadata.get("type") == "thread"
    assert "Thread starts here" in chunks[0].content
    assert "Thread continues" in chunks[0].content


@pytest.mark.network
def test_twitter_ingest_live():
    """Integration test - requires twitter_cookies.json."""
    from pathlib import Path
    if not Path("/root/rag/twitter_cookies.json").exists():
        pytest.skip("No Twitter cookies configured")

    ingestor = TwitterIngestor()
    doc, chunks = ingestor.ingest("https://x.com/elonmusk/status/1234567890")
    assert doc.platform.value == "twitter"
    assert len(chunks) >= 1
