import pytest
from rag.ingestion.twitter import TwitterIngestor


@pytest.fixture
def ingestor():
    return TwitterIngestor()


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
