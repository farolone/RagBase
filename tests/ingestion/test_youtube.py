import pytest
from rag.ingestion.youtube import YouTubeIngestor
from rag.models import Platform


@pytest.fixture
def ingestor():
    return YouTubeIngestor()


def test_extract_video_id(ingestor):
    assert ingestor._extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert ingestor._extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert ingestor._extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid(ingestor):
    with pytest.raises(ValueError):
        ingestor._extract_video_id("not-a-valid-url")


def test_group_by_time(ingestor):
    segments = [
        {"text": "Hello", "start": 0.0},
        {"text": "World", "start": 30.0},
        {"text": "Next", "start": 65.0},
        {"text": "More", "start": 90.0},
    ]
    groups = ingestor._group_by_time(segments, window_seconds=60.0)
    assert len(groups) == 2
    assert len(groups[0]) == 2  # first 60s window
    assert len(groups[1]) == 2  # second window


@pytest.mark.network
def test_youtube_ingest_live(ingestor):
    """Integration test - requires network. Run with: pytest -m network"""
    doc, chunks = ingestor.ingest("https://www.youtube.com/watch?v=jGwO_UgTS7I")
    assert doc.platform == Platform.YOUTUBE
    assert len(chunks) >= 1
    assert chunks[0].metadata["video_id"] == "jGwO_UgTS7I"
