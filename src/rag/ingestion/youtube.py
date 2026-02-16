import re

from youtube_transcript_api import YouTubeTranscriptApi

from rag.ingestion.base import BaseIngestor
from rag.models import Chunk, Document, Platform
from rag.processing.chunking import MediaChunker


class YouTubeIngestor(BaseIngestor):
    def __init__(self):
        self.media_chunker = MediaChunker()

    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        video_id = self._extract_video_id(source)

        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)

        segments = [
            {
                "text": entry.text,
                "start": entry.start,
                "chapter": "default",
            }
            for entry in transcript_list
        ]

        # Group into ~60 second windows if no chapters
        grouped = self._group_by_time(segments, window_seconds=60)

        doc = Document(
            title=f"YouTube: {video_id}",
            source_url=f"https://www.youtube.com/watch?v={video_id}",
            platform=Platform.YOUTUBE,
            metadata={"video_id": video_id},
        )

        chunks = []
        for idx, group in enumerate(grouped):
            text = " ".join(s["text"] for s in group)
            start_time = group[0]["start"]
            chunks.append(
                Chunk(
                    document_id=doc.id,
                    content=text,
                    chunk_index=idx,
                    token_count=len(text.split()),
                    metadata={
                        "platform": "youtube",
                        "video_id": video_id,
                        "start_time": start_time,
                        "source_url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}",
                    },
                )
            )

        return doc, chunks

    @staticmethod
    def _extract_video_id(url: str) -> str:
        patterns = [
            r"(?:v=|\/v\/|youtu\.be\/)([a-zA-Z0-9_-]{11})",
            r"^([a-zA-Z0-9_-]{11})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"Cannot extract video ID from: {url}")

    @staticmethod
    def _group_by_time(
        segments: list[dict], window_seconds: float = 60.0
    ) -> list[list[dict]]:
        if not segments:
            return []
        groups = []
        current_group = [segments[0]]
        group_start = segments[0]["start"]

        for seg in segments[1:]:
            if seg["start"] - group_start >= window_seconds:
                groups.append(current_group)
                current_group = [seg]
                group_start = seg["start"]
            else:
                current_group.append(seg)

        if current_group:
            groups.append(current_group)
        return groups
