from datetime import datetime, timezone

import praw

from rag.ingestion.base import BaseIngestor
from rag.models import Chunk, Document, Platform
from rag.processing.chunking import MediaChunker
from rag.config import settings


class RedditIngestor(BaseIngestor):
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str = "rag-bot/0.1",
    ):
        self.reddit = praw.Reddit(
            client_id=client_id or getattr(settings, "reddit_client_id", ""),
            client_secret=client_secret or getattr(settings, "reddit_client_secret", ""),
            user_agent=user_agent,
        )
        self.media_chunker = MediaChunker()

    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        submission = self.reddit.submission(url=source)

        doc = Document(
            title=submission.title,
            source_url=source,
            platform=Platform.REDDIT,
            author=str(submission.author) if submission.author else None,
            created_at=datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
            metadata={
                "subreddit": str(submission.subreddit),
                "score": submission.score,
                "num_comments": submission.num_comments,
                "submission_id": submission.id,
            },
        )

        post_data = {
            "title": submission.title,
            "body": submission.selftext or "",
            "comments": [],
        }

        # Get top-level comments
        submission.comments.replace_more(limit=0)
        for comment in submission.comments[:20]:  # Limit to top 20
            if hasattr(comment, "body"):
                post_data["comments"].append(comment.body)

        chunks = self.media_chunker.chunk_reddit(
            post=post_data,
            document_id=doc.id,
            metadata={
                "platform": "reddit",
                "subreddit": str(submission.subreddit),
                "source_url": source,
            },
        )

        return doc, chunks

    def ingest_subreddit(
        self, subreddit_name: str, sort: str = "hot", limit: int = 25
    ) -> list[tuple[Document, list[Chunk]]]:
        subreddit = self.reddit.subreddit(subreddit_name)
        getter = getattr(subreddit, sort)
        results = []
        for submission in getter(limit=limit):
            try:
                result = self.ingest(submission.url)
                results.append(result)
            except Exception:
                continue
        return results
