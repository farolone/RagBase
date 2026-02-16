import asyncio
import re

from rag.ingestion.base import BaseIngestor
from rag.models import Chunk, Document, Platform
from rag.processing.chunking import MediaChunker


class TwitterIngestor(BaseIngestor):
    """Twitter/X ingestor using Twikit. Requires authenticated session."""

    def __init__(self, cookies_path: str | None = None):
        self._cookies_path = cookies_path
        self._client = None
        self.media_chunker = MediaChunker()

    async def _get_client(self):
        if self._client is None:
            from twikit import Client
            self._client = Client("en-US")
            if self._cookies_path:
                self._client.load_cookies(self._cookies_path)
        return self._client

    def ingest(self, source: str) -> tuple[Document, list[Chunk]]:
        return asyncio.run(self._ingest_async(source))

    async def _ingest_async(self, source: str) -> tuple[Document, list[Chunk]]:
        tweet_id = self._extract_tweet_id(source)
        client = await self._get_client()

        tweet = await client.get_tweet_by_id(tweet_id)

        # Collect thread if this is part of one
        tweets_data = []
        tweets_data.append({
            "text": tweet.text,
            "id": tweet.id,
            "author": tweet.user.name if tweet.user else None,
        })

        doc = Document(
            title=f"Tweet by {tweet.user.name if tweet.user else 'unknown'}",
            source_url=source,
            platform=Platform.TWITTER,
            author=tweet.user.name if tweet.user else None,
            metadata={
                "tweet_id": tweet_id,
                "likes": tweet.favorite_count,
                "retweets": tweet.retweet_count,
            },
        )

        if len(tweets_data) > 1:
            chunks = self.media_chunker.chunk_twitter_thread(
                tweets=tweets_data,
                document_id=doc.id,
                metadata={"platform": "twitter", "source_url": source},
            )
        else:
            chunks = [
                Chunk(
                    document_id=doc.id,
                    content=tweet.text,
                    chunk_index=0,
                    token_count=len(tweet.text.split()),
                    metadata={
                        "platform": "twitter",
                        "tweet_id": tweet_id,
                        "source_url": source,
                    },
                )
            ]

        return doc, chunks

    @staticmethod
    def _extract_tweet_id(url: str) -> str:
        patterns = [
            r"status/(\d+)",
            r"^(\d+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"Cannot extract tweet ID from: {url}")
