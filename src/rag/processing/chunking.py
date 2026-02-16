from rag.models import Chunk


class HierarchicalChunker:
    """Hierarchical chunker creating leaf, parent, and grandparent chunks."""

    def __init__(
        self,
        leaf_size: int = 512,
        parent_size: int = 1024,
        grandparent_size: int = 2048,
        overlap: int = 50,
    ):
        self.leaf_size = leaf_size
        self.parent_size = parent_size
        self.grandparent_size = grandparent_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        metadata = metadata or {}
        words = text.split()

        if len(words) <= self.leaf_size:
            return [
                Chunk(
                    document_id=document_id,
                    content=text,
                    chunk_index=0,
                    token_count=len(words),
                    metadata=metadata,
                )
            ]

        # Create parent-level chunks first
        parent_chunks = self._split_words(
            words, self.parent_size, self.overlap
        )
        all_chunks = []

        for pi, parent_words in enumerate(parent_chunks):
            parent_text = " ".join(parent_words)
            parent = Chunk(
                document_id=document_id,
                content=parent_text,
                chunk_index=pi,
                token_count=len(parent_words),
                metadata={**metadata, "level": "parent"},
            )
            all_chunks.append(parent)

            # Split parent into leaf chunks
            leaf_groups = self._split_words(
                parent_words, self.leaf_size, self.overlap
            )
            for li, leaf_words in enumerate(leaf_groups):
                leaf_text = " ".join(leaf_words)
                leaf = Chunk(
                    document_id=document_id,
                    content=leaf_text,
                    chunk_index=pi * 100 + li,
                    token_count=len(leaf_words),
                    parent_chunk_id=parent.id,
                    metadata={**metadata, "level": "leaf"},
                )
                all_chunks.append(leaf)

        return all_chunks

    @staticmethod
    def _split_words(
        words: list[str], size: int, overlap: int
    ) -> list[list[str]]:
        if len(words) <= size:
            return [words]
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append(words[start:end])
            start += size - overlap
            if start >= len(words):
                break
        return chunks


class MediaChunker:
    """Media-specific chunking strategies."""

    def chunk_youtube(
        self,
        segments: list[dict],
        document_id: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []

        # Group segments by chapter
        chapters: dict[str, list[dict]] = {}
        for seg in segments:
            chapter = seg.get("chapter", "default")
            chapters.setdefault(chapter, []).append(seg)

        for idx, (chapter_name, segs) in enumerate(chapters.items()):
            text = " ".join(s["text"] for s in segs)
            start_time = segs[0].get("start", 0.0)
            chunks.append(
                Chunk(
                    document_id=document_id,
                    content=text,
                    chunk_index=idx,
                    token_count=len(text.split()),
                    metadata={
                        **metadata,
                        "chapter": chapter_name,
                        "start_time": start_time,
                        "type": "youtube_chapter",
                    },
                )
            )

        return chunks

    def chunk_reddit(
        self,
        post: dict,
        document_id: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []

        # Post title + body as first chunk
        post_text = post.get("title", "")
        if post.get("body"):
            post_text += "\n\n" + post["body"]

        post_chunk = Chunk(
            document_id=document_id,
            content=post_text,
            chunk_index=0,
            token_count=len(post_text.split()),
            metadata={**metadata, "type": "post"},
        )
        chunks.append(post_chunk)

        # Each comment as a separate chunk linked to the post
        for ci, comment in enumerate(post.get("comments", [])):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    content=comment,
                    chunk_index=ci + 1,
                    token_count=len(comment.split()),
                    parent_chunk_id=post_chunk.id,
                    metadata={**metadata, "type": "comment"},
                )
            )

        return chunks

    def chunk_twitter_thread(
        self,
        tweets: list[dict],
        document_id: str,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []

        # Full thread as parent
        full_text = "\n\n".join(t["text"] for t in tweets)
        parent = Chunk(
            document_id=document_id,
            content=full_text,
            chunk_index=0,
            token_count=len(full_text.split()),
            metadata={**metadata, "type": "thread"},
        )
        chunks.append(parent)

        # Individual tweets as children
        for ti, tweet in enumerate(tweets):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    content=tweet["text"],
                    chunk_index=ti + 1,
                    token_count=len(tweet["text"].split()),
                    parent_chunk_id=parent.id,
                    metadata={
                        **metadata,
                        "type": "tweet",
                        "tweet_id": tweet.get("id"),
                    },
                )
            )

        return chunks
