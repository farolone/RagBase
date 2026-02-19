"""Prefect tasks for document ingestion."""

import logging

from prefect import task

from rag.pipeline.dedup import is_already_ingested

logger = logging.getLogger(__name__)


def _get_youtube_client():
    """Get an authenticated YouTube API client.
    Prefers OAuth token (needed for private playlists / Watch Later),
    falls back to API key (sufficient for public content / channels).
    """
    from googleapiclient.discovery import build
    from pathlib import Path

    token_path = Path("/root/rag/youtube_oauth_token.json")
    if token_path.exists():
        try:
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_file(str(token_path))
            return build("youtube", "v3", credentials=creds)
        except Exception as e:
            print(f"OAuth token invalid, falling back to API key: {e}")

    from rag.config import settings
    if settings.youtube_api_key:
        return build("youtube", "v3", developerKey=settings.youtube_api_key)

    return None


def _run_full_ingest(source: str, source_type: str) -> dict:
    """Shared ingestion logic: ingest, embed, store, NER, graph."""
    from rag.ingestion.pdf import PDFIngestor
    from rag.ingestion.youtube import YouTubeIngestor
    from rag.ingestion.web import WebIngestor
    from rag.ingestion.document import DocumentIngestor
    from rag.processing.embedding import Embedder
    from rag.processing.ner import EntityExtractor
    from rag.processing.graph_builder import GraphBuilder
    from rag.storage.qdrant import QdrantStore
    from rag.storage.postgres import PostgresStore

    from rag.ingestion.reddit import RedditIngestor

    ingestors = {
        "pdf": PDFIngestor,
        "youtube": YouTubeIngestor,
        "web": WebIngestor,
        "document": DocumentIngestor,
        "reddit": RedditIngestor,
    }

    if source_type not in ingestors:
        raise ValueError(f"Unsupported source type: {source_type}")

    ingestor = ingestors[source_type]()
    doc, chunks = ingestor.ingest(source)

    embedder = Embedder()
    qdrant = QdrantStore()
    qdrant.ensure_collection()
    postgres = PostgresStore()

    for chunk in chunks:
        emb = embedder.embed(chunk.content)
        qdrant.upsert(
            chunk=chunk,
            dense_vector=emb.dense,
            sparse_indices=emb.sparse_indices,
            sparse_values=emb.sparse_values,
        )

    postgres.save_document(doc)
    postgres.update_document_counts(doc.id, len(chunks), 0)

    # Save transcript if available (YouTube)
    if hasattr(ingestor, '_last_segments') and ingestor._last_segments:
        postgres.save_transcript(doc.id, ingestor._last_segments, ingestor._last_transcript_source)

    # Save published_at if available
    if doc.created_at:
        postgres.update_published_at(doc.id, doc.created_at)

    ner = EntityExtractor()
    graph = GraphBuilder()
    all_entities = []
    for chunk in chunks:
        entities = ner.extract(chunk.content, document_id=doc.id, chunk_id=chunk.id)
        all_entities.extend(entities)

    graph.process_document(doc, all_entities)
    graph.close()

    postgres.update_document_counts(doc.id, len(chunks), len(all_entities))

    return {
        "doc_id": doc.id,
        "title": doc.title,
        "chunks": len(chunks),
        "entities": len(all_entities),
    }


@task(retries=2, retry_delay_seconds=30, log_prints=True)
def ingest_web_url(url: str) -> dict | None:
    """Ingest a single web URL with dedup check."""
    if is_already_ingested(url):
        print(f"SKIP (already ingested): {url}")
        return None

    print(f"Ingesting web: {url}")
    result = _run_full_ingest(url, "web")
    print(f"Done: {result['title']} ({result['chunks']} chunks, {result['entities']} entities)")
    return result


@task(retries=2, retry_delay_seconds=30, log_prints=True)
def ingest_youtube_video(url: str) -> dict | None:
    """Ingest a single YouTube video with dedup check."""
    if is_already_ingested(url):
        print(f"SKIP (already ingested): {url}")
        return None

    print(f"Ingesting YouTube: {url}")
    result = _run_full_ingest(url, "youtube")
    print(f"Done: {result['title']} ({result['chunks']} chunks, {result['entities']} entities)")
    return result


@task(retries=1, retry_delay_seconds=30, log_prints=True)
def ingest_file(file_path: str) -> dict | None:
    """Ingest a single local file with dedup check."""
    from pathlib import Path
    resolved = str(Path(file_path).resolve())

    if is_already_ingested(resolved):
        print(f"SKIP (already ingested): {resolved}")
        return None

    print(f"Ingesting file: {resolved}")
    result = _run_full_ingest(resolved, "document")
    print(f"Done: {result['title']} ({result['chunks']} chunks, {result['entities']} entities)")
    return result


@task(retries=1, retry_delay_seconds=30, log_prints=True)
def scan_and_ingest_folder(folder_config: dict) -> dict:
    """Scan a folder and ingest all new files."""
    from rag.ingestion.folder_scanner import FolderScanner

    scanner = FolderScanner(
        extensions=folder_config.get("extensions"),
        exclude_patterns=folder_config.get("exclude_patterns"),
        max_file_size_mb=folder_config.get("max_file_size_mb", 100),
    )
    scan_result = scanner.scan(folder_config["path"])
    print(f"Scanned {folder_config['path']}: {len(scan_result.files)} files, {scan_result.skipped_count} skipped")

    ingested = 0
    skipped = 0
    errors = 0

    for f in scan_result.files:
        resolved = str(f.path.resolve())
        if is_already_ingested(resolved):
            skipped += 1
            continue
        try:
            result = ingest_file(resolved)
            if result:
                ingested += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"ERROR ingesting {f.relative_path}: {e}")
            errors += 1

    print(f"Folder done: {ingested} ingested, {skipped} skipped, {errors} errors")
    return {"ingested": ingested, "skipped": skipped, "errors": errors}


@task(retries=1, retry_delay_seconds=60, log_prints=True)
def fetch_youtube_watch_later() -> list[str]:
    """Fetch video URLs from YouTube Watch Later playlist (requires OAuth)."""
    try:
        youtube = _get_youtube_client()
        if not youtube:
            print("SKIP YouTube Watch Later: no credentials configured")
            return []

        urls = []
        request = youtube.playlistItems().list(
            part="contentDetails", playlistId="WL", maxResults=50
        )
        response = request.execute()
        for item in response.get("items", []):
            video_id = item["contentDetails"]["videoId"]
            urls.append(f"https://www.youtube.com/watch?v={video_id}")
        print(f"Found {len(urls)} videos in Watch Later")
        return urls
    except Exception as e:
        print(f"YouTube Watch Later error: {e}")
        return []


@task(retries=1, retry_delay_seconds=60, log_prints=True)
def fetch_youtube_playlist(playlist_id: str) -> list[str]:
    """Fetch video URLs from a YouTube playlist."""
    try:
        youtube = _get_youtube_client()
        if not youtube:
            print("SKIP YouTube playlist: no credentials configured")
            return []

        urls = []
        next_page = None
        while True:
            request = youtube.playlistItems().list(
                part="contentDetails", playlistId=playlist_id,
                maxResults=50, pageToken=next_page,
            )
            response = request.execute()
            for item in response.get("items", []):
                video_id = item["contentDetails"]["videoId"]
                urls.append(f"https://www.youtube.com/watch?v={video_id}")
            next_page = response.get("nextPageToken")
            if not next_page:
                break
        print(f"Found {len(urls)} videos in playlist {playlist_id}")
        return urls
    except Exception as e:
        print(f"YouTube playlist error: {e}")
        return []


@task(retries=1, retry_delay_seconds=60, log_prints=True)
def fetch_youtube_channel_uploads(channel_id: str, max_results: int = 10) -> list[str]:
    """Fetch recent video URLs from a YouTube channel's uploads."""
    try:
        youtube = _get_youtube_client()
        if not youtube:
            print("SKIP YouTube channel: no credentials configured")
            return []

        # Get the uploads playlist ID for this channel
        ch_resp = youtube.channels().list(part="contentDetails", id=channel_id).execute()
        items = ch_resp.get("items", [])
        if not items:
            # Try as handle (e.g. @3blue1brown)
            ch_resp = youtube.channels().list(part="contentDetails", forHandle=channel_id).execute()
            items = ch_resp.get("items", [])
        if not items:
            print(f"YouTube channel not found: {channel_id}")
            return []

        uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        request = youtube.playlistItems().list(
            part="contentDetails", playlistId=uploads_playlist, maxResults=max_results,
        )
        response = request.execute()
        urls = []
        for item in response.get("items", []):
            video_id = item["contentDetails"]["videoId"]
            urls.append(f"https://www.youtube.com/watch?v={video_id}")

        print(f"Found {len(urls)} recent videos from channel {channel_id}")
        return urls
    except Exception as e:
        print(f"YouTube channel error for {channel_id}: {e}")
        return []


@task(retries=1, retry_delay_seconds=60, log_prints=True)
def fetch_reddit_saved(limit: int = 50) -> list[str]:
    """Fetch saved post URLs from Reddit."""
    try:
        import praw
        from rag.config import settings

        if not settings.reddit_client_id or not settings.reddit_client_secret:
            print("SKIP Reddit: no credentials configured")
            return []

        reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent="rag-wissensdatenbank/0.1",
            username=settings.reddit_username,
            password=settings.reddit_password,
        )

        urls = []
        for item in reddit.user.me().saved(limit=limit):
            if hasattr(item, "permalink"):
                urls.append(f"https://www.reddit.com{item.permalink}")
        print(f"Found {len(urls)} saved Reddit posts")
        return urls
    except Exception as e:
        print(f"Reddit fetch error: {e}")
        return []


@task(retries=1, retry_delay_seconds=60, log_prints=True)
def ingest_reddit_post(url: str) -> dict | None:
    """Ingest a single Reddit post."""
    if is_already_ingested(url):
        print(f"SKIP (already ingested): {url}")
        return None

    try:
        from rag.ingestion.reddit import RedditIngestor
        from rag.processing.embedding import Embedder
        from rag.processing.ner import EntityExtractor
        from rag.processing.graph_builder import GraphBuilder
        from rag.storage.qdrant import QdrantStore
        from rag.storage.postgres import PostgresStore

        print(f"Ingesting Reddit: {url}")
        ingestor = RedditIngestor()
        doc, chunks = ingestor.ingest(url)

        embedder = Embedder()
        qdrant = QdrantStore()
        qdrant.ensure_collection()
        postgres = PostgresStore()

        for chunk in chunks:
            emb = embedder.embed(chunk.content)
            qdrant.upsert(
                chunk=chunk,
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )

        postgres.save_document(doc)

        ner = EntityExtractor()
        graph = GraphBuilder()
        all_entities = []
        for chunk in chunks:
            entities = ner.extract(chunk.content, document_id=doc.id, chunk_id=chunk.id)
            all_entities.extend(entities)

        graph.process_document(doc, all_entities)
        graph.close()

        postgres.update_document_counts(doc.id, len(chunks), len(all_entities))

        result = {"doc_id": doc.id, "title": doc.title, "chunks": len(chunks), "entities": len(all_entities)}
        print(f"Done: {result['title']} ({result['chunks']} chunks)")
        return result
    except Exception as e:
        print(f"Reddit ingest error for {url}: {e}")
        return None


@task(retries=1, retry_delay_seconds=60, log_prints=True)
def fetch_twitter_bookmarks(limit: int = 50) -> list[str]:
    """Fetch bookmarked tweet URLs from Twitter/X."""
    try:
        import asyncio
        from twikit import Client

        cookies_path = "/root/rag/twitter_cookies.json"
        try:
            client = Client("de-DE")
            client.load_cookies(cookies_path)
        except FileNotFoundError:
            print("SKIP Twitter: no cookies file found")
            return []

        async def _get_bookmarks():
            bookmarks = await client.get_bookmarks(count=limit)
            return [f"https://twitter.com/i/status/{t.id}" for t in bookmarks]

        urls = asyncio.run(_get_bookmarks())
        print(f"Found {len(urls)} Twitter bookmarks")
        return urls
    except Exception as e:
        print(f"Twitter fetch error: {e}")
        return []


@task(retries=1, retry_delay_seconds=60, log_prints=True)
def ingest_tweet(url: str) -> dict | None:
    """Ingest a single tweet."""
    if is_already_ingested(url):
        print(f"SKIP (already ingested): {url}")
        return None

    try:
        from rag.ingestion.twitter import TwitterIngestor
        from rag.processing.embedding import Embedder
        from rag.processing.ner import EntityExtractor
        from rag.processing.graph_builder import GraphBuilder
        from rag.storage.qdrant import QdrantStore
        from rag.storage.postgres import PostgresStore

        print(f"Ingesting Tweet: {url}")
        ingestor = TwitterIngestor(cookies_path="/root/rag/twitter_cookies.json")
        doc, chunks = ingestor.ingest(url)

        embedder = Embedder()
        qdrant = QdrantStore()
        qdrant.ensure_collection()
        postgres = PostgresStore()

        for chunk in chunks:
            emb = embedder.embed(chunk.content)
            qdrant.upsert(
                chunk=chunk,
                dense_vector=emb.dense,
                sparse_indices=emb.sparse_indices,
                sparse_values=emb.sparse_values,
            )

        postgres.save_document(doc)

        ner = EntityExtractor()
        graph = GraphBuilder()
        all_entities = []
        for chunk in chunks:
            entities = ner.extract(chunk.content, document_id=doc.id, chunk_id=chunk.id)
            all_entities.extend(entities)

        graph.process_document(doc, all_entities)
        graph.close()

        postgres.update_document_counts(doc.id, len(chunks), len(all_entities))

        result = {"doc_id": doc.id, "title": doc.title, "chunks": len(chunks), "entities": len(all_entities)}
        print(f"Done: {result['title']} ({result['chunks']} chunks)")
        return result
    except Exception as e:
        print(f"Tweet ingest error for {url}: {e}")
        return None
