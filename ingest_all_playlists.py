"""Ingest all YouTube playlists (except Music) + Liked Videos."""
import logging
import time
import sys
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from rag.ingestion.youtube import YouTubeIngestor
from rag.processing.embedding import Embedder
from rag.processing.ner import EntityExtractor
from rag.processing.graph_builder import GraphBuilder
from rag.storage.qdrant import QdrantStore
from rag.storage.postgres import PostgresStore
from rag.pipeline.dedup import is_already_ingested

# All playlists except Music
PLAYLISTS = {
    "PLU1o0Hb7N8SZ3U9FZTRnDXbXSVqb9-B8N": "Investing",
    "PLU1o0Hb7N8SZkFuIXTbUkeeYtSd4Wuga4": "Claude",
    "PLU1o0Hb7N8Sai672JKg3HGVUGuOxeQMsy": "Schießen",
    "PLU1o0Hb7N8SaTEUezlh3qN3KcWMAtBuPz": "Bitcoin",
    "PLU1o0Hb7N8Sa-aYosYWoso_sdwiGGJmkQ": "Politik",
    "PLU1o0Hb7N8SbVDU1kRcJWVcI2I-KGVyda": "Trading",
    "PLU1o0Hb7N8SbLng1qvAJ0TKiGVbZL88_Z": "AI",
    "PLU1o0Hb7N8SbcDUa9lH6YhxRAMcdoQoXL": "Vatsim",
    "PLU1o0Hb7N8SZkELiVaDOHUA8g7Qqm7jOg": "Mooney",
    "PLU1o0Hb7N8SbHck6ImMWPO4CHnzNf00mO": "Matthias",
    "PLU1o0Hb7N8SaJyeVjQQH9ERcCeWV8QIOr": "IFR",
    "PLU1o0Hb7N8SZ4VzbVrFb15feYnFNX2RQc": "Ppl",
}
INCLUDE_LIKED = True  # Also ingest Liked Videos (LL)


def get_youtube_client():
    """Get authenticated YouTube API client with token refresh."""
    creds = Credentials.from_authorized_user_file("/root/rag/youtube_oauth_token.json")
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open("/root/rag/youtube_oauth_token.json", "w") as f:
            f.write(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def get_playlist_videos(yt, playlist_id, playlist_name):
    """Fetch all video URLs from a playlist."""
    urls = []
    next_page = None
    while True:
        try:
            req = yt.playlistItems().list(
                part="contentDetails,snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page,
            )
            resp = req.execute()
        except Exception as e:
            logger.error(f"Error fetching playlist {playlist_name}: {e}")
            break
        for item in resp.get("items", []):
            vid_id = item["contentDetails"]["videoId"]
            title = item["snippet"]["title"][:80]
            url = f"https://www.youtube.com/watch?v={vid_id}"
            urls.append((url, title))
        next_page = resp.get("nextPageToken")
        if not next_page:
            break
    return urls


def ingest_video(url, title, ingestor, embedder, qdrant, postgres):
    """Ingest a single video through the full pipeline."""
    t0 = time.time()

    # Ingest (transcript or Whisper)
    doc, chunks = ingestor.ingest(url)

    # Embed + store vectors
    for chunk in chunks:
        emb = embedder.embed(chunk.content)
        qdrant.upsert(
            chunk=chunk,
            dense_vector=emb.dense,
            sparse_indices=emb.sparse_indices,
            sparse_values=emb.sparse_values,
        )

    # PostgreSQL
    postgres.save_document(doc)
    postgres.update_document_counts(doc.id, len(chunks), 0)

    # NER + Knowledge Graph
    ner = EntityExtractor()
    graph = GraphBuilder()
    all_entities = []
    for chunk in chunks:
        entities = ner.extract(chunk.content, document_id=doc.id, chunk_id=chunk.id)
        all_entities.extend(entities)
    graph.process_document(doc, all_entities)
    graph.close()
    postgres.update_document_counts(doc.id, len(chunks), len(all_entities))

    elapsed = time.time() - t0
    return len(chunks), len(all_entities), elapsed


def main():
    # Initialize shared resources
    ingestor = YouTubeIngestor()
    embedder = Embedder()
    qdrant = QdrantStore()
    qdrant.ensure_collection()
    postgres = PostgresStore()
    yt = get_youtube_client()

    total_videos = 0
    total_chunks = 0
    total_entities = 0
    skipped = 0
    failed = 0
    failed_list = []
    t_start = time.time()

    # Collect all videos from all playlists
    all_sources = []
    for playlist_id, playlist_name in PLAYLISTS.items():
        videos = get_playlist_videos(yt, playlist_id, playlist_name)
        print(f"  Playlist '{playlist_name}': {len(videos)} videos")
        for url, title in videos:
            all_sources.append((url, title, playlist_name))

    # Liked Videos
    if INCLUDE_LIKED:
        liked = get_playlist_videos(yt, "LL", "Liked Videos")
        print(f"  Liked Videos: {len(liked)} videos")
        for url, title in liked:
            all_sources.append((url, title, "Liked"))

    # Deduplicate by URL (in case same video is in multiple playlists)
    seen_urls = set()
    unique_sources = []
    for url, title, source in all_sources:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append((url, title, source))

    print(f"\n{'='*60}")
    print(f"Total unique videos: {len(unique_sources)}")
    print(f"{'='*60}\n")

    for i, (url, title, source) in enumerate(unique_sources, 1):
        # Dedup check against DB
        if is_already_ingested(url):
            print(f"  [{i:>3}/{len(unique_sources)}] SKIP  [{source:>10}] {title[:55]}")
            skipped += 1
            continue

        print(f"  [{i:>3}/{len(unique_sources)}] INGEST [{source:>10}] {title[:55]}")
        try:
            n_chunks, n_entities, elapsed = ingest_video(
                url, title, ingestor, embedder, qdrant, postgres
            )
            total_videos += 1
            total_chunks += n_chunks
            total_entities += n_entities
            print(f"       OK: {n_chunks} chunks, {n_entities} entities ({elapsed:.0f}s)")
        except Exception as e:
            failed += 1
            failed_list.append((title, str(e)[:100]))
            print(f"       FAILED: {e}")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Ingested:  {total_videos} videos")
    print(f"  Skipped:   {skipped} (already in DB)")
    print(f"  Failed:    {failed}")
    print(f"  Chunks:    {total_chunks}")
    print(f"  Entities:  {total_entities}")
    print(f"  Total time: {total_time/60:.1f} min")
    if failed_list:
        print(f"\nFailed videos:")
        for title, err in failed_list:
            print(f"  - {title}: {err}")


if __name__ == "__main__":
    main()
