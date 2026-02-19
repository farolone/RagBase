#!/usr/bin/env python3
"""Backfill script: reconstruct transcripts from Qdrant chunks and set published_at.

Phase A: Reconstruct transcripts from existing Qdrant chunks (all YouTube docs, ~430, no network)
Phase B: Re-fetch fine-grained transcripts from YouTube API (optional, needs network)
Phase C: Backfill published_at from existing metadata (all docs)

Usage:
    python scripts/backfill_transcripts.py           # All phases
    python scripts/backfill_transcripts.py --phase a  # Only Phase A
    python scripts/backfill_transcripts.py --phase b  # Only Phase B
    python scripts/backfill_transcripts.py --phase c  # Only Phase C
"""

import argparse
import json
import logging
import sys

sys.path.insert(0, "/root/rag/src")

from rag.storage.postgres import PostgresStore
from rag.storage.qdrant import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def phase_a_reconstruct_from_chunks():
    """Reconstruct transcripts from existing Qdrant chunks (~60s windows)."""
    logger.info("=== Phase A: Reconstruct transcripts from Qdrant chunks ===")

    pg = PostgresStore()
    qdrant = QdrantStore()

    # Get all YouTube documents
    rows, total = pg.list_documents(platform="youtube", limit=1000)
    logger.info(f"Found {total} YouTube documents")

    count = 0
    skipped = 0
    for row in rows:
        doc_id = str(row["id"])

        # Check if transcript already exists
        existing = pg.get_transcript(doc_id)
        if existing:
            skipped += 1
            continue

        # Get chunks from Qdrant
        chunks = qdrant.get_chunks_for_document(doc_id)
        if not chunks:
            logger.warning(f"No chunks found for {doc_id} ({row.get('title', '?')})")
            continue

        # Reconstruct segments from chunks
        segments = []
        for chunk in chunks:
            start_time = chunk.get("metadata", {}).get("start_time", chunk.get("start_time"))
            if start_time is None:
                # Fallback: use chunk_index * 60 as estimate
                start_time = chunk["chunk_index"] * 60.0

            segments.append({
                "text": chunk["content"],
                "start": float(start_time),
                "end": None,  # Not available from chunks
            })

        pg.save_transcript(doc_id, segments, "chunks_reconstructed")
        count += 1

    logger.info(f"Phase A complete: {count} transcripts reconstructed, {skipped} already existed")
    return count


def phase_b_refetch_from_api():
    """Re-fetch fine-grained transcripts from YouTube API for videos that have subtitles."""
    logger.info("=== Phase B: Re-fetch transcripts from YouTube API ===")

    pg = PostgresStore()

    # Get all YouTube documents
    rows, total = pg.list_documents(platform="youtube", limit=1000)
    logger.info(f"Found {total} YouTube documents to check")

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        logger.error("youtube-transcript-api not installed, skipping Phase B")
        return 0

    ytt_api = YouTubeTranscriptApi()
    count = 0
    errors = 0

    for row in rows:
        doc_id = str(row["id"])
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        video_id = metadata.get("video_id")
        if not video_id:
            continue

        # Check if we already have a fine-grained transcript
        existing = pg.get_transcript(doc_id)
        if existing and existing["source"] in ("api", "whisper"):
            continue  # Already have fine-grained data

        try:
            transcript_list = ytt_api.fetch(video_id)
            segments = [
                {
                    "text": entry.text,
                    "start": entry.start,
                    "end": entry.start + entry.duration,
                }
                for entry in transcript_list
            ]
            pg.save_transcript(doc_id, segments, "api")
            count += 1
            if count % 50 == 0:
                logger.info(f"  ... {count} transcripts re-fetched so far")
        except Exception:
            errors += 1  # No subtitles available, keep reconstructed version

    logger.info(f"Phase B complete: {count} transcripts upgraded to API, {errors} without subtitles")
    return count


def phase_c_backfill_published_at():
    """Backfill published_at from existing metadata."""
    logger.info("=== Phase C: Backfill published_at from metadata ===")

    pg = PostgresStore()
    count = pg.backfill_published_at()
    logger.info(f"Phase C complete: {count} documents got published_at set")
    return count


def main():
    parser = argparse.ArgumentParser(description="Backfill transcripts and published_at")
    parser.add_argument("--phase", choices=["a", "b", "c"], help="Run only a specific phase")
    args = parser.parse_args()

    if args.phase is None or args.phase == "a":
        phase_a_reconstruct_from_chunks()
    if args.phase is None or args.phase == "b":
        phase_b_refetch_from_api()
    if args.phase is None or args.phase == "c":
        phase_c_backfill_published_at()

    logger.info("Done!")


if __name__ == "__main__":
    main()
