"""Prefect flows for scheduled ingestion."""

import logging
from dataclasses import asdict
from datetime import datetime

from prefect import flow
from prefect.logging import get_run_logger

from rag.pipeline.sources import load_sources
from rag.pipeline.tasks import (
    ingest_web_url,
    ingest_youtube_video,
    fetch_youtube_watch_later,
    fetch_youtube_playlist,
    fetch_youtube_channel_uploads,
    fetch_reddit_saved,
    ingest_reddit_post,
    fetch_twitter_bookmarks,
    ingest_tweet,
    scan_and_ingest_folder,
)


@flow(name="daily-ingestion", log_prints=True)
def daily_ingestion(sources_path: str | None = None):
    """Main daily ingestion flow.

    Loads sources from YAML, fetches new items from personal collections
    (YouTube Watch Later, Reddit Saved, Twitter Bookmarks) and web URLs,
    then ingests everything with dedup.
    """
    config = load_sources(sources_path)
    results = {"web": [], "youtube": [], "reddit": [], "twitter": [], "folders": [], "errors": 0}

    # --- Web URLs ---
    print(f"\n=== Web Sources ({len(config.web)}) ===")
    for source in config.web:
        try:
            result = ingest_web_url(source.url)
            if result:
                results["web"].append(result)
        except Exception as e:
            print(f"ERROR web {source.url}: {e}")
            results["errors"] += 1

    # --- YouTube ---
    print("\n=== YouTube ===")
    yt_urls = []

    if config.youtube.watch_later:
        wl_urls = fetch_youtube_watch_later()
        yt_urls.extend(wl_urls or [])

    for playlist_id in config.youtube.playlists:
        pl_urls = fetch_youtube_playlist(playlist_id)
        yt_urls.extend(pl_urls or [])

    for channel_id in config.youtube.channels:
        ch_urls = fetch_youtube_channel_uploads(
            channel_id, max_results=config.youtube.channel_max_videos,
        )
        yt_urls.extend(ch_urls or [])

    for url in yt_urls:
        try:
            result = ingest_youtube_video(url)
            if result:
                results["youtube"].append(result)
        except Exception as e:
            print(f"ERROR youtube {url}: {e}")
            results["errors"] += 1

    # --- Reddit Saved ---
    print("\n=== Reddit Saved ===")
    if config.reddit.saved_posts:
        reddit_urls = fetch_reddit_saved(limit=config.reddit.limit)
        for url in reddit_urls or []:
            try:
                result = ingest_reddit_post(url)
                if result:
                    results["reddit"].append(result)
            except Exception as e:
                print(f"ERROR reddit {url}: {e}")
                results["errors"] += 1

    # --- Twitter Bookmarks ---
    print("\n=== Twitter Bookmarks ===")
    if config.twitter.bookmarks:
        tweet_urls = fetch_twitter_bookmarks(limit=config.twitter.limit)
        for url in tweet_urls or []:
            try:
                result = ingest_tweet(url)
                if result:
                    results["twitter"].append(result)
            except Exception as e:
                print(f"ERROR twitter {url}: {e}")
                results["errors"] += 1

    # --- Folders ---
    if config.folders.enabled:
        print(f"\n=== Folders ({len(config.folders.sources)}) ===")
        for folder_src in config.folders.sources:
            try:
                result = scan_and_ingest_folder(asdict(folder_src))
                results["folders"].append(result)
            except Exception as e:
                print(f"ERROR folder {folder_src.path}: {e}")
                results["errors"] += 1

    # --- Summary ---
    total = sum(len(v) for k, v in results.items() if k != "errors")
    print(f"\n=== Summary ({datetime.now().isoformat()}) ===")
    print(f"  Web:     {len(results['web'])} new")
    print(f"  YouTube: {len(results['youtube'])} new")
    print(f"  Reddit:  {len(results['reddit'])} new")
    print(f"  Twitter: {len(results['twitter'])} new")
    print(f"  Folders: {len(results['folders'])} scanned")
    print(f"  Total:   {total} ingested, {results['errors']} errors")

    return results
