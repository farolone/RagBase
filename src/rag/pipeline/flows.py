"""Prefect flows for scheduled ingestion."""

import logging
from datetime import datetime

from prefect import flow
from prefect.logging import get_run_logger

from rag.pipeline.sources import load_sources
from rag.pipeline.tasks import (
    ingest_web_url,
    ingest_youtube_video,
    ingest_web_url_to_collection,
    ingest_youtube_video_to_collection,
    fetch_youtube_watch_later,
    fetch_youtube_playlist,
    fetch_youtube_channel_uploads,
    fetch_reddit_saved,
    ingest_reddit_post,
    fetch_twitter_bookmarks,
    ingest_tweet,
)


@flow(name="daily-ingestion", log_prints=True)
def daily_ingestion(sources_path: str | None = None):
    """Main daily ingestion flow.

    Loads sources from YAML, fetches new items from personal collections
    (YouTube Watch Later, Reddit Saved, Twitter Bookmarks) and web URLs,
    then ingests everything with dedup.
    """
    config = load_sources(sources_path)
    results = {"web": [], "youtube": [], "reddit": [], "twitter": [], "errors": 0}

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

    # --- Summary ---
    total = sum(len(v) for k, v in results.items() if k != "errors")
    print(f"\n=== Summary ({datetime.now().isoformat()}) ===")
    print(f"  Web:     {len(results['web'])} new")
    print(f"  YouTube: {len(results['youtube'])} new")
    print(f"  Reddit:  {len(results['reddit'])} new")
    print(f"  Twitter: {len(results['twitter'])} new")
    print(f"  Total:   {total} ingested, {results['errors']} errors")

    return results


@flow(name="daily-ingestion-v2", log_prints=True)
def daily_ingestion_v2():
    """DB-driven ingestion flow with collection assignment.

    Reads source_configs from PostgreSQL, processes each enabled source,
    and assigns ingested documents to their configured collection.
    """
    from rag.storage.postgres import PostgresStore
    from rag.pipeline.dedup import is_already_ingested

    pg = PostgresStore()
    sources = pg.list_source_configs()
    results = {"ingested": 0, "skipped": 0, "errors": 0}

    for source in sources:
        if not source["enabled"]:
            continue

        config = source["config"] or {}
        collection_id = str(source["collection_id"]) if source["collection_id"] else None
        source_id = str(source["id"])

        try:
            match source["source_type"]:
                case "web_url":
                    url = config.get("url", "")
                    if url:
                        r = ingest_web_url_to_collection(url, collection_id)
                        if r:
                            results["ingested"] += 1
                        else:
                            results["skipped"] += 1

                case "youtube_watch_later":
                    urls = fetch_youtube_watch_later()
                    for url in urls or []:
                        try:
                            r = ingest_youtube_video_to_collection(url, collection_id)
                            if r:
                                results["ingested"] += 1
                            else:
                                results["skipped"] += 1
                        except Exception as e:
                            print(f"ERROR youtube {url}: {e}")
                            results["errors"] += 1

                case "youtube_playlist":
                    playlist_id = config.get("playlist_id", "")
                    if playlist_id:
                        urls = fetch_youtube_playlist(playlist_id)
                        for url in urls or []:
                            try:
                                r = ingest_youtube_video_to_collection(url, collection_id)
                                if r:
                                    results["ingested"] += 1
                                else:
                                    results["skipped"] += 1
                            except Exception as e:
                                print(f"ERROR youtube {url}: {e}")
                                results["errors"] += 1

                case "youtube_channel":
                    channel_id = config.get("channel_id", "")
                    max_videos = config.get("max_videos", 5)
                    if channel_id:
                        urls = fetch_youtube_channel_uploads(channel_id, max_results=max_videos)
                        for url in urls or []:
                            try:
                                r = ingest_youtube_video_to_collection(url, collection_id)
                                if r:
                                    results["ingested"] += 1
                                else:
                                    results["skipped"] += 1
                            except Exception as e:
                                print(f"ERROR youtube {url}: {e}")
                                results["errors"] += 1

                case "reddit_saved":
                    limit = config.get("limit", 50)
                    subreddit_filter = config.get("subreddit_filter", "")
                    reddit_urls = fetch_reddit_saved(limit=limit)
                    for url in reddit_urls or []:
                        # Filter by subreddit if specified
                        if subreddit_filter and f"/r/{subreddit_filter.lower()}/" not in url.lower():
                            continue
                        try:
                            result = ingest_reddit_post(url)
                            if result and collection_id:
                                pg.add_document_to_collection(result["doc_id"], collection_id)
                            if result:
                                results["ingested"] += 1
                            else:
                                results["skipped"] += 1
                        except Exception as e:
                            print(f"ERROR reddit {url}: {e}")
                            results["errors"] += 1

                case "twitter_bookmarks":
                    limit = config.get("limit", 50)
                    folder = config.get("folder", "")
                    keywords = config.get("keywords", [])
                    tweet_urls = fetch_twitter_bookmarks(limit=limit)
                    for url in tweet_urls or []:
                        try:
                            result = ingest_tweet(url)
                            if result and collection_id:
                                pg.add_document_to_collection(result["doc_id"], collection_id)
                            if result:
                                results["ingested"] += 1
                            else:
                                results["skipped"] += 1
                        except Exception as e:
                            print(f"ERROR twitter {url}: {e}")
                            results["errors"] += 1

                case "folder":
                    folder_path = config.get("path", "")
                    extensions = config.get("extensions", [".pdf"])
                    max_size_mb = config.get("max_file_size_mb", 50)
                    if folder_path:
                        from pathlib import Path
                        folder = Path(folder_path)
                        if folder.exists():
                            for ext in extensions:
                                for fpath in folder.glob(f"*{ext}"):
                                    if fpath.stat().st_size > max_size_mb * 1024 * 1024:
                                        continue
                                    url_str = str(fpath)
                                    if is_already_ingested(url_str):
                                        results["skipped"] += 1
                                        continue
                                    try:
                                        from rag.pipeline.tasks import _run_full_ingest_to_collection
                                        stype = "pdf" if ext == ".pdf" else "web"
                                        r = _run_full_ingest_to_collection(url_str, stype, collection_id)
                                        if r:
                                            results["ingested"] += 1
                                            print(f"Done folder file: {fpath.name}")
                                    except Exception as e:
                                        print(f"ERROR folder {fpath}: {e}")
                                        results["errors"] += 1

            pg.update_source_last_run(source_id)

        except Exception as e:
            print(f"ERROR processing source {source['name']}: {e}")
            results["errors"] += 1

    # Summary
    print(f"\n=== Summary ({datetime.now().isoformat()}) ===")
    print(f"  Ingested: {results['ingested']}")
    print(f"  Skipped:  {results['skipped']}")
    print(f"  Errors:   {results['errors']}")

    return results
