"""Assign documents to collections based on YouTube playlists."""
import json
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from rag.storage.postgres import PostgresStore

# Playlist ID -> (Collection name, color)
PLAYLISTS = {
    "PLU1o0Hb7N8SZkFuIXTbUkeeYtSd4Wuga4": ("Claude", "#F97316"),
    "PLU1o0Hb7N8SbLng1qvAJ0TKiGVbZL88_Z": ("AI", "#8B5CF6"),
    "PLU1o0Hb7N8SZ3U9FZTRnDXbXSVqb9-B8N": ("Investing", "#10B981"),
    "PLU1o0Hb7N8SaTEUezlh3qN3KcWMAtBuPz": ("Bitcoin", "#F59E0B"),
    "PLU1o0Hb7N8SbVDU1kRcJWVcI2I-KGVyda": ("Trading", "#EF4444"),
    "PLU1o0Hb7N8Sa-aYosYWoso_sdwiGGJmkQ": ("Politik", "#6366F1"),
    "PLU1o0Hb7N8Sai672JKg3HGVUGuOxeQMsy": ("Schießen", "#78716C"),
    "PLU1o0Hb7N8SbcDUa9lH6YhxRAMcdoQoXL": ("Vatsim", "#0EA5E9"),
    "PLU1o0Hb7N8SZkELiVaDOHUA8g7Qqm7jOg": ("Mooney", "#14B8A6"),
    "PLU1o0Hb7N8SbHck6ImMWPO4CHnzNf00mO": ("Matthias", "#EC4899"),
    "PLU1o0Hb7N8SaJyeVjQQH9ERcCeWV8QIOr": ("IFR", "#3B82F6"),
    "PLU1o0Hb7N8SZ4VzbVrFb15feYnFNX2RQc": ("Ppl", "#A855F7"),
}

WEB_COLLECTION = ("Web-Artikel", "#6B7280")


def get_youtube_client():
    creds = Credentials.from_authorized_user_file("/root/rag/youtube_oauth_token.json")
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open("/root/rag/youtube_oauth_token.json", "w") as f:
            f.write(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def get_playlist_video_ids(yt, playlist_id):
    """Get all video IDs from a playlist."""
    video_ids = set()
    next_page = None
    while True:
        req = yt.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page,
        )
        resp = req.execute()
        for item in resp.get("items", []):
            video_ids.add(item["contentDetails"]["videoId"])
        next_page = resp.get("nextPageToken")
        if not next_page:
            break
    return video_ids


def main():
    pg = PostgresStore()
    yt = get_youtube_client()

    # Step 1: Create collections
    print("Creating collections...")
    collection_map = {}  # name -> collection_id

    for playlist_id, (name, color) in PLAYLISTS.items():
        coll = pg.create_collection(name, f"YouTube Playlist: {name}", color)
        collection_map[name] = str(coll["id"])
        print(f"  Created: {name} ({color})")

    # Web collection
    coll = pg.create_collection(WEB_COLLECTION[0], "Web-Artikel und Dokumentation", WEB_COLLECTION[1])
    collection_map[WEB_COLLECTION[0]] = str(coll["id"])
    print(f"  Created: {WEB_COLLECTION[0]}")

    # Step 2: Build video_id -> playlist mapping from YouTube API
    print("\nFetching playlist contents from YouTube API...")
    video_to_playlists = {}  # video_id -> list of playlist names

    for playlist_id, (name, color) in PLAYLISTS.items():
        video_ids = get_playlist_video_ids(yt, playlist_id)
        print(f"  {name}: {len(video_ids)} videos")
        for vid in video_ids:
            if vid not in video_to_playlists:
                video_to_playlists[vid] = []
            video_to_playlists[vid].append(name)

    # Step 3: Get all documents from DB
    print("\nAssigning documents to collections...")
    docs, total = pg.list_documents(limit=500)
    assigned = 0
    unassigned = []

    for doc in docs:
        doc_id = str(doc["id"])
        platform = doc.get("platform", "")
        source_url = doc.get("source_url", "") or ""

        if platform == "web":
            # All web docs go to Web-Artikel
            pg.add_document_to_collection(doc_id, collection_map[WEB_COLLECTION[0]])
            assigned += 1
            continue

        if platform == "youtube" and "v=" in source_url:
            # Extract video ID
            vid_id = source_url.split("v=")[1].split("&")[0]

            if vid_id in video_to_playlists:
                # Assign to all matching playlists
                for playlist_name in video_to_playlists[vid_id]:
                    pg.add_document_to_collection(doc_id, collection_map[playlist_name])
                assigned += 1
            else:
                # Not in any specific playlist (from Liked Videos only)
                unassigned.append((doc.get("title", "?")[:60], vid_id))
        else:
            unassigned.append((doc.get("title", "?")[:60], "no-url"))

    # Step 4: Handle unassigned (Liked-only videos)
    # Create a "Liked Videos" collection for those
    if unassigned:
        coll = pg.create_collection("Liked Videos", "YouTube Liked Videos (keine spezifische Playlist)", "#F43F5E")
        liked_id = str(coll["id"])
        print(f"\n  Created: Liked Videos for {len(unassigned)} unassigned docs")

        for doc in docs:
            doc_id = str(doc["id"])
            source_url = doc.get("source_url", "") or ""
            if doc.get("platform") == "youtube" and "v=" in source_url:
                vid_id = source_url.split("v=")[1].split("&")[0]
                if vid_id not in video_to_playlists:
                    pg.add_document_to_collection(doc_id, liked_id)
                    assigned += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"{'='*50}")
    print(f"  Collections created: {len(collection_map) + (1 if unassigned else 0)}")
    print(f"  Documents assigned:  {assigned}/{total}")
    if unassigned:
        print(f"  Liked-only videos:   {len(unassigned)}")

    # Print collection stats
    print(f"\nCollection overview:")
    for coll in pg.list_collections():
        print(f"  {coll['doc_count']:>4} docs | {coll['name']}")


if __name__ == "__main__":
    main()
