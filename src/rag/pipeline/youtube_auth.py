"""YouTube OAuth 2.0 authentication helper.

Run this once to get a refresh token for Watch Later access:

    python -m rag.pipeline.youtube_auth

This opens a browser for Google login. The resulting token is saved
to youtube_oauth_token.json and reused by the pipeline.
"""

import json
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

# YouTube read-only scope
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
TOKEN_PATH = Path(__file__).parents[3] / "youtube_oauth_token.json"
CLIENT_CONFIG_PATH = Path(__file__).parents[3] / "youtube_client_secret.json"


def authenticate():
    """Run OAuth flow and save token."""
    if not CLIENT_CONFIG_PATH.exists():
        print(f"ERROR: {CLIENT_CONFIG_PATH} not found")
        print("Place your Google OAuth client_secret JSON file there.")
        return

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_CONFIG_PATH), SCOPES)

    # Try local server first (works if browser available), fall back to console
    try:
        creds = flow.run_local_server(port=8090, open_browser=False)
        print(f"\nOpen this URL in your browser to authorize:\n")
        print(f"  http://192.168.178.182:8090  (if on LXC)")
        print(f"\nOr use the URL shown above.\n")
    except Exception:
        creds = flow.run_console()

    # Save token
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes),
    }
    with open(TOKEN_PATH, "w") as f:
        json.dump(token_data, f, indent=2)

    print(f"Token saved to {TOKEN_PATH}")

    # Quick test
    from googleapiclient.discovery import build
    youtube = build("youtube", "v3", credentials=creds)
    resp = youtube.playlists().list(part="snippet", mine=True, maxResults=5).execute()
    print(f"Authenticated! Found {resp.get('pageInfo', {}).get('totalResults', 0)} playlists.")


if __name__ == "__main__":
    authenticate()
