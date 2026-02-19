"""Load and parse sources.yaml configuration."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WebSource:
    url: str
    enabled: bool = True


@dataclass
class YouTubeConfig:
    watch_later: bool = True
    playlists: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    channel_max_videos: int = 5


@dataclass
class RedditConfig:
    saved_posts: bool = True
    limit: int = 50


@dataclass
class TwitterConfig:
    bookmarks: bool = True
    limit: int = 50


@dataclass
class FolderSource:
    path: str
    extensions: list[str] | None = None
    exclude_patterns: list[str] | None = None
    max_file_size_mb: float = 100


@dataclass
class FoldersConfig:
    enabled: bool = False
    sources: list[FolderSource] = field(default_factory=list)


@dataclass
class SourcesConfig:
    cron: str = "0 6 * * *"
    web: list[WebSource] = field(default_factory=list)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    folders: FoldersConfig = field(default_factory=FoldersConfig)


def load_sources(path: str | Path | None = None) -> SourcesConfig:
    """Load sources from YAML file."""
    if path is None:
        path = Path(__file__).parents[3] / "sources.yaml"
    path = Path(path)

    if not path.exists():
        return SourcesConfig()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    config = SourcesConfig()

    # Schedule
    schedule = data.get("schedule", {})
    if schedule:
        config.cron = schedule.get("cron", config.cron)

    # Web sources
    for item in data.get("web", []) or []:
        if isinstance(item, dict) and item.get("enabled", True):
            config.web.append(WebSource(url=item["url"]))

    # YouTube
    yt = data.get("youtube", {}) or {}
    config.youtube = YouTubeConfig(
        watch_later=yt.get("watch_later", True),
        playlists=yt.get("playlists", []) or [],
        channels=yt.get("channels", []) or [],
        channel_max_videos=yt.get("channel_max_videos", 5),
    )

    # Reddit
    rd = data.get("reddit", {}) or {}
    config.reddit = RedditConfig(
        saved_posts=rd.get("saved_posts", True),
        limit=rd.get("limit", 50),
    )

    # Twitter
    tw = data.get("twitter", {}) or {}
    config.twitter = TwitterConfig(
        bookmarks=tw.get("bookmarks", True),
        limit=tw.get("limit", 50),
    )

    # Folders
    fd = data.get("folders", {}) or {}
    if fd:
        folder_sources = []
        for item in fd.get("sources", []) or []:
            if isinstance(item, dict):
                folder_sources.append(FolderSource(
                    path=item["path"],
                    extensions=item.get("extensions"),
                    exclude_patterns=item.get("exclude_patterns"),
                    max_file_size_mb=item.get("max_file_size_mb", 100),
                ))
        config.folders = FoldersConfig(
            enabled=fd.get("enabled", False),
            sources=folder_sources,
        )

    return config
