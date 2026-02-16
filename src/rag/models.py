from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class Platform(str, Enum):
    YOUTUBE = "youtube"
    TWITTER = "twitter"
    REDDIT = "reddit"
    WEB = "web"
    PDF = "pdf"


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_url: str | None = None
    platform: Platform
    author: str | None = None
    language: str | None = None
    created_at: datetime | None = None
    ingested_at: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    parent_chunk_id: str | None = None
    metadata: dict = Field(default_factory=dict)


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    entity_type: str
    source_document_id: str
    source_chunk_id: str | None = None
    confidence: float = 1.0
    metadata: dict = Field(default_factory=dict)


class Topic(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    hierarchy_path: str | None = None
    bertopic_id: int | None = None
    metadata: dict = Field(default_factory=dict)
