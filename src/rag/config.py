from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag"
    postgres_user: str = "rag"
    postgres_password: str = "changeme"

    # Ollama (Mac Studio Ultra im LAN)
    ollama_host: str = "192.168.178.y"
    ollama_port: int = 11434
    ollama_model_rag: str = "qwen2.5:72b-instruct-q8_0"
    ollama_model_agent: str = "minimax-m2.5:q4_K_M"
    ollama_model_reranker: str = "qwen3-reranker:8b"

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # Chunking
    chunk_size_leaf: int = 512
    chunk_size_parent: int = 1024
    chunk_size_grandparent: int = 2048
    chunk_overlap: int = 50

    model_config = {"env_file": ".env", "env_prefix": ""}


settings = Settings()
