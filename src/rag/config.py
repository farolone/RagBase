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

    # LLM (OpenAI-compatible API on Mac Studio Ultra)
    llm_base_url: str = "http://192.168.178.8:54321/v1"
    llm_model_rag: str = "mlx-community/MiniMax-M2.5-8bit"
    llm_model_agent: str = "mlx-community/MiniMax-M2.5-8bit"
    llm_model_reranker: str = ""
    llm_timeout: float = 120.0
    llm_max_retries: int = 3

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
