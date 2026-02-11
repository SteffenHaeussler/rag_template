import os


QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "documents")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DATA_DIR: str = os.environ.get("DATA_DIR", "/data")
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "50"))
