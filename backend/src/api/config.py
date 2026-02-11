import os


GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME", "documents")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GENERATION_MODEL: str = os.environ.get("GENERATION_MODEL", "gemini-2.0-flash")
