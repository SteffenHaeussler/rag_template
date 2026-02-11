from contextlib import asynccontextmanager

from fastapi import FastAPI
from google import genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from api import config
from api.routes import router

_app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _app_state["gemini"] = genai.Client(api_key=config.GEMINI_API_KEY)
    _app_state["qdrant"] = QdrantClient(
        host=config.QDRANT_HOST, port=config.QDRANT_PORT
    )
    _app_state["embedder"] = SentenceTransformer(config.EMBEDDING_MODEL)
    yield
    _app_state["qdrant"].close()
    _app_state.clear()


app = FastAPI(title="RAG Service", lifespan=lifespan)
app.include_router(router)
