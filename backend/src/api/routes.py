from fastapi import APIRouter, HTTPException

from api.rag import embed_query, generate_answer, retrieve
from api.schemas import HealthResponse, QueryRequest, QueryResponse

router = APIRouter()


def _get_clients():
    """Get clients from app state (set in lifespan)."""
    from api.main import _app_state

    return _app_state["gemini"], _app_state["qdrant"], _app_state["embedder"]


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    gemini, qdrant, embedder = _get_clients()

    try:
        query_vector = embed_query(embedder, req.question)
        sources = retrieve(qdrant, query_vector, req.top_k)
        answer = generate_answer(gemini, req.question, sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(answer=answer, sources=sources)


@router.get("/health", response_model=HealthResponse)
async def health():
    _, qdrant, _ = _get_clients()

    try:
        qdrant.get_collections()
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "disconnected"

    return HealthResponse(status="ok", qdrant=qdrant_status)
