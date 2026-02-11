from fastapi import APIRouter, HTTPException, Request

from api.rag import embed_query, generate_answer, retrieve
from api.schemas import HealthResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    config = request.app.state
    qdrant = config.qdrant
    embedder = config.models["bi_encoder"]

    try:
        query_vector = embed_query(embedder, req.question)
        sources = retrieve(qdrant, query_vector, req.top_k, config.kb_name)
        answer = generate_answer(req.question, sources, config.generation_model, config.llm_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(answer=answer, sources=sources)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    qdrant = request.app.state.qdrant

    try:
        qdrant.get_collections()
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "disconnected"

    return HealthResponse(status="ok", qdrant=qdrant_status)
