import uuid
from time import time
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from loguru import logger
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from src.app.v1.schema import (
    BulkInsertResponse,
    CollectionCreate,
    CollectionResponse,
    DatapointCreate,
    DatapointEmbeddingResponse,
    DatapointResponse,
    DatapointUpdate,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthCheckResponse,
    RankingResponse,
    QueryRequest,
    QueryResponse,
    SearchPoint,
    SearchRequest,
    SearchResponse,
    StatusResponse,
    SearchResult,
    RankingRequest,
    RankedItem,
)

v1 = APIRouter()


DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


@v1.get("/health", response_model=HealthCheckResponse)
def health_get(request: Request) -> HealthCheckResponse:
    logger.debug(f"Methode: {request.method} on {request.url.path}")
    return {"version": request.app.state.config.VERSION, "timestamp": time()}


@v1.post("/health", response_model=HealthCheckResponse)
def health_post(request: Request) -> HealthCheckResponse:
    logger.debug(f"Methode: {request.method} on {request.url.path}")
    return {"version": request.app.state.config.VERSION, "timestamp": time()}


# ==========================================
# 1. Collection Operations
# ==========================================


@v1.post("/collections/", status_code=status.HTTP_201_CREATED, response_model=CollectionResponse)
def create_collection(request: Request, collection: CollectionCreate):
    qdrant = request.app.state.qdrant
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    if qdrant.collection_exists(collection.name):
        raise HTTPException(status_code=400, detail="Collection already exists")

    distance = DISTANCE_MAP.get(collection.distance_metric.lower())
    if not distance:
        raise HTTPException(status_code=400, detail=f"Unknown distance metric: {collection.distance_metric}")

    qdrant.create_collection(
        collection_name=collection.name,
        vectors_config=VectorParams(size=collection.dimension, distance=distance),
    )

    return CollectionResponse(name=collection.name, status="created", count=0)


@v1.delete("/collections/{collection_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(request: Request, collection_name: str):
    qdrant = request.app.state.qdrant
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    if not qdrant.collection_exists(collection_name):
        raise HTTPException(status_code=404, detail="Collection not found")

    qdrant.delete_collection(collection_name)


# ==========================================
# 2. Datapoint Operations (Single & Bulk)
# ==========================================


def _embed_text(request: Request, text: str) -> list[float]:
    """Generate embedding for text using the bi-encoder."""
    config = request.app.state.config
    inputs = config.models["bi_tokenizer"](
        text, padding=True, truncation=True, return_tensors="np"
    )
    outputs = config.models["bi_encoder"](**inputs)
    return np.mean(outputs.last_hidden_state, axis=1).tolist()[0]


def _query_qdrant(
    request: Request,
    collection_name: str,
    vector: List[float],
    limit: int
) -> List[ScoredPoint]:
    """
    Generic wrapper for Qdrant querying.
    Returns the raw Qdrant points (ID, Score, Payload).
    """
    qdrant = request.app.state.qdrant

    results = qdrant.query_points(
        collection_name=collection_name,
        query=vector,
        limit=limit,
    )
    return results.points


def _rerank_candidates(
    request: Request,
    question: str,
    candidates: List[Dict[str, Any]],
    top_k: int
) -> List[SearchResult]:
    """Sorts candidates by relevance using the Cross-Encoder."""

    if not candidates:
        return []

    config = request.app.state.config
    tokenizer = config.models["cross_tokenizer"]
    model = config.models["cross_encoder"]

    # 1. Prepare pairs for the model: [ [Question, Text1], [Question, Text2], ... ]
    candidate_texts = [c.get("text", "") for c in candidates]
    pairs = [[question, text] for text in candidate_texts]

    # 2. Tokenize & Predict
    inputs = tokenizer(
        pairs, padding=True, truncation=True, return_tensors="np"
    )
    outputs = model(**inputs)

    # 3. Extract scores
    # Flatten logits to a 1D array
    scores = outputs.logits.reshape(-1).tolist()

    # 4. Zip, Sort, and Format
    ranked_results = []
    for score, content in zip(scores, candidates):
        ranked_results.append(
            SearchResult(
                text=content.get("text", ""),
                score=score,
                metadata=content # Pass full payload as metadata
            )
        )

    # Sort descending (Highest score first)
    ranked_results.sort(key=lambda x: x.score, reverse=True)

    # Return only the requested amount
    return ranked_results[:top_k]

@v1.post("/collections/{collection_name}/datapoints/", status_code=status.HTTP_201_CREATED, response_model=StatusResponse)
def insert_datapoint(request: Request, collection_name: str, datapoint: DatapointCreate):
    qdrant = request.app.state.qdrant
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    if not qdrant.collection_exists(collection_name):
        raise HTTPException(status_code=404, detail="Collection not found")

    point_id = datapoint.id or str(uuid.uuid4())
    embedding = datapoint.embedding or _embed_text(request, datapoint.text)

    qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"text": datapoint.text, **datapoint.metadata},
            )
        ],
    )

    return StatusResponse(id=point_id, status="inserted")


@v1.post("/collections/{collection_name}/datapoints/bulk", status_code=status.HTTP_201_CREATED, response_model=BulkInsertResponse)
def insert_bulk_datapoints(request: Request, collection_name: str, datapoints: List[DatapointCreate]):
    qdrant = request.app.state.qdrant
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    if not qdrant.collection_exists(collection_name):
        raise HTTPException(status_code=404, detail="Collection not found")

    points = []
    for dp in datapoints:
        point_id = dp.id or str(uuid.uuid4())
        embedding = dp.embedding or _embed_text(request, dp.text)
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"text": dp.text, **dp.metadata},
            )
        )

    # Batch upsert
    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
        )

    return BulkInsertResponse(inserted_count=len(points))


# ==========================================
# 3. Retrieval & Embedding
# ==========================================


@v1.get("/collections/{collection_name}/datapoints/{datapoint_id}", response_model=DatapointResponse)
def get_datapoint(request: Request, collection_name: str, datapoint_id: Union[str, int]):
    qdrant = request.app.state.qdrant
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    datapoint_id = int(datapoint_id) if datapoint_id.isdigit() else datapoint_id
    results = qdrant.retrieve(
        collection_name=collection_name,
        ids=[datapoint_id],
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        raise HTTPException(status_code=404, detail="Datapoint not found")

    point = results[0]
    payload = point.payload or {}
    text = payload.pop("text", "")
    return DatapointResponse(id=str(point.id), text=text, metadata=payload)


@v1.get("/collections/{collection_name}/datapoints/{datapoint_id}/embedding", response_model=DatapointEmbeddingResponse)
def get_datapoint_embedding(request: Request, collection_name: str, datapoint_id: str):
    qdrant = request.app.state.qdrant
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    datapoint_id = int(datapoint_id) if datapoint_id.isdigit() else datapoint_id

    results = qdrant.retrieve(
        collection_name=collection_name,
        ids=[datapoint_id],
        with_payload=False,
        with_vectors=True,
    )

    if not results:
        raise HTTPException(status_code=404, detail="Datapoint not found")

    return DatapointEmbeddingResponse(id=str(results[0].id), embedding=results[0].vector)


# ==========================================
# 4. Update & Delete
# ==========================================


@v1.put("/collections/{collection_name}/datapoints/{datapoint_id}", response_model=StatusResponse)
def update_datapoint(request: Request, collection_name: str, datapoint_id: str, update_data: DatapointUpdate):
    qdrant = request.app.state.qdrant
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    datapoint_id = int(datapoint_id) if datapoint_id.isdigit() else datapoint_id

    # Check point exists
    results = qdrant.retrieve(
        collection_name=collection_name,
        ids=[datapoint_id],
        with_payload=True,
        with_vectors=True,
    )

    if not results:
        raise HTTPException(status_code=404, detail="Datapoint not found")

    existing = results[0]
    payload = existing.payload or {}
    vector = existing.vector

    # Update text and re-embed if changed
    if update_data.text is not None:
        payload["text"] = update_data.text
        vector = _embed_text(request, update_data.text)

    # Merge metadata
    if update_data.metadata is not None:
        payload.update(update_data.metadata)

    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=datapoint_id, vector=vector, payload=payload)],
    )

    return StatusResponse(id=datapoint_id, status="updated")


@v1.delete("/collections/{collection_name}/datapoints/{datapoint_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_datapoint(request: Request, collection_name: str, datapoint_id: str):
    qdrant = request.app.state.qdrant
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    datapoint_id = int(datapoint_id) if datapoint_id.isdigit() else datapoint_id

    qdrant.delete(
        collection_name=collection_name,
        points_selector=[datapoint_id],
    )

    return StatusResponse(id=datapoint_id, status="deleted")

# ==========================================
# 5. Embedding & Ranking & Search
# ==========================================


@v1.post("/embedding/", response_model=EmbeddingResponse)
def embedding(request: Request, body: EmbeddingRequest) -> EmbeddingResponse:
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    mean_embedding = _embed_text(request, body.text)
    return EmbeddingResponse(text=body.text, embedding=mean_embedding)


@v1.post("/ranking/")
def ranking(request: Request, body: RankingRequest) -> RankingResponse:
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    tokenizer = request.app.state.config.models["cross_tokenizer"]
    model = request.app.state.config.models["cross_encoder"]

    pairs = [[body.question, text] for text in body.texts]

    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="np"
    )

    outputs = model(**inputs)
    raw_scores = outputs.logits.reshape(-1).tolist()

    ranked_results = []
    for text, score in zip(body.texts, raw_scores):
        ranked_results.append(RankedItem(text=text, score=score))

    # Sort descending (best match first)
    ranked_results.sort(key=lambda x: x.score, reverse=True)

    return RankingResponse(question=body.question, results=ranked_results)


@v1.post("/collections/{collection_name}/search/", response_model=SearchResponse)
def search(request: Request, collection_name: str, body: SearchRequest) -> SearchResponse:
    config = request.app.state.config
    qdrant = request.app.state.qdrant
    n_items = body.n_items or config.kb_limit

    logger.debug(f"Methode: {request.method} on {request.url.path}")

    points = _query_qdrant(
            request=request,
            collection_name=collection_name,
            vector=body.embedding,
            limit=n_items
        )

    # 2. Format for API Response
    response_data = []
    for point in points:
        # safely merge payload with top-level attributes (id, score)
        # point.payload might be None, so we default to {}
        payload = point.payload or {}
        response_data.append(
            SearchPoint(
                id=point.id,
                score=point.score,
                **payload  # Unpacks 'text', 'metadata', etc. from the DB payload
            )
        )

    return SearchResponse(results=response_data)


@v1.post("/query/", response_model=QueryResponse)
def full_rag_pipeline(request: Request, body: QueryRequest) -> QueryResponse:
    logger.debug(f"Pipeline started for: {body.question}")

    query_vector = _embed_text(request, body.question)

    table = body.table_name or request.app.state.config.kb_name

    candidates = _query_qdrant(
        request=request,
        vector=query_vector,
        collection_name=table,
        limit=body.n_retrieval
    )

    candidates_list = [point.payload or {} for point in candidates]

    final_results = _rerank_candidates(
        request=request,
        question=body.question,
        candidates=candidates_list,
        top_k=body.n_ranking
    )

    return QueryResponse(question=body.question, results=final_results)
