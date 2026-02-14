import uuid
from time import time
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from qdrant_client import QdrantClient

from qdrant_client.models import Distance, PointStruct, VectorParams

import src.app.v1.schema as schema


from src.app.services.retrieval import RetrievalService
from src.app.services.generation import GenerationService

v1 = APIRouter()


DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


# ==========================================
# Dependencies
# ==========================================


def get_qdrant(request: Request) -> QdrantClient:
    return request.app.state.qdrant


def verify_collection(collection_name: str, qdrant: QdrantClient = Depends(get_qdrant)) -> str:
    if not qdrant.collection_exists(collection_name):
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection_name


# ==========================================
# Helper
# ==========================================


def parse_datapoint_id(datapoint_id: str) -> Union[str, int]:
    """Convert datapoint ID to int if numeric, validate UUID otherwise."""
    if datapoint_id.isdigit():
        return int(datapoint_id)
    try:
        uuid.UUID(datapoint_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid datapoint ID: {datapoint_id}. Must be an integer or UUID.")
    return datapoint_id


# ==========================================
# Health Checks
# ==========================================


@v1.api_route("/health", methods=["GET", "POST"], response_model=schema.HealthCheckResponse)
def health(request: Request) -> schema.HealthCheckResponse:
    return {"version": request.app.state.config.VERSION, "timestamp": time()}


# ==========================================
# Collection Operations
# ==========================================


@v1.post("/collections/", status_code=status.HTTP_201_CREATED, response_model=schema.CollectionResponse)
def create_collection(request: Request, collection: schema.CollectionCreate, qdrant: QdrantClient = Depends(get_qdrant)):
    if qdrant.collection_exists(collection.name):
        raise HTTPException(status_code=400, detail="Collection already exists")

    distance = DISTANCE_MAP.get(collection.distance_metric.lower())
    if not distance:
        raise HTTPException(status_code=400, detail=f"Unknown distance metric: {collection.distance_metric}")

    qdrant.create_collection(
        collection_name=collection.name,
        vectors_config=VectorParams(size=collection.dimension, distance=distance),
    )

    return schema.CollectionResponse(name=collection.name, status="created", count=0)


@v1.delete("/collections/{collection_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(request: Request, collection_name: str = Depends(verify_collection), qdrant: QdrantClient = Depends(get_qdrant)):
    qdrant.delete_collection(collection_name)


# ==========================================
# Datapoint Operations (Single & Bulk)
# ==========================================


@v1.post("/collections/{collection_name}/datapoints/", status_code=status.HTTP_201_CREATED, response_model=schema.StatusResponse)
def insert_datapoint(request: Request, datapoint: schema.DatapointCreate, collection_name: str = Depends(verify_collection), qdrant: QdrantClient = Depends(get_qdrant)):
    point_id = datapoint.id or str(uuid.uuid4())

    retrieval_service = RetrievalService(request)
    embedding = datapoint.embedding or retrieval_service._embed_text(datapoint.text)

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

    return schema.StatusResponse(id=point_id, status="inserted")


@v1.post("/collections/{collection_name}/datapoints/bulk", status_code=status.HTTP_201_CREATED, response_model=schema.BulkInsertResponse)
def insert_bulk_datapoints(request: Request, datapoints: List[schema.DatapointCreate], collection_name: str = Depends(verify_collection), qdrant: QdrantClient = Depends(get_qdrant)):
    retrieval_service = RetrievalService(request)
    points = []

    # Process in batches for embedding if needed (optimization possible here but kept simple for now)
    # Ideally should batch embed, but for now reuse single embed logic or Service method

    for dp in datapoints:
        point_id = dp.id or str(uuid.uuid4())
        embedding = dp.embedding or retrieval_service._embed_text(dp.text)
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"text": dp.text, **dp.metadata},
            )
        )

    # Batch upsert
    batch_size = request.app.state.config.kb_batch_size
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
        )

    return schema.BulkInsertResponse(inserted_count=len(points))


# ==========================================
# Datapoint Retrieval
# ==========================================


@v1.get("/collections/{collection_name}/datapoints/{datapoint_id}", response_model=schema.DatapointResponse)
def get_datapoint(request: Request, collection_name: str, datapoint_id: Union[str, int], qdrant: QdrantClient = Depends(get_qdrant)):
    datapoint_id = parse_datapoint_id(datapoint_id)
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
    return schema.DatapointResponse(id=str(point.id), text=text, metadata=payload)


@v1.get("/collections/{collection_name}/datapoints/{datapoint_id}/embedding", response_model=schema.DatapointEmbeddingResponse)
def get_datapoint_embedding(request: Request, collection_name: str, datapoint_id: str, qdrant: QdrantClient = Depends(get_qdrant)):
    datapoint_id = parse_datapoint_id(datapoint_id)

    results = qdrant.retrieve(
        collection_name=collection_name,
        ids=[datapoint_id],
        with_payload=False,
        with_vectors=True,
    )

    if not results:
        raise HTTPException(status_code=404, detail="Datapoint not found")

    return schema.DatapointEmbeddingResponse(id=str(results[0].id), embedding=results[0].vector)


# ==========================================
# Datapoint Update & Delete
# ==========================================


@v1.put("/collections/{collection_name}/datapoints/{datapoint_id}", response_model=schema.StatusResponse)
def update_datapoint(request: Request, collection_name: str, datapoint_id: str, update_data: schema.DatapointUpdate, qdrant: QdrantClient = Depends(get_qdrant)):
    datapoint_id = parse_datapoint_id(datapoint_id)
    retrieval_service = RetrievalService(request)

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
        vector = retrieval_service._embed_text(update_data.text)

    # Merge metadata
    if update_data.metadata is not None:
        payload.update(update_data.metadata)

    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=datapoint_id, vector=vector, payload=payload)],
    )

    return schema.StatusResponse(id=datapoint_id, status="updated")


@v1.delete("/collections/{collection_name}/datapoints/{datapoint_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_datapoint(request: Request, collection_name: str, datapoint_id: str, qdrant: QdrantClient = Depends(get_qdrant)):
    datapoint_id = parse_datapoint_id(datapoint_id)

    qdrant.delete(
        collection_name=collection_name,
        points_selector=[datapoint_id],
    )

# ==========================================
# Embedding, Ranking & Search
# ==========================================


@v1.post("/embedding/", response_model=schema.EmbeddingResponse)
def embedding(request: Request, body: schema.EmbeddingRequest) -> schema.EmbeddingResponse:
    retrieval_service = RetrievalService(request)
    mean_embedding = retrieval_service._embed_text(body.text)
    return schema.EmbeddingResponse(text=body.text, embedding=mean_embedding)


@v1.post("/ranking/")
def ranking(request: Request, body: schema.RankingRequest) -> schema.RankingResponse:
    retrieval_service = RetrievalService(request)
    candidates = [{"text": text} for text in body.texts]
    ranked_results = retrieval_service.rerank(
        question=body.question,
        candidates=candidates,
        top_k=len(body.texts),
    )

    return schema.RankingResponse(question=body.question, results=ranked_results)


@v1.post("/collections/{collection_name}/search/", response_model=schema.SearchResponse)
def search(request: Request, collection_name: str, body: schema.SearchRequest, qdrant: QdrantClient = Depends(get_qdrant)) -> schema.SearchResponse:
    config = request.app.state.config
    n_items = body.n_retrieval or config.kb_limit

    retrieval_service = RetrievalService(request)

    points = retrieval_service.search(
            collection_name=collection_name,
            query_vector=body.embedding,
            limit=n_items
        )

    response_data = []
    for point in points:
        payload = point.payload or {}
        text = payload.pop("text", "")
        metadata = {"id": point.id, **payload}
        response_data.append(
            schema.SearchResult(text=text, score=point.score, metadata=metadata)
        )

    return schema.SearchResponse(results=response_data)


@v1.post("/query/", response_model=schema.QueryResponse)
def full_rag_pipeline(request: Request, body: schema.QueryRequest, qdrant: QdrantClient = Depends(get_qdrant)) -> schema.QueryResponse:
    retrieval_service = RetrievalService(request)

    final_results = retrieval_service.retrieve_context(
        question=body.question,
        collection_name=body.collection_name,
        n_retrieval=body.n_retrieval,
        n_ranking=body.n_ranking
    )

    return schema.QueryResponse(question=body.question, results=final_results)


@v1.post("/chat/", response_model=schema.ChatResponse)
def chat(request: Request, body: schema.ChatRequest, qdrant: QdrantClient = Depends(get_qdrant)) -> schema.ChatResponse:
    retrieval_service = RetrievalService(request)

    # Retrieve context
    final_results = retrieval_service.retrieve_context(
        question=body.question
    )

    context_texts = [result.text for result in final_results]

    # Generate Answer
    generation_service = GenerationService(request)
    answer = generation_service.generate_answer(body.question, context_texts)

    return schema.ChatResponse(answer=answer)
