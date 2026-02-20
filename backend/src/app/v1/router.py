import uuid
from time import time
from typing import List, Union

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client import AsyncQdrantClient
from loguru import logger

from qdrant_client.models import Distance, PointStruct, VectorParams

import src.app.v1.schema as schema

from src.app.config import Config
from src.app.dependencies import (
    get_config,
    get_qdrant,
    get_retrieval_service,
    get_generation_service,
)
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


async def verify_collection(collection_name: str, qdrant: AsyncQdrantClient = Depends(get_qdrant)) -> str:
    """Verify that collection exists, raise 404 if not."""
    if not await qdrant.collection_exists(collection_name):
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection_name


# ==========================================
# Helper
# ==========================================


async def validate_embedding_dimension(embedding: List[float], collection_name: str, qdrant: AsyncQdrantClient) -> None:
    """
    Validate that embedding dimension matches collection configuration.

    Args:
        embedding: Embedding vector to validate
        collection_name: Name of the collection
        qdrant: Qdrant client instance

    Raises:
        HTTPException: If dimension doesn't match or collection uses named vectors
    """
    collection_info = await qdrant.get_collection(collection_name)
    vectors_config = collection_info.config.params.vectors

    if not hasattr(vectors_config, 'size'):
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{collection_name}' uses named vectors, which are not supported by this endpoint"
        )

    expected_dim = vectors_config.size
    actual_dim = len(embedding)

    if actual_dim != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding dimension mismatch: collection '{collection_name}' expects {expected_dim}D vectors, got {actual_dim}D"
        )


def parse_datapoint_id(datapoint_id: str) -> Union[str, int]:
    """Convert datapoint ID to int if numeric, validate UUID otherwise.

    Args:
        datapoint_id: The ID as a string

    Returns:
        Parsed ID as int or validated UUID string

    Raises:
        HTTPException: If ID format is invalid
    """
    if not datapoint_id or not datapoint_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Datapoint ID cannot be empty"
        )

    datapoint_id = datapoint_id.strip()

    # Try parsing as non-negative integer (.isdigit() guarantees >= 0)
    if datapoint_id.isdigit():
        return int(datapoint_id)

    # Try parsing as UUID
    try:
        uuid.UUID(datapoint_id)
        return datapoint_id
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid datapoint ID format: '{datapoint_id}'. Must be a positive integer or valid UUID."
        )


# ==========================================
# Health Checks
# ==========================================


@v1.get("/health", response_model=schema.HealthCheckResponse)
def health(config: Config = Depends(get_config)) -> schema.HealthCheckResponse:
    return {"version": config.VERSION, "timestamp": time()}


# ==========================================
# Collection Operations
# ==========================================


@v1.get("/collections/", response_model=schema.CollectionListResponse)
async def list_collections(qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    collections = await qdrant.get_collections()
    return schema.CollectionListResponse(
        collections=[c.name for c in collections.collections]
    )


@v1.post("/collections/", status_code=status.HTTP_201_CREATED, response_model=schema.CollectionResponse)
async def create_collection(collection: schema.CollectionCreate, qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    if await qdrant.collection_exists(collection.name):
        raise HTTPException(status_code=400, detail="Collection already exists")

    distance = DISTANCE_MAP.get(collection.distance_metric.lower())
    if not distance:
        raise HTTPException(status_code=400, detail=f"Unknown distance metric: {collection.distance_metric}")

    await qdrant.create_collection(
        collection_name=collection.name,
        vectors_config=VectorParams(size=collection.dimension, distance=distance),
    )

    return schema.CollectionResponse(name=collection.name, status="created", count=0)


@v1.delete("/collections/{collection_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(collection_name: str = Depends(verify_collection), qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    await qdrant.delete_collection(collection_name)


# ==========================================
# Datapoint Operations (Single & Bulk)
# ==========================================


@v1.post("/collections/{collection_name}/datapoints/", status_code=status.HTTP_201_CREATED, response_model=schema.StatusResponse)
async def insert_datapoint(
    datapoint: schema.DatapointCreate,
    collection_name: str = Depends(verify_collection),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Insert a single datapoint into collection."""
    point_id = datapoint.id or str(uuid.uuid4())

    embedding = datapoint.embedding or await retrieval_service._embed_text(datapoint.text)

    # Validate embedding dimension
    await validate_embedding_dimension(embedding, collection_name, qdrant)

    await qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={**datapoint.metadata, "text": datapoint.text},
            )
        ],
    )

    return schema.StatusResponse(id=point_id, status="inserted")


@v1.post("/collections/{collection_name}/datapoints/bulk", status_code=status.HTTP_201_CREATED, response_model=schema.BulkInsertResponse)
async def insert_bulk_datapoints(
    datapoints: List[schema.DatapointCreate],
    collection_name: str = Depends(verify_collection),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    config: Config = Depends(get_config),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Insert multiple datapoints in bulk with batch embedding for 10-50x speedup."""
    # Fetch collection dimension once (avoids N get_collection calls)
    collection_info = await qdrant.get_collection(collection_name)
    vectors_config = collection_info.config.params.vectors
    if not hasattr(vectors_config, 'size'):
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{collection_name}' uses named vectors, which are not supported by this endpoint"
        )
    expected_dim = vectors_config.size

    # Separate datapoints that need embedding from those with pre-computed embeddings
    texts_to_embed = []
    indices_to_embed = []
    embeddings = [None] * len(datapoints)

    for idx, dp in enumerate(datapoints):
        if dp.embedding:
            # Already has embedding, validate inline
            if len(dp.embedding) != expected_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Embedding dimension mismatch: collection '{collection_name}' expects {expected_dim}D vectors, got {len(dp.embedding)}D"
                )
            embeddings[idx] = dp.embedding
        else:
            # Needs embedding
            texts_to_embed.append(dp.text)
            indices_to_embed.append(idx)

    # Batch embed all texts that need embedding (10-50x faster!)
    if texts_to_embed:
        batch_embeddings = await retrieval_service._embed_texts_batch(texts_to_embed)

        # Validate dimensions and assign embeddings
        for batch_idx, dp_idx in enumerate(indices_to_embed):
            embedding = batch_embeddings[batch_idx]
            if len(embedding) != expected_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Embedding dimension mismatch: collection '{collection_name}' expects {expected_dim}D vectors, got {len(embedding)}D"
                )
            embeddings[dp_idx] = embedding

    # Build points list
    points = []
    for idx, dp in enumerate(datapoints):
        point_id = dp.id or str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embeddings[idx],
                payload={**dp.metadata, "text": dp.text},
            )
        )

    # Batch upsert to Qdrant
    batch_size = config.kb_batch_size
    for i in range(0, len(points), batch_size):
        await qdrant.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
        )

    logger.info(f"Successfully inserted {len(points)} datapoints using batch embedding")
    return schema.BulkInsertResponse(inserted_count=len(points))


# ==========================================
# Datapoint Retrieval
# ==========================================


@v1.get("/collections/{collection_name}/datapoints/{datapoint_id}", response_model=schema.DatapointResponse)
async def get_datapoint(datapoint_id: Union[str, int], collection_name: str = Depends(verify_collection), qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    datapoint_id = parse_datapoint_id(datapoint_id)
    results = await qdrant.retrieve(
        collection_name=collection_name,
        ids=[datapoint_id],
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        raise HTTPException(status_code=404, detail="Datapoint not found")

    point = results[0]
    payload = point.payload or {}
    text = payload.get("text", "")
    metadata = {k: v for k, v in payload.items() if k != "text"}
    return schema.DatapointResponse(id=str(point.id), text=text, metadata=metadata)


@v1.get("/collections/{collection_name}/datapoints/{datapoint_id}/embedding", response_model=schema.DatapointEmbeddingResponse)
async def get_datapoint_embedding(datapoint_id: str, collection_name: str = Depends(verify_collection), qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    datapoint_id = parse_datapoint_id(datapoint_id)

    results = await qdrant.retrieve(
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
async def update_datapoint(
    datapoint_id: str,
    update_data: schema.DatapointUpdate,
    collection_name: str = Depends(verify_collection),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Update an existing datapoint."""
    datapoint_id = parse_datapoint_id(datapoint_id)

    # Check point exists
    results = await qdrant.retrieve(
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
        vector = await retrieval_service._embed_text(update_data.text)
        await validate_embedding_dimension(vector, collection_name, qdrant)

    # Merge metadata
    if update_data.metadata is not None:
        payload.update(update_data.metadata)

    await qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=datapoint_id, vector=vector, payload=payload)],
    )

    return schema.StatusResponse(id=datapoint_id, status="updated")


@v1.delete("/collections/{collection_name}/datapoints/{datapoint_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_datapoint(datapoint_id: str, collection_name: str = Depends(verify_collection), qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    datapoint_id = parse_datapoint_id(datapoint_id)

    await qdrant.delete(
        collection_name=collection_name,
        points_selector=[datapoint_id],
    )

# ==========================================
# Embedding, Ranking & Search
# ==========================================


@v1.post("/embedding/", response_model=schema.EmbeddingResponse)
async def embedding(body: schema.EmbeddingRequest, retrieval_service: RetrievalService = Depends(get_retrieval_service)) -> schema.EmbeddingResponse:
    """Generate embedding for text."""
    mean_embedding = await retrieval_service._embed_text(body.text)
    return schema.EmbeddingResponse(text=body.text, embedding=mean_embedding)


@v1.post("/ranking/")
async def ranking(body: schema.RankingRequest, retrieval_service: RetrievalService = Depends(get_retrieval_service)) -> schema.RankingResponse:
    """Rerank texts by relevance to question."""
    candidates = [{"text": text} for text in body.texts]
    ranked_results = await retrieval_service.rerank(
        question=body.question,
        candidates=candidates,
        top_k=len(body.texts),
    )

    return schema.RankingResponse(question=body.question, results=ranked_results)


@v1.post("/collections/{collection_name}/search/", response_model=schema.SearchResponse)
async def search(
    collection_name: str,
    body: schema.SearchRequest,
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    config: Config = Depends(get_config),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> schema.SearchResponse:
    """Search collection using embedding vector."""
    n_items = body.n_retrieval or config.kb_limit

    # Validate embedding dimension
    await validate_embedding_dimension(body.embedding, collection_name, qdrant)

    points = await retrieval_service.search(
            collection_name=collection_name,
            query_vector=body.embedding,
            limit=n_items
        )

    response_data = []
    for point in points:
        payload = point.payload or {}
        text = payload.get("text", "")
        metadata = {"id": point.id, **{k: v for k, v in payload.items() if k != "text"}}
        response_data.append(
            schema.SearchResult(text=text, score=point.score, metadata=metadata)
        )

    return schema.SearchResponse(results=response_data)


@v1.post("/query/", response_model=schema.QueryResponse)
async def full_rag_pipeline(body: schema.QueryRequest, retrieval_service: RetrievalService = Depends(get_retrieval_service)) -> schema.QueryResponse:
    """Retrieve context: embed query, search, and rerank."""
    final_results = await retrieval_service.retrieve_context(
        question=body.question,
        collection_name=body.collection_name,
        n_retrieval=body.n_retrieval,
        n_ranking=body.n_ranking
    )

    return schema.QueryResponse(question=body.question, results=final_results)


@v1.post("/chat/", response_model=schema.ChatResponse)
async def chat(body: schema.ChatRequest, generation_service: GenerationService = Depends(get_generation_service)) -> schema.ChatResponse:
    """Generate answer from provided context (no retrieval)."""
    answer = await generation_service.generate_answer(
        question=body.question,
        context=body.context,
        prompt_key=body.prompt_key,
        prompt_language=body.prompt_language,
        temperature=body.temperature
    )

    return schema.ChatResponse(answer=answer)


@v1.post("/rag/", response_model=schema.ChatResponse)
async def rag(
    body: schema.RagRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
) -> schema.ChatResponse:
    """Full RAG pipeline: Retrieve context from collection + Generate answer."""
    # Retrieve context
    final_results = await retrieval_service.retrieve_context(
        question=body.question,
        collection_name=body.collection_name,
        n_retrieval=body.n_retrieval,
        n_ranking=body.n_ranking
    )

    context_texts = [result.text for result in final_results]

    # Generate Answer
    answer = await generation_service.generate_answer(
        question=body.question,
        context=context_texts,
        prompt_key=body.prompt_key,
        prompt_language=body.prompt_language,
        temperature=body.temperature
    )

    return schema.ChatResponse(answer=answer)
