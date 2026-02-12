from time import time

import numpy as np
from fastapi import APIRouter, Request
from loguru import logger
from qdrant_client import QdrantClient
from src.app.v1.schema import (
    EmbeddingResponse,
    HealthCheckResponse,
    RankingResponse,
    SearchPoint,
    SearchRequest,
    SearchResponse,
)

v1 = APIRouter()


@v1.get("/health", response_model=HealthCheckResponse)
def health(request: Request) -> HealthCheckResponse:
    logger.debug(f"Methode: {request.method} on {request.url.path}")
    return {"version": request.app.state.config.VERSION, "timestamp": time()}


@v1.post("/health", response_model=HealthCheckResponse)
def health(request: Request) -> HealthCheckResponse:
    logger.debug(f"Methode: {request.method} on {request.url.path}")
    return {"version": request.app.state.config.VERSION, "timestamp": time()}


@v1.get("/embedding/")
def embedding(request: Request, text: str) -> EmbeddingResponse:
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    inputs = request.app.state.config.models["bi_tokenizer"](
        text, padding=True, truncation=True, return_tensors="np"
    )

    outputs = request.app.state.config.models["bi_encoder"](**inputs)
    mean_embedding = np.mean(outputs.last_hidden_state, axis=1).tolist()[0]
    return EmbeddingResponse(text=text, embedding=mean_embedding)


@v1.get("/ranking/")
def ranking(request: Request, question: str, text: str) -> RankingResponse:
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    inputs = request.app.state.config.models["cross_tokenizer"](
        [(question, text)], padding=True, truncation=True, return_tensors="np"
    )

    outputs = request.app.state.config.models["cross_encoder"](**inputs)

    score = outputs.logits.tolist()[0][0]

    return RankingResponse(question=question, text=text, score=score)


@v1.post("/search/")
def search(request: Request, body: SearchRequest) -> SearchResponse:
    embedding = body.embedding
    n_items = body.n_items
    table = body.table

    if n_items is None:
        n_items = request.app.state.config.kb_limit

    if table is None:
        table = request.app.state.config.kb_name

    # Your search logic using the embedding
    logger.debug(f"Methode: {request.method} on {request.url.path}")

    client = QdrantClient(
        host=request.app.state.config.kb_host, port=int(request.app.state.config.kb_port)
    )
    results = client.query_points(
        collection_name=table,
        query=embedding,
        limit=n_items,
    )

    response = []

    for point in results.points:
        temp = point.model_dump()
        temp.pop("id")
        response.append(SearchPoint(**temp, **temp["payload"]))

    return SearchResponse(results=response)
