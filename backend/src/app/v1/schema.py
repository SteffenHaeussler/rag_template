from typing import List, Optional

from pydantic import BaseModel

#######################################################
### Request Schema
#######################################################


class RankingRequest(BaseModel):
    query: str
    text: str


class EmbeddingRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    embedding: List[float]
    n_items: Optional[int] = None
    table: Optional[str] = None


#######################################################
### Response Schema
#######################################################


class HealthCheckResponse(BaseModel):
    version: str
    timestamp: float


class RankingResponse(BaseModel):
    question: str
    text: str
    score: float


class EmbeddingResponse(BaseModel):
    text: str
    embedding: List[float]


class SearchPoint(BaseModel):
    id: str
    score: float
    description: str
    name: str
    tag: str
    location: str
    area: str


class SearchResponse(BaseModel):
    results: List[SearchPoint]
