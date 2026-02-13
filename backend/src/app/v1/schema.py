from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


#######################################################
### Internal Schema
#######################################################


class CollectionCreate(BaseModel):
    name: str
    dimension: int = Field(..., description="Vector dimension (e.g., 384 for all-MiniLM-L6-v2)")
    distance_metric: str = "cosine"


class DatapointCreate(BaseModel):
    id: Optional[Union[str, int]] = None
    text: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None


class DatapointUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QdrantPoint(BaseModel):
    id: Union[str, int]
    score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[Union[List[float], List[List[float]]]] = None


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = {}


#######################################################
### Request Schema
#######################################################


class EmbeddingRequest(BaseModel):
    text: str


class QueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None
    n_retrieval: Optional[int] = None
    n_ranking: Optional[int] = None


class RankingRequest(BaseModel):
    question: str
    texts: List[str]


class SearchRequest(BaseModel):
    embedding: List[float]
    n_retrieval: Optional[int] = None


#######################################################
### Response Schema
#######################################################



class BulkInsertResponse(BaseModel):
    inserted_count: int


class CollectionResponse(BaseModel):
    name: str
    status: str
    count: int


class DatapointEmbeddingResponse(BaseModel):
    id: str
    embedding: List[float]


class DatapointResponse(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}


class EmbeddingResponse(BaseModel):
    text: str
    embedding: List[float]


class HealthCheckResponse(BaseModel):
    version: str
    timestamp: float


class QueryResponse(BaseModel):
    question: str
    results: List[SearchResult]


RankingResponse = QueryResponse


class SearchResponse(BaseModel):
    results: List[SearchResult]


class StatusResponse(BaseModel):
    id: Union[str, int]
    status: str
