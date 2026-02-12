from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

#######################################################
### Request Schema
#######################################################


class RankingRequest(BaseModel):
    question: str
    texts: List[str]

class RankedItem(BaseModel):
    text: str
    score: float

class RankingResponse(BaseModel):
    question: str
    results: List[RankedItem]

class EmbeddingRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    embedding: List[float]
    n_items: Optional[int] = None


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
    text: Optional[Union[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


#######################################################
### Response Schema
#######################################################


class HealthCheckResponse(BaseModel):
    version: str
    timestamp: float



class EmbeddingResponse(BaseModel):
    text: str
    embedding: List[float]


class SearchPoint(BaseModel):
    id: Optional[Union[str, int]]
    score: float
    text: str
    category: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchPoint]


class CollectionResponse(BaseModel):
    name: str
    status: str
    count: int


class DatapointResponse(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}


class DatapointEmbeddingResponse(BaseModel):
    id: str
    embedding: List[float]


class BulkInsertResponse(BaseModel):
    inserted_count: int


class StatusResponse(BaseModel):
    id: Union[str, int]
    status: str


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = {}

class QueryRequest(BaseModel):
    question: str
    table_name: Optional[str] = None
    n_retrieval: Optional[int] = None
    n_ranking: Optional[int] = None

class QueryResponse(BaseModel):
    question: str
    results: List[SearchResult]
