from typing import Any, Dict, List, Optional, Union
import re
import uuid as uuid_lib

from pydantic import BaseModel, Field, field_validator, model_validator


#######################################################
### Internal Schema
#######################################################


class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, pattern=r'^[a-zA-Z0-9_-]+$')
    dimension: int = Field(..., gt=0, le=4096, description="Vector dimension (e.g., 384 for all-MiniLM-L6-v2)")
    distance_metric: str = Field(default="cosine", pattern=r'^(cosine|euclid|dot)$')


class DatapointCreate(BaseModel):
    id: Optional[Union[str, int]] = None
    text: str = Field(..., min_length=1, max_length=100000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(default=None, min_length=1, max_length=4096)

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: Optional[Union[str, int]]) -> Optional[Union[str, int]]:
        if v is None:
            return v
        if isinstance(v, int):
            if v < 0:
                raise ValueError('Integer ID must be non-negative')
            return v
        # String must be a valid UUID
        try:
            uuid_lib.UUID(str(v))
        except ValueError:
            raise ValueError(f"String ID must be a valid UUID, got: '{v}'")
        return v

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v

    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError('Embedding must contain only numeric values')
        return v


class DatapointUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @model_validator(mode='after')
    def validate_at_least_one_field(self) -> 'DatapointUpdate':
        if self.text is None and self.metadata is None:
            raise ValueError('At least one field (text or metadata) must be provided')
        return self


class QdrantPoint(BaseModel):
    id: Union[str, int]
    score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[Union[List[float], List[List[float]]]] = None


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


#######################################################
### Request Schema
#######################################################


class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    collection_name: Optional[str] = Field(default=None, min_length=1, max_length=255, pattern=r'^[a-zA-Z0-9_-]+$')
    n_retrieval: Optional[int] = Field(default=None, gt=0, le=1000)
    n_ranking: Optional[int] = Field(default=None, gt=0, le=1000)

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Question cannot be empty or only whitespace')
        return v


class RankingRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    texts: List[str] = Field(..., min_length=1, max_length=1000)

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Question cannot be empty or only whitespace')
        return v

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Texts list cannot be empty')
        for idx, text in enumerate(v):
            if not text.strip():
                raise ValueError(f'Text at index {idx} cannot be empty or only whitespace')
        return v


class SearchRequest(BaseModel):
    embedding: List[float] = Field(..., min_length=1, max_length=4096)
    n_retrieval: Optional[int] = Field(default=None, gt=0, le=1000)

    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('Embedding must contain only numeric values')
        return v


#######################################################
### Response Schema
#######################################################



class BulkInsertResponse(BaseModel):
    inserted_count: int


class CollectionListResponse(BaseModel):
    collections: List[str]


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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingResponse(BaseModel):
    text: str
    embedding: List[float]


class HealthCheckResponse(BaseModel):
    version: str
    timestamp: float


class QueryResponse(BaseModel):
    question: str
    results: List[SearchResult]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    context: List[str] = Field(..., min_length=1, max_length=100)
    prompt_key: Optional[str] = Field(default=None, min_length=1, max_length=100)
    prompt_language: Optional[str] = Field(default=None, min_length=2, max_length=10)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Question cannot be empty or only whitespace')
        return v

    @field_validator('context')
    @classmethod
    def validate_context(cls, v: List[str]) -> List[str]:
        for idx, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f'Context item at index {idx} must be a string')
        return v


class RagRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    collection_name: str = Field(..., min_length=1, max_length=255, pattern=r'^[a-zA-Z0-9_-]+$')
    n_retrieval: Optional[int] = Field(default=None, gt=0, le=1000)
    n_ranking: Optional[int] = Field(default=None, gt=0, le=1000)
    prompt_key: Optional[str] = Field(default=None, min_length=1, max_length=100)
    prompt_language: Optional[str] = Field(default=None, min_length=2, max_length=10)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)

    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Question cannot be empty or only whitespace')
        return v

    @field_validator('collection_name')
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Collection name must contain only alphanumeric characters, hyphens, and underscores')
        return v


class ChatResponse(BaseModel):
    answer: str


RankingResponse = QueryResponse


class SearchResponse(BaseModel):
    results: List[SearchResult]


class StatusResponse(BaseModel):
    id: Union[str, int]
    status: str
