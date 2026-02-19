from typing import List, Dict, Any, Optional
import math
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastapi import Request, HTTPException
from loguru import logger
import src.app.v1.schema as schema
from src.app.exceptions import EmbeddingError, RerankingError, VectorDBError
from src.app.retry import retry_with_backoff, is_transient_error

class RetrievalService:
    def __init__(self, request: Request):
        self.request = request
        self.config = request.app.state.config
        self.models = request.app.state.models
        self.qdrant: QdrantClient = request.app.state.qdrant

    def _embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using the bi-encoder.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            tokenizer = self.models["bi_tokenizer"]
            model = self.models["bi_encoder"]

            inputs = tokenizer(
                text, padding=True, truncation=True, return_tensors="np", max_length=512
            )

            outputs = model(**inputs)

            if not hasattr(outputs, "last_hidden_state"):
                raise EmbeddingError("Model output missing 'last_hidden_state'")

            embedding = np.mean(outputs.last_hidden_state, axis=1).tolist()[0]

            if not embedding or len(embedding) == 0:
                raise EmbeddingError("Generated embedding is empty")

            return embedding

        except KeyError as e:
            logger.error(f"Model or tokenizer not found in config: {e}")
            raise EmbeddingError(f"Model configuration error: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Embedding generation failed for text (length={len(text)}): {e}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}", original_error=e)

    def _embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch (10-50x faster than one-by-one).

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors, one per input text

        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        if not texts:
            return []

        # Filter out empty texts but keep track of indices
        valid_indices = []
        valid_texts = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(idx)
                valid_texts.append(text)

        if not valid_texts:
            raise EmbeddingError("Cannot embed batch: all texts are empty")

        try:
            tokenizer = self.models["bi_tokenizer"]
            model = self.models["bi_encoder"]

            # Batch tokenization - much faster than one-by-one
            inputs = tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                return_tensors="np",
                max_length=512
            )

            # Batch inference - this is where we get the big speedup
            outputs = model(**inputs)

            if not hasattr(outputs, "last_hidden_state"):
                raise EmbeddingError("Model output missing 'last_hidden_state'")

            # Mean pooling across sequence dimension (axis=1)
            # Shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
            embeddings = np.mean(outputs.last_hidden_state, axis=1).tolist()

            if len(embeddings) != len(valid_texts):
                raise EmbeddingError(
                    f"Embedding count mismatch: expected {len(valid_texts)}, got {len(embeddings)}"
                )

            # Reconstruct full list preserving original indices
            result = [None] * len(texts)
            for idx, embedding in zip(valid_indices, embeddings):
                if not embedding or len(embedding) == 0:
                    raise EmbeddingError(f"Generated embedding is empty for text at index {idx}")
                result[idx] = embedding

            # Raise for any skipped empty-text positions
            for idx, emb in enumerate(result):
                if emb is None:
                    raise EmbeddingError(f"Text at index {idx} was empty or whitespace-only")

            return result

        except KeyError as e:
            logger.error(f"Model or tokenizer not found: {e}")
            raise EmbeddingError(f"Model configuration error: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Batch embedding generation failed for {len(texts)} texts: {e}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {str(e)}", original_error=e)

    @retry_with_backoff(max_retries=2, initial_delay=0.5, exceptions=(Exception,), retryable=is_transient_error)
    def _search_qdrant(self, collection_name: str, query_vector: List[float], limit: int) -> List[PointStruct]:
        """
        Raw Qdrant query with retry logic for transient failures.

        Raw exceptions propagate so the decorator's retryable check sees the
        original exception, not a wrapped VectorDBError.

        Args:
            collection_name: Name of the collection to search
            query_vector: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of matching points
        """
        results = self.qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        )
        return results.points

    def search(self, collection_name: str, query_vector: List[float], limit: int) -> List[PointStruct]:
        """
        Search collection with retry logic and structured error handling.

        Args:
            collection_name: Name of the collection to search
            query_vector: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of matching points

        Raises:
            VectorDBError: If search operation fails after retries
        """
        try:
            return self._search_qdrant(collection_name, query_vector, limit)
        except Exception as e:
            logger.error(f"Qdrant search failed for collection '{collection_name}': {e}")
            raise VectorDBError(
                f"Vector search failed in collection '{collection_name}'",
                original_error=e
            )

    def rerank(self, question: str, candidates: List[Dict[str, Any]], top_k: int) -> List[schema.SearchResult]:
        """
        Sorts candidates by relevance using the Cross-Encoder.

        Args:
            question: Query question
            candidates: List of candidate documents
            top_k: Number of top results to return

        Returns:
            Ranked list of search results

        Raises:
            RerankingError: If reranking operation fails
        """
        if not candidates:
            return []

        if not question or not question.strip():
            logger.warning("Empty question provided for reranking, returning unranked results")
            return [
                schema.SearchResult(
                    text=c.get("text", ""),
                    score=0.0,
                    metadata={k: v for k, v in c.items() if k != "text"}
                )
                for c in candidates[:top_k]
            ]

        try:
            tokenizer = self.models["cross_tokenizer"]
            model = self.models["cross_encoder"]

            candidate_texts = [c.get("text", "") for c in candidates]
            pairs = [[question, text] for text in candidate_texts]

            inputs = tokenizer(
                pairs, padding=True, truncation=True, return_tensors="np", max_length=512
            )
            outputs = model(**inputs)

            if not hasattr(outputs, "logits"):
                raise RerankingError("Model output missing 'logits'")

            # Apply sigmoid to logits so scores are in (0, 1) and
            # comparable with bi-encoder cosine similarity scores
            raw_logits = outputs.logits.reshape(-1).tolist()
            scores = [1.0 / (1.0 + math.exp(-x)) for x in raw_logits]

            if len(scores) != len(candidates):
                raise RerankingError(
                    f"Score count mismatch: got {len(scores)} scores for {len(candidates)} candidates"
                )

            ranked_results = []
            for score, content in zip(scores, candidates):
                ranked_results.append(
                    schema.SearchResult(
                        text=content.get("text", ""),
                        score=score,
                        metadata={k: v for k, v in content.items() if k != "text"}
                    )
                )

            # Sort descending (Highest score first)
            ranked_results.sort(key=lambda x: x.score, reverse=True)

            return ranked_results[:top_k]

        except KeyError as e:
            logger.error(f"Model or tokenizer not found in config: {e}")
            raise RerankingError(f"Model configuration error: {e}", original_error=e)
        except Exception as e:
            logger.error(f"Reranking failed for {len(candidates)} candidates: {e}")
            raise RerankingError(f"Failed to rerank results: {str(e)}", original_error=e)

    def retrieve_context(self, question: str, collection_name: Optional[str] = None, n_retrieval: Optional[int] = None, n_ranking: Optional[int] = None) -> List[schema.SearchResult]:
        """
        Orchestrates the full retrieval pipeline: Embed -> Search -> Rerank.

        Args:
            question: Query question
            collection_name: Optional collection name (uses default if not provided)
            n_retrieval: Number of candidates to retrieve
            n_ranking: Number of top results to return after reranking

        Returns:
            Ranked list of search results

        Raises:
            HTTPException: If collection not found
            EmbeddingError: If embedding fails
            VectorDBError: If search fails
            RerankingError: If reranking fails
        """
        collection = collection_name or self.config.kb_name

        try:
            if not self.qdrant.collection_exists(collection):
                raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            raise VectorDBError(f"Failed to access collection '{collection}'", original_error=e)

        limit = n_retrieval or self.config.kb_limit
        requested_top_k = n_ranking or limit
        if requested_top_k > limit:
            logger.warning(
                f"n_ranking ({requested_top_k}) > n_retrieval ({limit}); "
                f"capping to {limit} (cannot rank more candidates than retrieved)"
            )
        top_k = min(requested_top_k, limit)

        # 1. Embed (will raise EmbeddingError on failure)
        query_vector = self._embed_text(question)

        # 2. Search (will raise VectorDBError on failure, with retry)
        points = self.search(collection, query_vector, limit)

        if not points:
            logger.info(f"No results found for query in collection '{collection}'")
            return []

        candidates_list = [point.payload or {} for point in points]

        # 3. Rerank (will raise RerankingError on failure)
        final_results = self.rerank(question, candidates_list, top_k)

        return final_results
