from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastapi import Request, HTTPException
import src.app.v1.schema as schema

class RetrievalService:
    def __init__(self, request: Request):
        self.request = request
        self.config = request.app.state.config
        self.qdrant: QdrantClient = request.app.state.qdrant

    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using the bi-encoder."""
        inputs = self.config.models["bi_tokenizer"](
            text, padding=True, truncation=True, return_tensors="np"
        )
        outputs = self.config.models["bi_encoder"](**inputs)
        return np.mean(outputs.last_hidden_state, axis=1).tolist()[0]

    def search(self, collection_name: str, query_vector: List[float], limit: int) -> List[PointStruct]:
        """
        Generic wrapper for Qdrant querying.
        """
        results = self.qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        )
        return results.points

    def rerank(self, question: str, candidates: List[Dict[str, Any]], top_k: int) -> List[schema.SearchResult]:
        """Sorts candidates by relevance using the Cross-Encoder."""
        if not candidates:
            return []

        tokenizer = self.config.models["cross_tokenizer"]
        model = self.config.models["cross_encoder"]

        candidate_texts = [c.get("text", "") for c in candidates]
        pairs = [[question, text] for text in candidate_texts]

        inputs = tokenizer(
            pairs, padding=True, truncation=True, return_tensors="np"
        )
        outputs = model(**inputs)

        # Flatten logits to a 1D array
        scores = outputs.logits.reshape(-1).tolist()

        ranked_results = []
        for score, content in zip(scores, candidates):
            ranked_results.append(
                schema.SearchResult(
                    text=content.get("text", ""),
                    score=score,
                    metadata=content # Pass full payload as metadata
                )
            )

        # Sort descending (Highest score first)
        ranked_results.sort(key=lambda x: x.score, reverse=True)

        return ranked_results[:top_k]

    def retrieve_context(self, question: str, collection_name: Optional[str] = None, n_retrieval: Optional[int] = None, n_ranking: Optional[int] = None) -> List[schema.SearchResult]:
        """
        Orchestrates the full retrieval pipeline: Embed -> Search -> Rerank.
        """
        collection = collection_name or self.config.kb_name
        if not self.qdrant.collection_exists(collection):
            raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")

        limit = n_retrieval or self.config.kb_limit
        top_k = n_ranking or self.config.kb_limit

        # 1. Embed
        query_vector = self._embed_text(question)

        # 2. Search
        points = self.search(collection, query_vector, limit)
        candidates_list = [point.payload or {} for point in points]

        # 3. Rerank
        final_results = self.rerank(question, candidates_list, top_k)

        return final_results
