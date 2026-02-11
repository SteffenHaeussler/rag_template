"""RAG pipeline: retrieve context from Qdrant, generate answer via LiteLLM."""

import litellm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from api.schemas import Source


def embed_query(model: SentenceTransformer, query: str) -> list[float]:
    """Embed a single query string."""
    return model.encode(query).tolist()


def retrieve(
    qdrant: QdrantClient, query_vector: list[float], top_k: int, collection_name: str
) -> list[Source]:
    """Search Qdrant for the most similar chunks."""
    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    sources = []
    for point in results.points:
        sources.append(
            Source(
                text=point.payload["text"],
                filename=point.payload["filename"],
                score=round(point.score, 4),
            )
        )
    return sources


def generate_answer(
    question: str, sources: list[Source], generation_model: str, api_key: str
) -> str:
    """Call an LLM via LiteLLM to generate an answer given retrieved context."""
    if not sources:
        return "I couldn't find any relevant information to answer your question."

    context = "\n\n---\n\n".join(
        f"[Source: {s.filename}]\n{s.text}" for s in sources
    )

    response = litellm.completion(
        model=generation_model,
        api_key=api_key,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question based on the "
                    "provided context. If the context doesn't contain enough information, "
                    "say so. Cite the source filenames when relevant."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content
