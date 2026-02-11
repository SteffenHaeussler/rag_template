"""RAG pipeline: retrieve context from Qdrant, generate answer via Gemini."""

from google import genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from api import config
from api.schemas import Source


def embed_query(model: SentenceTransformer, query: str) -> list[float]:
    """Embed a single query string."""
    return model.encode(query).tolist()


def retrieve(
    qdrant: QdrantClient, query_vector: list[float], top_k: int
) -> list[Source]:
    """Search Qdrant for the most similar chunks."""
    results = qdrant.query_points(
        collection_name=config.COLLECTION_NAME,
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
    client: genai.Client, question: str, sources: list[Source]
) -> str:
    """Call Gemini to generate an answer given retrieved context."""
    if not sources:
        return "I couldn't find any relevant information to answer your question."

    context = "\n\n---\n\n".join(
        f"[Source: {s.filename}]\n{s.text}" for s in sources
    )

    prompt = (
        "You are a helpful assistant. Answer the user's question based on the "
        "provided context. If the context doesn't contain enough information, "
        "say so. Cite the source filenames when relevant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = client.models.generate_content(
        model=config.GENERATION_MODEL,
        contents=prompt,
    )
    return response.text
