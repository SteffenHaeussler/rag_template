"""Ingestion pipeline: load documents, chunk, embed, store in Qdrant."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from ingestion import config
from ingestion.chunker import chunk_text
from ingestion.loader import load_documents


def main() -> None:
    print(f"Loading documents from {config.DATA_DIR}...")
    documents = load_documents(config.DATA_DIR)
    if not documents:
        print("No documents found. Exiting.")
        return
    print(f"Loaded {len(documents)} document(s).")

    # Chunk documents
    all_chunks: list[dict] = []
    for doc in documents:
        chunks = chunk_text(doc["content"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {"text": chunk, "filename": doc["filename"], "chunk_index": i}
            )
    print(f"Created {len(all_chunks)} chunk(s).")

    # Load embedding model
    print(f"Loading embedding model '{config.EMBEDDING_MODEL}'...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    texts = [c["text"] for c in all_chunks]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embedding_dim = embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")

    # Store in Qdrant
    qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    # Recreate collection
    if qdrant.collection_exists(config.COLLECTION_NAME):
        qdrant.delete_collection(config.COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "text": chunk["text"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk, embedding in zip(all_chunks, embeddings)
    ]

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points[i : i + batch_size],
        )

    print(
        f"Stored {len(points)} vectors in Qdrant collection '{config.COLLECTION_NAME}'."
    )
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
