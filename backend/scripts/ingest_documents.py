#!/usr/bin/env python
"""
Document Ingestion Script for RAG Pipeline

This script loads markdown files from the data directory, chunks them,
and ingests them into Qdrant via the API.

IMPORTANT: The API server must be running before using this script.
           Run: make dev  OR  docker compose up

Usage:
    python scripts/ingest_documents.py --data-dir ../data --collection-name temp
    python scripts/ingest_documents.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app.ingestion import DocumentIngester

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    tqdm = None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest markdown documents into Qdrant for RAG (via API)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory containing markdown files (default: ../data)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="temp",
        help="Collection name (default: temp)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for ingestion (default: 50)",
    )
    parser.add_argument(
        "--skip-create-collection",
        action="store_true",
        help="Skip collection creation (assume it exists)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Document Ingestion Script")
    print("=" * 80)
    print(f"API URL: {args.api_url}")
    print(f"Collection: {args.collection_name}")
    print(f"Data directory: {args.data_dir}")
    print(f"Chunk size: {args.chunk_size} chars")
    print(f"Chunk overlap: {args.chunk_overlap} chars")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    # Initialize ingester
    try:
        with DocumentIngester(
            api_base_url=args.api_url,
            collection_name=args.collection_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
        ) as ingester:
            print(f"✓ Connected to API at {args.api_url}")

            # Load documents first to show progress
            print(f"\nLoading documents from {args.data_dir}...")
            documents = ingester.load_markdown_files(args.data_dir)
            print(f"Found {len(documents)} markdown files")
            for doc in documents:
                print(f"  - {doc['filename']} ({len(doc['content'])} characters)")

            # Chunk and prepare datapoints
            print(f"\nChunking documents...")
            all_chunks = []
            for doc in documents:
                chunks = ingester.chunk_text(doc["content"])
                for idx, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "filename": doc["filename"],
                        "filepath": doc["filepath"],
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                    })

            print(f"Created {len(all_chunks)} chunks total")

            # Show progress with tqdm if available
            if tqdm:
                print(f"\nIngesting datapoints...")
                datapoints = []
                for chunk in all_chunks:
                    datapoints.append({
                        "text": chunk["text"],
                        "metadata": {
                            "filename": chunk["filename"],
                            "filepath": chunk["filepath"],
                            "chunk_index": chunk["chunk_index"],
                            "total_chunks": chunk["total_chunks"],
                        },
                    })

                total_inserted = 0
                for i in tqdm(
                    range(0, len(datapoints), args.batch_size),
                    desc="Ingesting batches",
                    unit="batch"
                ):
                    batch = datapoints[i : i + args.batch_size]
                    try:
                        response = ingester.client.post(
                            f"{ingester.api_base_url}/v1/collections/{ingester.collection_name}/datapoints/bulk",
                            json=batch,
                        )
                        response.raise_for_status()
                        result = response.json()
                        total_inserted += result.get("inserted_count", len(batch))
                    except Exception as e:
                        print(f"\nError inserting batch: {e}")
                        continue

                print(f"\n✓ Ingestion complete! Inserted {total_inserted} datapoints")
            else:
                # Fallback without progress bar
                total_inserted = ingester.ingest(
                    data_dir=args.data_dir,
                    create_collection=not args.skip_create_collection,
                )
                print(f"\n✓ Ingestion complete! Inserted {total_inserted} datapoints")

    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
