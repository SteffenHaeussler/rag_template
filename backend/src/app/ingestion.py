import sys
from pathlib import Path
from typing import List, Dict
import httpx


class DocumentIngester:
    """Handles document ingestion into Qdrant via API."""

    def __init__(
        self,
        api_base_url: str,
        collection_name: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 50,
        verify_api: bool = True,
    ):
        """
        Initialize the document ingester.

        Args:
            api_base_url: Base URL of the API (e.g., http://localhost:8000)
            collection_name: Name of the collection
            chunk_size: Approximate characters per chunk
            chunk_overlap: Approximate overlapping characters
            batch_size: Batch size for bulk ingestion
            verify_api: Whether to verify API connection on init
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=60.0)

        # Verify API is accessible
        if verify_api:
            self._verify_api()

    def _verify_api(self) -> Dict[str, any]:
        """
        Verify the API is accessible.

        Returns:
            API health response

        Raises:
            SystemExit: If API is not accessible
        """
        try:
            response = self.client.get(f"{self.api_base_url}/v1/health")
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            print(f"✗ Cannot connect to API at {self.api_base_url}")
            print("  Make sure the API is running:")
            print("    - Run: make dev")
            print("    - Or: docker compose up")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error connecting to API: {e}")
            sys.exit(1)

    def load_markdown_files(self, data_dir: str) -> List[Dict[str, str]]:
        """
        Load all markdown files from the data directory.

        Args:
            data_dir: Directory containing markdown files

        Returns:
            List of document dictionaries with filename, filepath, and content

        Raises:
            ValueError: If data directory doesn't exist
        """
        documents = []
        data_path = Path(data_dir)

        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        md_files = list(data_path.glob("*.md"))
        if not md_files:
            return []

        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(
                    {
                        "filename": md_file.name,
                        "filepath": str(md_file.absolute()),
                        "content": content,
                    }
                )

        return documents

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks based on character count.
        Uses simple paragraph-aware chunking.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Split by double newline (paragraphs) first
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size and we have content
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def create_collection(self, vector_size: int = 384) -> bool:
        """
        Create Qdrant collection if it doesn't exist.

        Args:
            vector_size: Size of the embedding vectors

        Returns:
            True if collection was created, False if it already existed

        Raises:
            httpx.HTTPStatusError: If API request fails
        """
        try:
            response = self.client.post(
                f"{self.api_base_url}/v1/collections/",
                json={
                    "name": self.collection_name,
                    "dimension": vector_size,
                    "distance_metric": "cosine",
                },
            )

            if response.status_code == 201:
                return True
            elif response.status_code == 400 and "already exists" in response.text.lower():
                return False
            else:
                response.raise_for_status()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400 and "already exists" in e.response.text.lower():
                return False
            raise

    def ingest(self, data_dir: str, create_collection: bool = True) -> int:
        """
        Main ingestion pipeline.

        Args:
            data_dir: Directory containing markdown files
            create_collection: Whether to create collection if it doesn't exist

        Returns:
            Total number of datapoints inserted

        Raises:
            ValueError: If no documents found
        """
        # Create collection if needed
        if create_collection:
            self.create_collection()

        # Load documents
        documents = self.load_markdown_files(data_dir)

        if not documents:
            raise ValueError(f"No markdown files found in {data_dir}")

        # Chunk documents
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(doc["content"])

            for idx, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "text": chunk,
                        "filename": doc["filename"],
                        "filepath": doc["filepath"],
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                    }
                )

        # Prepare datapoints for bulk insert
        datapoints = []

        for chunk in all_chunks:
            datapoints.append(
                {
                    "text": chunk["text"],
                    "metadata": {
                        "filename": chunk["filename"],
                        "filepath": chunk["filepath"],
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": chunk["total_chunks"],
                    },
                }
            )

        # Ingest into Qdrant in batches via bulk endpoint
        total_inserted = 0

        for i in range(0, len(datapoints), self.batch_size):
            batch = datapoints[i : i + self.batch_size]

            try:
                response = self.client.post(
                    f"{self.api_base_url}/v1/collections/{self.collection_name}/datapoints/bulk",
                    json=batch,
                )
                response.raise_for_status()
                result = response.json()
                total_inserted += result.get("inserted_count", len(batch))

            except httpx.HTTPStatusError as e:
                print(f"\nError inserting batch {i // self.batch_size + 1}: {e}")
                print(f"Response: {e.response.text}")
                continue

        return total_inserted

    def close(self):
        """Close the HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
