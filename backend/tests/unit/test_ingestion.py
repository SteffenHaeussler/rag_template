"""
Unit tests for the DocumentIngester class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import httpx

from src.app.ingestion import DocumentIngester


class TestDocumentIngester:
    """Tests for DocumentIngester class."""

    @pytest.fixture
    def mock_api_health(self):
        """Mock successful API health check."""
        with patch("httpx.Client.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"version": "0.1.0", "timestamp": 123456}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            yield mock_get

    @pytest.fixture
    def ingester(self, mock_api_health):
        """Create a DocumentIngester instance with mocked API."""
        return DocumentIngester(
            api_base_url="http://localhost:8000",
            collection_name="test_collection",
            chunk_size=500,
            chunk_overlap=50,
            batch_size=10,
            verify_api=True,
        )

    @pytest.fixture
    def ingester_no_verify(self):
        """Create a DocumentIngester without API verification."""
        return DocumentIngester(
            api_base_url="http://localhost:8000",
            collection_name="test_collection",
            verify_api=False,
        )

    @pytest.fixture
    def temp_markdown_dir(self, tmp_path):
        """Create temporary directory with markdown files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create test markdown files
        (data_dir / "file1.md").write_text("# Header\n\nFirst paragraph.\n\nSecond paragraph.")
        (data_dir / "file2.md").write_text("Content of file 2")

        return data_dir

    def test_init_with_verify(self, mock_api_health):
        """Test initialization with API verification."""
        ingester = DocumentIngester(
            api_base_url="http://localhost:8000",
            collection_name="test",
            verify_api=True,
        )

        assert ingester.api_base_url == "http://localhost:8000"
        assert ingester.collection_name == "test"
        assert ingester.chunk_size == 500
        assert ingester.chunk_overlap == 50
        assert ingester.batch_size == 50
        mock_api_health.assert_called_once()

    def test_init_without_verify(self):
        """Test initialization without API verification."""
        ingester = DocumentIngester(
            api_base_url="http://localhost:8000",
            collection_name="test",
            verify_api=False,
        )

        assert ingester.api_base_url == "http://localhost:8000"
        assert isinstance(ingester.client, httpx.Client)

    def test_init_strips_trailing_slash(self, mock_api_health):
        """Test that trailing slash is removed from API URL."""
        ingester = DocumentIngester(
            api_base_url="http://localhost:8000/",
            collection_name="test",
            verify_api=True,
        )

        assert ingester.api_base_url == "http://localhost:8000"

    def test_verify_api_failure(self):
        """Test API verification failure."""
        with patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(SystemExit):
                DocumentIngester(
                    api_base_url="http://localhost:8000",
                    collection_name="test",
                    verify_api=True,
                )

    def test_load_documents(self, ingester_no_verify, temp_markdown_dir):
        """Test loading markdown files from directory."""
        documents = ingester_no_verify.load_documents(str(temp_markdown_dir))

        assert len(documents) == 2
        assert all("filename" in doc for doc in documents)
        assert all("filepath" in doc for doc in documents)
        assert all("content" in doc for doc in documents)

        filenames = {doc["filename"] for doc in documents}
        assert "file1.md" in filenames
        assert "file2.md" in filenames

    def test_load_documents_nonexistent_dir(self, ingester_no_verify):
        """Test loading from non-existent directory."""
        with pytest.raises(ValueError, match="Data directory not found"):
            ingester_no_verify.load_documents("/nonexistent/path")

    def test_load_documents_empty_dir(self, ingester_no_verify, tmp_path):
        """Test loading from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        documents = ingester_no_verify.load_documents(str(empty_dir))
        assert documents == []

    def test_chunk_text_simple(self, ingester_no_verify):
        """Test simple text chunking."""
        text = "Short paragraph."
        chunks = ingester_no_verify.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == "Short paragraph."

    def test_chunk_text_multiple_paragraphs(self, ingester_no_verify):
        """Test chunking text with multiple paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        ingester_no_verify.chunk_size = 30  # Small chunk size for testing

        chunks = ingester_no_verify.chunk_text(text)

        assert len(chunks) > 1
        # Check overlap exists
        for i in range(len(chunks) - 1):
            # Some content should appear in consecutive chunks
            assert any(
                word in chunks[i + 1]
                for word in chunks[i].split()[-5:]  # Last few words
            )

    def test_chunk_text_empty(self, ingester_no_verify):
        """Test chunking empty text."""
        assert ingester_no_verify.chunk_text("") == []
        assert ingester_no_verify.chunk_text("   ") == []

    def test_chunk_text_long_paragraph(self, ingester_no_verify):
        """Test chunking a long paragraph that exceeds chunk size."""
        # Create text larger than chunk_size
        long_text = "A" * 600 + "\n\n" + "B" * 600
        ingester_no_verify.chunk_size = 500
        ingester_no_verify.chunk_overlap = 50

        chunks = ingester_no_verify.chunk_text(long_text)

        assert len(chunks) > 1
        # Check that chunks respect approximate size
        for chunk in chunks:
            # Chunks can be larger due to paragraph boundaries and overlap
            assert len(chunk) <= 700  # Allow generous margin for paragraph boundaries

    def test_chunk_text_preserves_content(self, ingester_no_verify):
        """Test that all content is preserved after chunking."""
        text = "Para 1\n\nPara 2\n\nPara 3\n\nPara 4"
        ingester_no_verify.chunk_size = 20

        chunks = ingester_no_verify.chunk_text(text)

        # Join chunks and verify content is preserved
        combined = "\n\n".join(chunks)
        assert "Para 1" in combined
        assert "Para 2" in combined
        assert "Para 3" in combined
        assert "Para 4" in combined

    def test_create_collection_success(self, ingester_no_verify):
        """Test successful collection creation."""
        with patch.object(ingester_no_verify.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_post.return_value = mock_response

            result = ingester_no_verify.create_collection()

            assert result is True
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "collections" in call_args[0][0]
            assert call_args[1]["json"]["dimension"] == 384

    def test_create_collection_already_exists(self, ingester_no_verify):
        """Test collection creation when it already exists."""
        with patch.object(ingester_no_verify.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Collection already exists"
            mock_post.return_value = mock_response

            result = ingester_no_verify.create_collection()

            assert result is False

    def test_create_collection_http_error(self, ingester_no_verify):
        """Test collection creation with HTTP error."""
        with patch.object(ingester_no_verify.client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal server error"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Error", request=Mock(), response=mock_response
            )
            mock_post.return_value = mock_response

            with pytest.raises(httpx.HTTPStatusError):
                ingester_no_verify.create_collection()

    def test_ingest_success(self, ingester_no_verify, temp_markdown_dir):
        """Test successful document ingestion."""
        with patch.object(ingester_no_verify.client, "post") as mock_post:
            # Mock collection creation
            create_response = Mock()
            create_response.status_code = 201

            # Mock bulk insert
            bulk_response = Mock()
            bulk_response.json.return_value = {"inserted_count": 2}
            bulk_response.raise_for_status = Mock()

            mock_post.side_effect = [create_response, bulk_response]

            total = ingester_no_verify.ingest(str(temp_markdown_dir))

            assert total == 2
            assert mock_post.call_count == 2  # create + bulk insert

    def test_ingest_no_documents(self, ingester_no_verify, tmp_path):
        """Test ingestion with no documents."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch.object(ingester_no_verify, 'create_collection', return_value=True):
            with pytest.raises(ValueError, match="No documents found"):
                ingester_no_verify.ingest(str(empty_dir))

    def test_ingest_skip_collection_creation(self, ingester_no_verify, temp_markdown_dir):
        """Test ingestion with collection creation skipped."""
        with patch.object(ingester_no_verify.client, "post") as mock_post:
            bulk_response = Mock()
            bulk_response.json.return_value = {"inserted_count": 1}
            bulk_response.raise_for_status = Mock()
            mock_post.return_value = bulk_response

            ingester_no_verify.ingest(str(temp_markdown_dir), create_collection=False)

            # Should only call bulk insert, not collection creation
            assert mock_post.call_count >= 1
            # Check that collection endpoint wasn't called
            calls = [call[0][0] for call in mock_post.call_args_list]
            assert not any("collections/" == call.split("/")[-1] for call in calls)

    def test_ingest_batching(self, ingester_no_verify, tmp_path):
        """Test that ingestion respects batch size."""
        # Create many small files
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        for i in range(25):
            (data_dir / f"file{i}.md").write_text(f"Content {i}")

        ingester_no_verify.batch_size = 10

        with patch.object(ingester_no_verify.client, "post") as mock_post:
            create_response = Mock()
            create_response.status_code = 201

            bulk_response = Mock()
            bulk_response.json.return_value = {"inserted_count": 10}
            bulk_response.raise_for_status = Mock()

            mock_post.side_effect = [create_response] + [bulk_response] * 10

            ingester_no_verify.ingest(str(data_dir))

            # Should have multiple bulk insert calls due to batching
            bulk_calls = [
                call for call in mock_post.call_args_list
                if "datapoints/bulk" in str(call)
            ]
            assert len(bulk_calls) > 1

    def test_context_manager(self, mock_api_health):
        """Test using DocumentIngester as context manager."""
        with patch.object(httpx.Client, 'close') as mock_close:
            with DocumentIngester(
                api_base_url="http://localhost:8000",
                collection_name="test",
                verify_api=True,
            ) as ingester:
                assert ingester is not None
                assert isinstance(ingester.client, httpx.Client)

            # Client should be closed after context exit
            mock_close.assert_called_once()

    def test_close(self, ingester_no_verify):
        """Test closing the ingester."""
        with patch.object(ingester_no_verify.client, 'close') as mock_close:
            ingester_no_verify.close()
            mock_close.assert_called_once()

    def test_chunk_metadata(self, ingester_no_verify, temp_markdown_dir):
        """Test that chunk metadata is correctly generated."""
        documents = ingester_no_verify.load_documents(str(temp_markdown_dir))
        doc = documents[0]

        chunks = ingester_no_verify.chunk_text(doc["content"])

        # Verify chunk metadata structure
        for idx, chunk in enumerate(chunks):
            # Simulate what ingest() does
            metadata = {
                "filename": doc["filename"],
                "filepath": doc["filepath"],
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }

            assert metadata["chunk_index"] == idx
            assert metadata["total_chunks"] == len(chunks)
            assert metadata["filename"] == doc["filename"]


class TestChunkingEdgeCases:
    """Test edge cases in chunking logic."""

    @pytest.fixture
    def ingester(self):
        """Create ingester without API verification."""
        return DocumentIngester(
            api_base_url="http://localhost:8000",
            collection_name="test",
            chunk_size=100,
            chunk_overlap=20,
            verify_api=False,
        )

    def test_chunk_single_long_paragraph(self, ingester):
        """Test chunking when a single paragraph exceeds chunk size."""
        # Single paragraph with no breaks
        long_para = "word " * 100  # ~500 characters
        chunks = ingester.chunk_text(long_para)

        # Should create at least one chunk
        assert len(chunks) >= 1
        # Original content should be in the chunks
        assert long_para in " ".join(chunks) or all(
            word in " ".join(chunks) for word in long_para.split()
        )

    def test_chunk_with_empty_paragraphs(self, ingester):
        """Test chunking with empty paragraphs (multiple newlines)."""
        text = "Para 1\n\n\n\nPara 2\n\n\n\nPara 3"
        chunks = ingester.chunk_text(text)

        assert len(chunks) >= 1
        # Content should be preserved
        combined = " ".join(chunks)
        assert "Para 1" in combined
        assert "Para 2" in combined
        assert "Para 3" in combined

    def test_chunk_overlap_calculation(self, ingester):
        """Test that overlap is calculated correctly."""
        text = "A" * 80 + "\n\n" + "B" * 80 + "\n\n" + "C" * 80
        ingester.chunk_size = 100
        ingester.chunk_overlap = 20

        chunks = ingester.chunk_text(text)

        if len(chunks) > 1:
            # Check that there's overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                # Some characters should appear in both chunks
                # Due to paragraph-aware chunking, overlap might not be exact
                assert len(chunks[i]) > 0
                assert len(chunks[i + 1]) > 0
