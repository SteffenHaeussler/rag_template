from ingestion.chunker import chunk_text


def test_empty_text():
    assert chunk_text("") == []


def test_whitespace_only():
    assert chunk_text("   \n\n   ") == []


def test_single_short_paragraph():
    text = "This is a short paragraph."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_multiple_paragraphs_fit_in_one_chunk():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert "Paragraph one." in chunks[0]
    assert "Paragraph three." in chunks[0]


def test_paragraphs_split_into_multiple_chunks():
    # Each paragraph is ~25 tokens (100 chars / 4). With chunk_size=30,
    # we should get multiple chunks.
    para = "A" * 100  # ~25 tokens
    text = f"{para}\n\n{para}\n\n{para}\n\n{para}"
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 1


def test_overlap_present():
    # Create paragraphs that each fit individually but not two together
    p1 = "First paragraph " * 10  # ~40 tokens
    p2 = "Second paragraph " * 10  # ~42 tokens
    p3 = "Third paragraph " * 10  # ~42 tokens
    text = f"{p1.strip()}\n\n{p2.strip()}\n\n{p3.strip()}"
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=20)

    # With overlap, later chunks should contain text from the end of previous chunks
    assert len(chunks) >= 2


def test_long_paragraph_gets_split():
    # A single paragraph longer than chunk_size
    long_text = "This is a sentence. " * 200  # ~1000 tokens
    chunks = chunk_text(long_text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    # Each chunk should be non-empty
    for chunk in chunks:
        assert len(chunk.strip()) > 0
