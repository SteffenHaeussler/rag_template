def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return len(text) // 4


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens with overlap.

    Strategy:
    1. Split into paragraphs (double newline).
    2. Greedily merge paragraphs into chunks up to chunk_size tokens.
    3. If a single paragraph exceeds chunk_size, split it by sentences.
    4. Maintain chunk_overlap tokens of overlap between consecutive chunks.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        if para_tokens > chunk_size:
            # Flush current buffer first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            # Split long paragraph by sentences
            sentences = _split_sentences(para)
            for sentence in sentences:
                sent_tokens = _estimate_tokens(sentence)
                if current_tokens + sent_tokens > chunk_size and current_parts:
                    chunks.append(" ".join(current_parts))
                    # Keep overlap
                    overlap_parts, overlap_tokens = _get_overlap(
                        current_parts, chunk_overlap
                    )
                    current_parts = overlap_parts
                    current_tokens = overlap_tokens
                current_parts.append(sentence)
                current_tokens += sent_tokens

            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_tokens = 0
            continue

        if current_tokens + para_tokens > chunk_size and current_parts:
            chunks.append("\n\n".join(current_parts))
            # Keep overlap from end of previous chunk
            overlap_parts, overlap_tokens = _get_overlap(current_parts, chunk_overlap)
            current_parts = overlap_parts
            current_tokens = overlap_tokens

        current_parts.append(para)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitting on '. ', '! ', '? '."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _get_overlap(parts: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Get trailing parts that fit within the overlap token budget."""
    result: list[str] = []
    tokens = 0
    for part in reversed(parts):
        part_tokens = _estimate_tokens(part)
        if tokens + part_tokens > overlap_tokens:
            break
        result.insert(0, part)
        tokens += part_tokens
    return result, tokens
