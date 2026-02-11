from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(data_dir: str) -> list[dict]:
    """Read all .txt and .md files from a directory.

    Returns a list of dicts with keys: filename, content.
    """
    documents = []
    data_path = Path(data_dir)

    for file_path in sorted(data_path.iterdir()):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS and file_path.is_file():
            content = file_path.read_text(encoding="utf-8")
            documents.append({"filename": file_path.name, "content": content})

    return documents
