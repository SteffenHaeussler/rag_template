from pathlib import Path
from typing import Dict, Any
from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
)
from transformers import AutoTokenizer


def load_models(rootdir: str, bi_encoder_path: str, cross_encoder_path: str) -> Dict[str, Any]:
    """
    Load ONNX models for embedding and reranking.

    Args:
        rootdir: Root directory path
        bi_encoder_path: Path to bi-encoder model
        cross_encoder_path: Path to cross-encoder model

    Returns:
        Dictionary containing models and tokenizers

    Raises:
        RuntimeError: If any model or tokenizer fails to load
    """
    models = {}
    bi_path = str(Path(rootdir) / bi_encoder_path)
    cross_path = str(Path(rootdir) / cross_encoder_path)

    try:
        models["bi_encoder"] = ORTModelForFeatureExtraction.from_pretrained(bi_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load bi-encoder from '{bi_path}': {e}") from e

    try:
        models["bi_tokenizer"] = AutoTokenizer.from_pretrained(bi_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load bi-encoder tokenizer from '{bi_path}': {e}") from e

    try:
        models["cross_encoder"] = ORTModelForSequenceClassification.from_pretrained(cross_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load cross-encoder from '{cross_path}': {e}") from e

    try:
        models["cross_tokenizer"] = AutoTokenizer.from_pretrained(cross_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load cross-encoder tokenizer from '{cross_path}': {e}") from e

    return models
