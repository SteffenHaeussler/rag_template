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
    """
    models = {}

    models["bi_encoder"] = ORTModelForFeatureExtraction.from_pretrained(
        f"{rootdir}/{bi_encoder_path}"
    )
    models["bi_tokenizer"] = AutoTokenizer.from_pretrained(
        f"{rootdir}/{bi_encoder_path}"
    )

    models["cross_encoder"] = ORTModelForSequenceClassification.from_pretrained(
        f"{rootdir}/{cross_encoder_path}"
    )
    models["cross_tokenizer"] = AutoTokenizer.from_pretrained(
        f"{rootdir}/{cross_encoder_path}"
    )

    return models
