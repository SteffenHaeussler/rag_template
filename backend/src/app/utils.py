from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
)
from transformers import AutoTokenizer


def load_models(rootdir, bi_encoder_path, cross_encoder_path):
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
