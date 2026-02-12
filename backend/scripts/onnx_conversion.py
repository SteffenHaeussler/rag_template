import os
from pathlib import Path

from dotenv import load_dotenv
import onnxruntime
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from transformers import AutoModel


def convert_to_onnx(model_name: str, output_path: str, kind: str):

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if kind == "cross_encoder":
        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,
        )

    elif kind == "bi_encoder":
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
        )

    else:
        raise ValueError("Invalid model kind. Must be 'cross_encoder' or 'bi_encoder'.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    ort_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path) # Save tokenizer for easy loading with ONNX model

    return True


load_dotenv("dev.env")

convert_to_onnx(os.getenv("bi_encoder"), "models/bi_encoder", "bi_encoder")
convert_to_onnx(os.getenv("cross_encoder"), "models/cross_encoder", "cross_encoder")
