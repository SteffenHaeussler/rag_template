import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import onnxruntime
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from transformers import AutoModel


def load_onnx(model_path: str, example_data: list, kind: str):


    if kind == "cross_encoder":
        ort_model_loaded = ORTModelForSequenceClassification.from_pretrained(
            model_path
        )

    elif kind == "bi_encoder":
        ort_model_loaded = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
        )
    else:
        raise ValueError("Invalid model kind. Must be 'cross_encoder' or 'bi_encoder'.")


    tokenizer_for_ort = AutoTokenizer.from_pretrained(model_path)

    # Prepare input (Optimum can handle PyTorch tensors or NumPy arrays)
    encoded_input_optimum = tokenizer_for_ort(
        example_data,
        padding=True,
        truncation=True,
        return_tensors="np" # Can be "pt" for PyTorch tensors or "np" for NumPy
    )

    outputs_optimum = ort_model_loaded(**encoded_input_optimum)

    if kind == "bi_encoder":
        mean_embedding = np.mean(outputs_optimum.last_hidden_state,axis=1).tolist()
        print("Mean embeddings from Optimum ORTModel:")
        for i, sentence in enumerate(example_data):
            print(f"  Sentence {i+1}: {mean_embedding[i][0:5]}")

    elif kind == "cross_encoder":
        logits = outputs_optimum.logits.tolist()
        print("Logits from Optimum ORTModel:")

        for i, pair in enumerate(example_data):
            print(f"  Pair {i+1}: Score = {logits[i][0]:.4f}")

    return True


test_pairs = [
    ("How is the weather today?", "The weather is sunny and warm."),
    ("What is ONNX Runtime?", "It is a cross-platform inferencing and training accelerator."),
    ("This is a relevant document.", "This is a relevant document."),
    ("This is a relevant document.", "This is a completely irrelevant document.")
]

test_sentences = [
    "This is an example sentence.",
    "Each sentence is converted to a vector.",
    "ONNX makes deployment easier."
]


load_onnx("models/bi_encoder", test_sentences, "bi_encoder")
load_onnx("models/cross_encoder", test_pairs, "cross_encoder")
