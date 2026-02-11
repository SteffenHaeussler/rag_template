#!/usr/bin/env python
"""Download HuggingFace transformer models and export them to ONNX format.

Usage:
    python scripts/export_onnx.py \
        --bi-encoder sentence-transformers/all-MiniLM-L6-v2 \
        --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --output-dir models
"""

import argparse
from pathlib import Path

from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification,
)
from transformers import AutoTokenizer


def export_model(model_name: str, output_path: Path, task: str) -> None:
    """Download a HuggingFace model and export it to ONNX."""
    print(f"Exporting {model_name} ({task}) -> {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    if task == "feature-extraction":
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name, export=True
        )
    elif task == "sequence-classification":
        model = ORTModelForSequenceClassification.from_pretrained(
            model_name, export=True
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)

    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace models to ONNX format"
    )
    parser.add_argument(
        "--bi-encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model ID for the bi-encoder",
    )
    parser.add_argument(
        "--cross-encoder",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="HuggingFace model ID for the cross-encoder",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Base output directory (default: models)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    export_model(
        args.bi_encoder,
        output_dir / "bi_encoder",
        task="feature-extraction",
    )

    export_model(
        args.cross_encoder,
        output_dir / "cross_encoder",
        task="sequence-classification",
    )

    print("\nDone. Set these in your env file:")
    print(f"  bi_encoder_path={output_dir / 'bi_encoder'}")
    print(f"  cross_encoder_path={output_dir / 'cross_encoder'}")


if __name__ == "__main__":
    main()
