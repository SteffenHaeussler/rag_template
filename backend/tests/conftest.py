import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.app.main import app, get_application
from src.app.config import Config


@pytest.fixture
def mock_qdrant():
    mock = MagicMock()

    # Mock get_collection for dimension validation
    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 384  # Default dimension
    mock.get_collection.return_value = mock_collection_info

    return mock


@pytest.fixture
def mock_models():
    # Mock both encoder models and tokenizers if needed,
    # but primarily the models are what we care about for the logic
    bi_encoder = MagicMock()
    # Mock the output of the bi-encoder
    mock_output = MagicMock()
    mock_output.last_hidden_state = MagicMock()
    # Return a numpy-like list for mean pooling simulation
    import numpy as np
    mock_output.last_hidden_state = np.array([[[0.1] * 384]])
    bi_encoder.return_value = mock_output

    cross_encoder = MagicMock()
    # Mock logits for cross-encoder
    cross_output = MagicMock()
    cross_output.logits = np.array([0.9, 0.1])
    cross_encoder.return_value = cross_output

    bi_tokenizer = MagicMock()
    bi_tokenizer.return_value = {"input_ids": []} # minimal mock

    cross_tokenizer = MagicMock()
    cross_tokenizer.return_value = {"input_ids": []}

    return {
        "bi_encoder": bi_encoder,
        "cross_encoder": cross_encoder,
        "bi_tokenizer": bi_tokenizer,
        "cross_tokenizer": cross_tokenizer,
    }


@pytest.fixture
def test_app(mock_qdrant, mock_models):
    # Override the lifespan or state setup
    # Since `app` is already created in main.py, we can modify its state directly
    # or create a new dependency override if needed.
    # However, main.py sets state in lifespan. We can mock that.

    app.state.qdrant = mock_qdrant
    app.state.config = Config()
    app.state.models = mock_models  # Models now in app.state, not config
    app.state.prompts = {
        "answer": {"en": "Answer: {{ question }} Context: {% for c in context %}{{ c }}{% endfor %}"},
        "prompt": {"en": "Prompt: {{ question }} Context: {% for c in context %}{{ c }}{% endfor %}"},
    }

    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)
