"""Unit tests for CLIP model wrapper."""

import numpy as np
import pytest
from PIL import Image
from src.semantic_search.clip_model import CLIPRetriever


def test_clip_retriever_cpu_and_shapes():
    # Initialize with batch_size=2
    retriever = CLIPRetriever(device="cpu", batch_size=2)
    assert retriever.device == "cpu"

    # Test text encoding
    texts = ["a satellite photo of an airport", "a small pond near trees"]
    text_emb = retriever.encode_text(texts)
    assert text_emb.shape == (2, 512)
    assert text_emb.dtype == np.float32

    # Verify L2 normalization
    norms = np.linalg.norm(text_emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Test image encoding with synthetic PIL images
    img1 = Image.new("RGB", (224, 224), color=(100, 150, 200))
    img2 = Image.new("RGB", (224, 224), color=(50, 80, 120))
    img_emb = retriever.encode_images([img1, img2])
    assert img_emb.shape == (2, 512)
    assert img_emb.dtype == np.float32

    img_norms = np.linalg.norm(img_emb, axis=1)
    assert np.allclose(img_norms, 1.0, atol=1e-4)
