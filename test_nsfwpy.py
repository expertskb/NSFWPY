"""
Automated unit & integration tests for nsfwpy.
"""
import pytest
import numpy as np
from PIL import Image

import nsfwpy


def test_package_import():
    """Verify package import."""
    assert nsfwpy.__version__ == "1.0.3"


def test_load_models():
    """Verify loading different ONNX model architectures."""
    model_v2 = nsfwpy.load_model("mobilenet_v2")
    assert model_v2 is not None

    model_v3 = nsfwpy.load_model("mobilenet_v3")
    assert model_v3 is not None

    model_inc = nsfwpy.load_model("inception_v3")
    assert model_inc is not None


def test_single_classification():
    """Verify single image classification return format."""
    model = nsfwpy.load_model("mobilenet_v2")

    # Generate synthetic RGB PIL image
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    results = model.classify(img, top_k=5)

    assert len(results) == 5
    assert "className" in results[0]
    assert "probability" in results[0]

    # Verify probability ordering
    probs = [item["probability"] for item in results]
    assert probs == sorted(probs, reverse=True)


def test_batch_classification():
    """Verify batch classification execution."""
    model = nsfwpy.load_model("mobilenet_v2")
    img1 = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    img2 = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

    results = model.classify_batch([img1, img2], top_k=3)
    assert len(results) == 2
    assert len(results[0]) == 3
    assert len(results[1]) == 3
