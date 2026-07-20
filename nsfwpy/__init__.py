"""
NSFWPY Library - Lightweight, CPU-optimized NSFW Image Classification with ONNX Runtime.
"""

from .core import NSFWModel, load_model
from .constants import NSFW_CATEGORIES, DEFAULT_MODEL_PATHS
from .image import load_image, preprocess_image

__version__ = "1.0.2"

__all__ = [
    "NSFWModel",
    "load_model",
    "NSFW_CATEGORIES",
    "DEFAULT_MODEL_PATHS",
    "load_image",
    "preprocess_image",
]
