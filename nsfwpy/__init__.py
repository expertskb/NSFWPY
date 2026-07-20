"""
NSFWPY Library - Lightweight, CPU-optimized NSFW Image Classification with ONNX Runtime.
Supports static images and Animated WebP, GIF, and APNG formats.
"""

from .core import NSFWModel, load_model, load
from .constants import NSFW_CATEGORIES, DEFAULT_MODEL_PATHS
from .image import load_image, load_animated_frames, preprocess_image

__version__ = "1.0.5"

__all__ = [
    "NSFWModel",
    "load_model",
    "load",
    "NSFW_CATEGORIES",
    "DEFAULT_MODEL_PATHS",
    "load_image",
    "load_animated_frames",
    "preprocess_image",
]
