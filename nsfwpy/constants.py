"""
Constants and configuration defaults for NSFWPY in Python.
"""
import os
from pathlib import Path
from typing import List, Dict

# Standard 5 safety classification categories
NSFW_CATEGORIES: List[str] = [
    "Drawing",
    "Hentai",
    "Neutral",
    "Porn",
    "Sexy",
]

# Image preprocessing defaults
DEFAULT_IMAGE_SIZE = (224, 224)

# Model filenames
MODEL_FILENAMES: Dict[str, str] = {
    "nsfw_vit_quantized": "nsfw_vit_quantized.onnx",
    "nsfw_vit": "nsfw_vit.onnx",
    "nsfw_vit_fp16": "model_fp16.onnx",
    "nsfw_vit_int8": "model_int8.onnx",
    "nsfw_vit_uint8": "model_uint8.onnx",
    "nsfw_vit_q4": "model_q4.onnx",
    "nsfw_vit_q4f16": "model_q4f16.onnx",
    "nsfw_vit_bnb4": "model_bnb4.onnx",
}

# Alias for backwards compatibility
DEFAULT_MODEL_PATHS = MODEL_FILENAMES

def get_system_cache_dir() -> Path:
    """
    Get safe cross-platform system model cache directory:
    - Windows: C:\\nsfwpy\\models or C:\\ProgramData\\nsfwpy\\models
    - macOS: /Library/Caches/nsfwpy/models or ~/Library/Caches/nsfwpy/models
    - Linux / Server: /etc/nsfwpy/models or /var/cache/nsfwpy/models
    - Android (Termux / Mobile): /sdcard/.nsfwpy/models or $PREFIX/tmp/nsfwpy/models
    - iOS / iPadOS / Tablet: ~/Documents/.nsfwpy/models
    Falls back to user home directory (~/.nsfwpy/models) if non-root without write permission.
    """
    import sys

    candidates = []

    if os.name == "nt":
        system_drive = os.environ.get("SystemDrive", "C:")
        candidates.extend([
            Path(system_drive) / "nsfwpy" / "models",
            Path(system_drive) / "ProgramData" / "nsfwpy" / "models",
        ])
    elif sys.platform == "darwin":
        candidates.extend([
            Path("/Library/Caches/nsfwpy/models"),
            Path("/etc/nsfwpy/models"),
            Path.home() / "Library" / "Caches" / "nsfwpy" / "models",
        ])
    else:
        # Linux / Android / iOS / Termux / Unix
        candidates.extend([
            Path("/etc/nsfwpy/models"),
            Path("/var/cache/nsfwpy/models"),
            Path("/sdcard/.nsfwpy/models"),
            Path.home() / "Documents" / ".nsfwpy" / "models",
        ])
        prefix = os.environ.get("PREFIX")
        if prefix:
            candidates.append(Path(prefix) / "tmp" / "nsfwpy" / "models")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / ".perm_check"
            test_file.touch()
            test_file.unlink()
            return candidate
        except Exception:
            continue

    user_candidate = Path.home() / ".nsfwpy" / "models"
    try:
        user_candidate.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return user_candidate


# Default model directory cache
USER_CACHE_DIR = get_system_cache_dir()

# Hugging Face model download URLs
BASE_HF_URL = "https://huggingface.co/onnx-community/nsfw_image_detection-ONNX/resolve/main/onnx"

MODEL_DOWNLOAD_URLS: Dict[str, str] = {
    "nsfw_vit_quantized": f"{BASE_HF_URL}/model_quantized.onnx?download=true",
    "nsfw_vit": f"{BASE_HF_URL}/model.onnx?download=true",
    "nsfw_vit_fp16": f"{BASE_HF_URL}/model_fp16.onnx?download=true",
    "nsfw_vit_int8": f"{BASE_HF_URL}/model_int8.onnx?download=true",
    "nsfw_vit_uint8": f"{BASE_HF_URL}/model_uint8.onnx?download=true",
    "nsfw_vit_q4": f"{BASE_HF_URL}/model_q4.onnx?download=true",
    "nsfw_vit_q4f16": f"{BASE_HF_URL}/model_q4f16.onnx?download=true",
    "nsfw_vit_bnb4": f"{BASE_HF_URL}/model_bnb4.onnx?download=true",
}

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

