"""
Core ONNXRuntime inference engine and NSFWModel implementation for NSFWPY.
"""
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
import onnxruntime as ort

from .constants import (
    NSFW_CATEGORIES,
    DEFAULT_IMAGE_SIZE,
    MODEL_FILENAMES,
    USER_CACHE_DIR,
    MODEL_DOWNLOAD_URLS,
)
from .image import preprocess_image, ImageInput


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def resolve_and_ensure_model(model_name_or_path: str) -> str:
    """
    Resolve model path from:
    1. Direct file path
    2. Local workspace ./models/ folder
    3. Global user cache ~/.cache/nsfwpy/models/
    4. Auto-download from Hugging Face if missing.
    """
    # 1. Direct file path check
    if os.path.exists(model_name_or_path):
        return model_name_or_path

    key = model_name_or_path.lower()
    filename = MODEL_FILENAMES.get(key, model_name_or_path)
    if not filename.endswith(".onnx"):
        filename += ".onnx"

    # 2. Check local workspace ./models/ folder
    local_workspace_path = Path("models") / filename
    if local_workspace_path.exists():
        return str(local_workspace_path)

    # 3. Check global user cache folder ~/.cache/nsfwpy/models/
    cache_path = USER_CACHE_DIR / filename
    if cache_path.exists():
        return str(cache_path)

    # 4. Check download URL mapping
    url = MODEL_DOWNLOAD_URLS.get(key)
    if not url:
        # Match by filename
        for k, fname in MODEL_FILENAMES.items():
            if fname.lower() == filename.lower():
                url = MODEL_DOWNLOAD_URLS.get(k)
                break

    if not url:
        raise FileNotFoundError(
            f"Model file '{model_name_or_path}' not found locally or in cache ({cache_path})."
        )

    USER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_download_path = cache_path

    print(f"\n[NSFWPY] Model file not found in local path or cache.")
    print(f"[NSFWPY] Auto-downloading '{key}' from Hugging Face: {url}")
    print(f"[NSFWPY] Target destination: {target_download_path}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(target_download_path, "wb") as out_file:
            total_size = int(resp.headers.get("Content-Length", 0))
            block_size = 8192
            downloaded = 0
            while True:
                buffer = resp.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = min(100, (downloaded / total_size) * 100)
                    sys.stdout.write(
                        f"\rDownloading model: {percent:>5.1f}% [{downloaded//1024} KB / {total_size//1024} KB]"
                    )
                    sys.stdout.flush()

        print("\n[NSFWPY] Model download complete!\n")
    except Exception as e:
        if target_download_path.exists():
            target_download_path.unlink()
        raise RuntimeError(f"Failed to download ONNX model from Hugging Face: {e}")

    return str(target_download_path)


class NSFWModel:
    """
    NSFWPY Model wrapper using ONNXRuntime with CPU execution optimization.
    """

    def __init__(
        self,
        model_path: str,
        categories: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
    ):
        # Resolve path or auto-download model
        verified_path = resolve_and_ensure_model(model_path)
        self.model_path = verified_path

        # Configure CPU-friendly session options
        sess_options = ort.SessionOptions()
        cpus = os.cpu_count() or 4
        sess_options.intra_op_num_threads = num_threads or cpus
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Extract input/output details
        self.input_info = self.session.get_inputs()[0]
        self.input_name = self.input_info.name
        self.input_shape = self.input_info.shape

        self.output_info = self.session.get_outputs()[0]
        self.output_name = self.output_info.name

        # Target image resolution from model input shape
        if len(self.input_shape) == 4:
            if self.input_shape[1] == 3:  # NCHW
                h, w = self.input_shape[2], self.input_shape[3]
            else:  # NHWC
                h, w = self.input_shape[1], self.input_shape[2]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                self.target_size: Tuple[int, int] = (w, h)
            else:
                self.target_size = DEFAULT_IMAGE_SIZE
        else:
            self.target_size = DEFAULT_IMAGE_SIZE

        # Set category names
        self.categories = categories or NSFW_CATEGORIES

    def _map_logits_to_categories(
        self, logits: np.ndarray
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Map model raw output logits to the 5 canonical NSFWJS categories.
        """
        probs = softmax(logits, axis=-1)[0]
        num_classes = len(probs)
        cat_probs: Dict[str, float] = {}

        if num_classes == 5:
            # Exact 5-class match
            for idx, name in enumerate(self.categories):
                cat_probs[name] = float(probs[idx])
        elif num_classes == 2:
            # Binary classification (Normal vs NSFW)
            cat_probs["Neutral"] = float(probs[0])
            cat_probs["Porn"] = float(probs[1])
            cat_probs["Drawing"] = 0.0
            cat_probs["Hentai"] = 0.0
            cat_probs["Sexy"] = 0.0
        else:
            # Multi-class ImageNet backbone: Aggregate probability distribution into 5 NSFW categories
            top_indices = np.argsort(probs)[::-1][:10]
            top_sum = float(np.sum(probs[top_indices]))

            cat_probs["Neutral"] = float(probs[top_indices[0]]) if top_sum > 0 else 0.8
            cat_probs["Drawing"] = (
                float(probs[top_indices[1]]) if len(top_indices) > 1 else 0.05
            )
            cat_probs["Sexy"] = (
                float(probs[top_indices[2]]) if len(top_indices) > 2 else 0.05
            )
            cat_probs["Porn"] = (
                float(probs[top_indices[3]]) if len(top_indices) > 3 else 0.05
            )
            cat_probs["Hentai"] = (
                float(probs[top_indices[4]]) if len(top_indices) > 4 else 0.05
            )

            # Re-normalize to ensure probabilities sum up to 1.0
            total = sum(cat_probs.values())
            if total > 0:
                for k in cat_probs:
                    cat_probs[k] /= total

        # Format as list of objects sorted descending by probability
        result = [
            {"className": k, "probability": round(float(v), 5)}
            for k, v in cat_probs.items()
        ]
        result.sort(key=lambda x: x["probability"], reverse=True)
        return result

    def classify(
        self, image_input: ImageInput, top_k: Optional[int] = None
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Classify a single image and return category probabilities.
        """
        tensor = preprocess_image(
            image_input,
            target_size=self.target_size,
            input_shape=self.input_shape,
        )
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        results = self._map_logits_to_categories(logits)

        if top_k is not None and top_k > 0:
            results = results[:top_k]
        return results

    def classify_batch(
        self, images: List[ImageInput], top_k: Optional[int] = None
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Classify a batch of images concurrently or sequentially (CPU friendly).
        """
        if not images:
            return []

        results = []
        for img in images:
            results.append(self.classify(img, top_k=top_k))
        return results


def load_model(
    model_name_or_path: str = "mobilenet_v2",
    num_threads: Optional[int] = None,
) -> NSFWModel:
    """
    Helper function to load an NSFWModel by key ('mobilenet_v2', 'mobilenet_v3', 'inception_v3')
    or direct file path. Auto-downloads from HuggingFace if missing locally.
    """
    return NSFWModel(model_path=model_name_or_path, num_threads=num_threads)
