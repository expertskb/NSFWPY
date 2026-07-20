"""
Core ONNXRuntime inference engine and NSFWModel implementation for NSFWPY.
Supports static images and animated WebP, GIF, and APNG formats.
"""
import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
import numpy as np
import onnxruntime as ort

from .constants import (
    NSFW_CATEGORIES,
    DEFAULT_IMAGE_SIZE,
    MODEL_FILENAMES,
    USER_CACHE_DIR,
    MODEL_DOWNLOAD_URLS,
)
from .image import (
    preprocess_image,
    load_animated_frames,
    ImageInput,
)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def resolve_and_ensure_model(
    model_name_or_path: str, force_redownload: bool = False
) -> str:
    """
    Resolve model path from:
    1. Direct file path
    2. Local workspace ./models/ folder
    3. Global user cache ~/.cache/nsfwpy/models/
    4. Auto-download from Hugging Face if missing or corrupted.
    """
    if os.path.exists(model_name_or_path) and not force_redownload:
        if os.path.getsize(model_name_or_path) > 0:
            return model_name_or_path

    key = model_name_or_path.lower()
    filename = MODEL_FILENAMES.get(key, model_name_or_path)
    if not filename.endswith(".onnx"):
        filename += ".onnx"

    # Search list of candidate cache paths:
    system_drive = os.environ.get("SystemDrive", "C:")
    candidate_paths = [
        USER_CACHE_DIR / filename,
        Path("models") / filename,
        Path(system_drive) / "nsfwpy" / "models" / filename,
        Path("/var/cache/nsfwpy/models") / filename,
        Path("/etc/nsfwpy/models") / filename,
        Path.home() / ".nsfwpy" / "models" / filename,
        Path.home() / ".cache" / "nsfwpy" / "models" / filename,
    ]

    for cand in candidate_paths:
        if cand.exists() and not force_redownload and cand.stat().st_size > 0:
            return str(cand)

    url = MODEL_DOWNLOAD_URLS.get(key)
    if not url:
        for k, fname in MODEL_FILENAMES.items():
            if fname.lower() == filename.lower():
                url = MODEL_DOWNLOAD_URLS.get(k)
                break

    target_dir = USER_CACHE_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        target_dir = Path.home() / ".nsfwpy" / "models"
        target_dir.mkdir(parents=True, exist_ok=True)

    target_download_path = target_dir / filename
    temp_download_path = target_dir / f"{filename}.tmp"

    if not url:
        raise FileNotFoundError(
            f"Model file '{model_name_or_path}' not found locally or in cache ({target_download_path})."
        )

    if target_download_path.exists():
        try:
            target_download_path.unlink()
        except Exception:
            pass

    if temp_download_path.exists():
        try:
            temp_download_path.unlink()
        except Exception:
            pass

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(temp_download_path, "wb") as out_file:
            total_size = int(resp.headers.get("Content-Length", 0))
            block_size = 65536
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
                        f"\rDownloading model '{filename}': {percent:>5.1f}% [{downloaded//1024} KB / {total_size//1024} KB]"
                    )
                    sys.stdout.flush()

        if sys.stdout.isatty():
            sys.stdout.write("\n")
            sys.stdout.flush()

        if total_size > 0 and temp_download_path.stat().st_size < (total_size * 0.95):
            raise RuntimeError(
                f"Incomplete download: Expected {total_size} bytes, got {temp_download_path.stat().st_size} bytes."
            )

        temp_download_path.replace(target_download_path)

    except Exception as e:
        if temp_download_path.exists():
            try:
                temp_download_path.unlink()
            except Exception:
                pass
        if target_download_path.exists():
            try:
                target_download_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Failed to download ONNX model from Hugging Face ({url}): {e}")

    return str(target_download_path)


class NSFWModel:
    """
    NSFWPY Model wrapper using ONNXRuntime with CPU execution optimization.
    Supports static & animated WebP, GIF, and APNG images.
    """

    def __init__(
        self,
        model_path: str,
        categories: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
        warmup: bool = True,
    ):
        verified_path = resolve_and_ensure_model(model_path)
        self.model_path = verified_path

        def _create_session(path: str) -> ort.InferenceSession:
            cpus = os.cpu_count() or 4
            opt_levels = [
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            ]
            last_err = None

            for level in opt_levels:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = num_threads or cpus
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.graph_optimization_level = level
                sess_options.log_severity_level = 3  # Silence ONNXRuntime internal C++ logs

                try:
                    return ort.InferenceSession(
                        path,
                        sess_options=sess_options,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as e:
                    last_err = e
                    continue

            raise last_err or RuntimeError(f"Failed to initialize ONNX session for {path}")

        try:
            self.session = _create_session(self.model_path)
        except Exception:
            # File might be corrupted or truncated; force redownload once and retry
            redownloaded_path = resolve_and_ensure_model(model_path, force_redownload=True)
            self.model_path = redownloaded_path
            self.session = _create_session(self.model_path)

        self.input_info = self.session.get_inputs()[0]
        self.input_name = self.input_info.name
        self.input_shape = self.input_info.shape

        self.output_info = self.session.get_outputs()[0]
        self.output_name = self.output_info.name

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

        self.categories = categories or NSFW_CATEGORIES

        if warmup:
            self.warmup()

    def warmup(self) -> None:
        """
        Pre-allocate ONNXRuntime execution memory buffers and perform graph compilation
        by running a single dummy inference pass.
        """
        try:
            from PIL import Image
            dummy_img = Image.new("RGB", self.target_size, color=0)
            tensor = preprocess_image(
                dummy_img,
                target_size=self.target_size,
                input_shape=self.input_shape,
            )
            _ = self.session.run([self.output_name], {self.input_name: tensor})
        except Exception:
            pass

    def _map_logits_to_categories(
        self, logits: np.ndarray
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Map model raw output logits to the 5 canonical safety classification categories.
        """
        probs = softmax(logits, axis=-1)[0]
        num_classes = len(probs)
        cat_probs: Dict[str, float] = {}

        if num_classes == 5:
            for idx, name in enumerate(self.categories):
                cat_probs[name] = float(probs[idx])
        elif num_classes == 2:
            cat_probs["Neutral"] = float(probs[0])
            cat_probs["Porn"] = float(probs[1])
            cat_probs["Drawing"] = 0.0
            cat_probs["Hentai"] = 0.0
            cat_probs["Sexy"] = 0.0
        else:
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

            total = sum(cat_probs.values())
            if total > 0:
                for k in cat_probs:
                    cat_probs[k] /= total

        result = [
            {"className": k, "probability": round(float(v), 5)}
            for k, v in cat_probs.items()
        ]
        result.sort(key=lambda x: x["probability"], reverse=True)
        return result

    def classify(
        self,
        image_input: Any,
        top_k: Optional[int] = None,
        max_animated_frames: int = 10,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Classify a single image or animated WebP/GIF/APNG.
        If the image is animated, extracts keyframes and aggregates max probabilities across frames.
        """
        frames = load_animated_frames(image_input, max_frames=max_animated_frames)

        if len(frames) == 1:
            tensor = preprocess_image(
                frames[0],
                target_size=self.target_size,
                input_shape=self.input_shape,
            )
            logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
            results = self._map_logits_to_categories(logits)
        else:
            # Animated image: aggregate frame predictions by taking max probability per category
            category_max_probs: Dict[str, float] = {}

            for frame in frames:
                tensor = preprocess_image(
                    frame,
                    target_size=self.target_size,
                    input_shape=self.input_shape,
                )
                logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
                frame_preds = self._map_logits_to_categories(logits)

                for item in frame_preds:
                    cat = item["className"]
                    prob = item["probability"]
                    if cat not in category_max_probs or prob > category_max_probs[cat]:
                        category_max_probs[cat] = prob

            # Re-normalize aggregated max probabilities
            total = sum(category_max_probs.values())
            if total > 0:
                for k in category_max_probs:
                    category_max_probs[k] /= total

            results = [
                {"className": k, "probability": round(float(v), 5)}
                for k, v in category_max_probs.items()
            ]
            results.sort(key=lambda x: x["probability"], reverse=True)

        if top_k is not None and top_k > 0:
            results = results[:top_k]
        return results

    def classify_gif(
        self,
        image_input: Any,
        top_k: Optional[int] = None,
        max_frames: int = 10,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Alias method specifically for animated WebP / GIF / APNG classification.
        """
        return self.classify(
            image_input=image_input,
            top_k=top_k,
            max_animated_frames=max_frames,
        )

    def classify_batch(
        self, images: List[Any], top_k: Optional[int] = None
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


_MODEL_CACHE: Dict[Tuple[str, Optional[int]], NSFWModel] = {}


def load_model(
    model_name_or_path: str = "nsfw_vit_quantized",
    num_threads: Optional[int] = None,
    use_cache: bool = True,
    warmup: bool = True,
) -> NSFWModel:
    """
    Helper function to load an NSFWModel by key ('nsfw_vit_quantized', 'nsfw_vit', etc.)
    or direct file path. Auto-downloads from HuggingFace if missing locally.
    Caches loaded model instances in memory to avoid repeated ONNX parsing overhead.
    """
    cache_key = (model_name_or_path, num_threads)
    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = NSFWModel(
        model_path=model_name_or_path, num_threads=num_threads, warmup=warmup
    )
    if use_cache:
        _MODEL_CACHE[cache_key] = model
    return model


def preload_model(
    model_name_or_path: str = "nsfw_vit_quantized",
    num_threads: Optional[int] = None,
) -> NSFWModel:
    """
    Preload and warm up an NSFWModel into memory.
    Downloads/resolves the model, compiles ONNX execution graph,
    pre-allocates CPU memory buffers, and caches the model instance.
    Subsequent classifications will have zero cold-start delay and minimal CPU overhead.
    """
    return load_model(
        model_name_or_path=model_name_or_path,
        num_threads=num_threads,
        use_cache=True,
        warmup=True,
    )


load = load_model
preload = preload_model
