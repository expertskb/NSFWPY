"""
Image loading and CPU-optimized preprocessing utilities for NSFWPY.
Supports static images (JPEG, PNG, WEBP) and animated images (Animated WEBP, GIF, APNG).
"""
import io
import urllib.request
from pathlib import Path
from typing import Union, Tuple, List, Any
import numpy as np
from PIL import Image, ImageSequence

from .constants import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

ImageInput = Union[str, Path, bytes, Image.Image, np.ndarray]


def load_image(image_input: Any) -> Image.Image:
    """
    Load any image input (path, URL, bytes, PIL Image, or NumPy array) into a PIL RGB Image.
    For animated images, returns the first keyframe (or primary frame).
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 2:  # Grayscale
            img = Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3:
            if arr.shape[2] == 4:  # RGBA
                img = Image.fromarray(arr, mode="RGBA")
            elif arr.shape[2] == 3:  # RGB / BGR
                img = Image.fromarray(arr, mode="RGB")
            else:
                img = Image.fromarray(arr)
        else:
            raise ValueError(f"Unsupported NumPy array shape: {arr.shape}")
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, Path)):
        src = str(image_input)
        if src.startswith(("http://", "https://")):
            req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data))
        else:
            img = Image.open(src)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Move to first frame if animated
    if getattr(img, "is_animated", False):
        img.seek(0)

    # Handle transparency / Alpha channel by alpha matting over white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_img = img.convert("RGBA")
        background = Image.new("RGBA", alpha_img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, alpha_img)
        return composite.convert("RGB")

    return img.convert("RGB")


def load_animated_frames(image_input: Any, max_frames: int = 10) -> List[Image.Image]:
    """
    Extract keyframes from an animated image (Animated WebP, GIF, APNG).
    If static, returns a single-item list.
    """
    if isinstance(image_input, Image.Image):
        raw_img = image_input
    elif isinstance(image_input, bytes):
        raw_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, Path)):
        src = str(image_input)
        if src.startswith(("http://", "https://")):
            req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            raw_img = Image.open(io.BytesIO(data))
        else:
            raw_img = Image.open(src)
    else:
        # Single static frame
        return [load_image(image_input)]

    if not getattr(raw_img, "is_animated", False) or getattr(raw_img, "n_frames", 1) <= 1:
        return [load_image(raw_img)]

    n_frames = raw_img.n_frames
    # Sample up to max_frames evenly across the animation duration
    step = max(1, n_frames // max_frames)
    frame_indices = list(range(0, n_frames, step))[:max_frames]

    frames = []
    for idx in frame_indices:
        raw_img.seek(idx)
        frame_copy = raw_img.copy()
        if frame_copy.mode in ("RGBA", "LA") or (frame_copy.mode == "P" and "transparency" in frame_copy.info):
            alpha_img = frame_copy.convert("RGBA")
            background = Image.new("RGBA", alpha_img.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(background, alpha_img)
            frames.append(composite.convert("RGB"))
        else:
            frames.append(frame_copy.convert("RGB"))

    return frames


def preprocess_image(
    image_input: Any,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    input_shape: Union[List[int], Tuple[int, ...], None] = None,
    normalize_type: str = "imagenet",
) -> np.ndarray:
    """
    Preprocess an image for ONNX model inference.
    Supports NCHW [1, 3, H, W] and NHWC [1, H, W, 3] layout formats.
    """
    img = load_image(image_input)
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32)

    # Apply normalization
    if normalize_type == "tf":
        arr = (arr / 127.5) - 1.0
    elif normalize_type == "scale":
        arr = arr / 255.0
    else:
        arr = arr / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std

    is_nchw = True
    if input_shape is not None and len(input_shape) == 4:
        if input_shape[1] != 3 and input_shape[3] == 3:
            is_nchw = False

    if is_nchw:
        arr = np.transpose(arr, (2, 0, 1))

    return np.expand_dims(arr, axis=0)


def preprocess_batch(
    images: List[Any],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    input_shape: Union[List[int], Tuple[int, ...], None] = None,
    normalize_type: str = "imagenet",
) -> np.ndarray:
    """
    Preprocess a list of images into a single batch NumPy tensor.
    """
    preprocessed = [
        preprocess_image(img, target_size, input_shape, normalize_type) for img in images
    ]
    return np.vstack(preprocessed)
