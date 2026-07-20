# NSFWPY 🔥 (`nsfwpy-onnx` v1.1.2)

[![PyPI Version](https://img.shields.io/pypi/v/nsfwpy-onnx?color=blue&label=PyPI%20version%201.1.2)](https://pypi.org/project/nsfwpy-onnx/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/nsfwpy-onnx?color=green)](https://pypi.org/project/nsfwpy-onnx/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-expertskb%2Fnsfwpy--onnx-blue?logo=github)](https://github.com/expertskb/nsfwpy-onnx)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-expertskb%2Fnsfwpy-yellow)](https://huggingface.co/expertskb/nsfwpy)

High-performance, CPU-optimized Python NSFW image safety classification library powered by state-of-the-art Vision Transformer (ViT) models from **Falcons AI** and the **ONNX Community** using **ONNX Runtime**.

Classify static images (JPEG, PNG, WEBP) and **Animated WebP / GIF / APNG** formats into 5 canonical NSFW safety categories (**Drawing**, **Hentai**, **Neutral**, **Porn**, **Sexy**).

---

## 🌟 Key Features

- 🐍 **Python 3.14 Native**: Fully compatible with Python 3.10+ and Python 3.14.
- 🎞️ **Animated WebP & GIF Support**: Keyframe sampling and frame-aggregated safety classification for Animated WebP, GIF, and APNG formats.
- ⚡ **CPU Optimized**: Powered by `onnxruntime` tuned for multi-threaded CPU execution with graph optimization.
- 📦 **HuggingFace Auto-Download**: Automatically downloads missing ONNX models from [Hugging Face (`expertskb/nsfwpy`)](https://huggingface.co/expertskb/nsfwpy).
- 🪶 **Ultra Lightweight**: Minimal dependencies (`numpy`, `pillow`, `onnxruntime`, `click`). No heavy web servers required.
- 🛠️ **CLI Tool Included**: Classify images directly from your terminal.

---

## ⚡ Why `nsfwpy-onnx` is Superior to Legacy Classifiers

| Feature / Metric | `nsfwpy-onnx` 🔥 | Legacy CNN Classifiers (MobileNet/Inception) 🐢 | Heavy PyTorch / TensorFlow Libs 📦 |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | Modern **Vision Transformer (ViT)** | Legacy MobileNet V2 / Inception V3 (CNN) | Older MobileNet V2 |
| **Accuracy & False Positives** | **High** (Understands visual context) | Medium (High false positives on drawings/art) | Medium |
| **Framework Overhead** | Light C++ ONNX Runtime (~150MB RAM) | Heavy TensorFlow / TFJS-Node (500MB+ RAM) | Heavy PyTorch / TensorFlow |
| **Startup Preloading (`nsfwpy.preload()`)** | ✅ **Pre-warmed CPU execution & graph** | ❌ Manual warmup script required | ❌ Cold start delays on first request |
| **Animated WebP & GIF** | ✅ Built-in keyframe sampling & scanning | ❌ Requires manual frame splitting | ❌ Static images only |
| **System Cache & Recovery** | ✅ **Full Cross-Platform** (Linux, Windows, macOS, Android/Termux, iOS/Tablet) + auto-recovery | ❌ Basic local cache | ❌ Manual download required |

---

### 🖥️ Cross-Platform System Model Caching

`nsfwpy` automatically detects host OS and environment permissions to cache models in optimal system directories:

- **Linux / Servers**: `/etc/nsfwpy/models` or `/var/cache/nsfwpy/models`
- **Windows**: `C:\nsfwpy\models` or `C:\ProgramData\nsfwpy\models`
- **macOS**: `/Library/Caches/nsfwpy/models` or `~/Library/Caches/nsfwpy/models`
- **Android / Termux / Mobile**: `/sdcard/.nsfwpy/models` or `$PREFIX/tmp/nsfwpy/models`
- **iOS / iPadOS / Tablets**: `~/Documents/.nsfwpy/models`
- **Automatic Fallback**: `~/.nsfwpy/models` (if running as a non-root standard user without system write access)

---

## 📊 Performance & Preload Benchmarks

Tested on CPU (Intel/AMD multi-thread execution):

```python
import nsfwpy

# Warm up model graph and CPU execution pools at app startup
nsfwpy.preload()
```

| Benchmark Metric | Latency / Throughput | Description |
| :--- | :--- | :--- |
| **Preload & Warmup (`nsfwpy.preload()`)** | **~370 ms** | Downloads/loads model & compiles ONNX graph |
| **Warmed-Up Inference Latency** | **~270 ms / image** | Instant classification with zero cold-start delay |
| **Batch Processing Throughput** | **~4.5 - 5.0 imgs/sec** | High-throughput batch classification on CPU |
| **Remote WebP Single Frame (HTTP + Inf)** | **~700 ms** | Network fetch + image decode + ONNX inference |
| **Remote WebP Full Animation (10 keyframes)** | **~2.3s** | Complete frame-aggregated animation safety scan |

---

## 📥 Installation

Install directly from **PyPI**:

```bash
pip install nsfwpy-onnx
```

Or install from GitHub:

```bash
git clone https://github.com/expertskb/nsfwpy-onnx.git
cd nsfwpy-onnx
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 🚀 Quick Usage

### 1. Python Library Usage

```python
import nsfwpy

# Load default quantized ViT model (auto-downloads if missing)
model = nsfwpy.load_model()

# Classify static image or animated WebP / GIF
results = model.classify("sample.jpg", top_k=5)
print(results)
# Output:
# [
#   {'className': 'Neutral', 'probability': 0.9850},
#   {'className': 'Porn', 'probability': 0.0150}
# ]

# Batch image classification
batch_results = model.classify_batch(["img1.jpg", "anim2.gif"])
```

### 2. Command Line Interface (CLI)

```bash
# Classify a local image file or URL using default nsfw_vit_quantized model
nsfwpy classify path/to/image.jpg

# Output formatted JSON
nsfwpy classify sample.jpg --json-out

# Choose full precision nsfw_vit model
nsfwpy classify sample.jpg --model nsfw_vit
```

---

## 🧠 Supported Models

Powered by [`onnx-community/nsfw_image_detection-ONNX`](https://huggingface.co/onnx-community/nsfw_image_detection-ONNX):

| Model Name | Format | Precision | Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `nsfw_vit_quantized` (Default) / `nsfw_vit_int8` | INT8 | 8-bit Quantized | ~87 MB | **Default model**: Best balance of speed, size, and accuracy |
| `nsfw_vit` | FP32 | Full Precision | ~343 MB | Original high-precision model (highest accuracy) |
| `nsfw_vit_fp16` | FP16 | Half Precision | ~172 MB | 2x faster, 50% smaller |
| `nsfw_vit_q4` | Q4 | 4-bit Quantized | ~56 MB | Ultra-lightweight for constrained memory |
| `nsfw_vit_q4f16` | Q4F16 | Mixed 4/16-bit | ~50 MB | Smallest model size |
| `nsfw_vit_bnb4` | BNB4 | BitsAndBytes 4-bit | ~51 MB | BitsAndBytes 4-bit quantized |
| `nsfw_vit_uint8` | UINT8 | Unsigned 8-bit | ~87 MB | Optimized for CPU hardware with uint8 support |

---

## 📂 Example Scripts Index

Professional production-ready scripts are included under [`examples/`](file:///home/shakib/Workspace/IMAGE%20DETECTOR/examples):

- **[`examples/01_basic_classification.py`](file:///home/shakib/Workspace/IMAGE%20DETECTOR/examples/01_basic_classification.py)**: Local image classification.
- **[`examples/02_url_classification.py`](file:///home/shakib/Workspace/IMAGE%20DETECTOR/examples/02_url_classification.py)**: Remote image URL classification.
- **[`examples/03_batch_classification.py`](file:///home/shakib/Workspace/IMAGE%20DETECTOR/examples/03_batch_classification.py)**: High-throughput batch benchmarking.
- **[`examples/04_pil_and_bytes_classification.py`](file:///home/shakib/Workspace/IMAGE%20DETECTOR/examples/04_pil_and_bytes_classification.py)**: In-memory PIL and bytes buffer processing.
- **[`examples/05_custom_model_and_threading.py`](file:///home/shakib/Workspace/IMAGE%20DETECTOR/examples/05_custom_model_and_threading.py)**: Model backbones & ONNX CPU thread tuning.
- **[`examples/06_animated_webp_gif_classification.py`](file:///home/shakib/Workspace/IMAGE%20DETECTOR/examples/06_animated_webp_gif_classification.py)**: Animated WebP & GIF frame-aggregated classification.

Run any example using:
```bash
python examples/01_basic_classification.py
```

## 🙏 Credits & Acknowledgments

Special thanks to the open-source AI community and model creators:

- **[Falcons AI (`falconsai`)](https://huggingface.co/falconsai)**: Creators of the high-accuracy Vision Transformer (ViT) model [`falconsai/nsfw_image_detection`](https://huggingface.co/falconsai/nsfw_image_detection) used as the core engine.
- **[ONNX Community (`onnx-community`)](https://huggingface.co/onnx-community)**: For converting, quantizing, and hosting the ONNX model suite at [`onnx-community/nsfw_image_detection-ONNX`](https://huggingface.co/onnx-community/nsfw_image_detection-ONNX).

---

## 📜 License

MIT License
