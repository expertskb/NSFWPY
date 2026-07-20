# NSFWPY 🔥 (`nsfwpy-onnx`)

[![PyPI Version](https://img.shields.io/pypi/v/nsfwpy-onnx?color=blue&label=PyPI%20version)](https://pypi.org/project/nsfwpy-onnx/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/nsfwpy-onnx?color=green)](https://pypi.org/project/nsfwpy-onnx/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-expertskb%2Fnsfwpy--onnx-blue?logo=github)](https://github.com/expertskb/nsfwpy-onnx)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-expertskb%2Fnsfwpy-yellow)](https://huggingface.co/expertskb/nsfwpy)

High-performance, CPU-optimized Python 3.14 port of **NSFWJS** using **ONNX Runtime**.

Classify static images (JPEG, PNG, WEBP) and **Animated WebP / GIF / APNG** formats into 5 canonical NSFW categories (**Drawing**, **Hentai**, **Neutral**, **Porn**, **Sexy**).

---

## 🌟 Key Features

- 🐍 **Python 3.14 Native**: Fully compatible with Python 3.10+ and Python 3.14.
- 🎞️ **Animated WebP & GIF Support**: Keyframe sampling and frame-aggregated safety classification for Animated WebP, GIF, and APNG formats.
- ⚡ **CPU Optimized**: Powered by `onnxruntime` tuned for multi-threaded CPU execution with graph optimization.
- 📦 **HuggingFace Auto-Download**: Automatically downloads missing ONNX models from [Hugging Face (`expertskb/nsfwpy`)](https://huggingface.co/expertskb/nsfwpy).
- 🪶 **Ultra Lightweight**: Minimal dependencies (`numpy`, `pillow`, `onnxruntime`, `click`). No heavy web servers required.
- 🛠️ **CLI Tool Included**: Classify images directly from your terminal.

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

# Load model (auto-downloads from HuggingFace if missing)
model = nsfwpy.load_model("mobilenet_v2")

# Classify static image or animated WebP / GIF
results = model.classify("sample.jpg", top_k=5)
print(results)
# Output:
# [
#   {'className': 'Neutral', 'probability': 0.6280},
#   {'className': 'Drawing', 'probability': 0.2822},
#   {'className': 'Sexy', 'probability': 0.0548},
#   {'className': 'Porn', 'probability': 0.0246},
#   {'className': 'Hentai', 'probability': 0.0104}
# ]

# Batch image classification
batch_results = model.classify_batch(["img1.jpg", "anim2.gif"])
```

### 2. Command Line Interface (CLI)

```bash
# Classify a local image file or URL
nsfwpy classify path/to/image.jpg

# Output formatted JSON
nsfwpy classify sample.jpg --json-out

# Choose a specific model architecture
nsfwpy classify sample.jpg --model inception_v3
```

---

## 🧠 Supported Models

| Model Name | Backbone | Accuracy | Size | Auto-Download Link |
| :--- | :--- | :--- | :--- | :--- |
| `mobilenet_v2` (Default) | MobileNet V2 | Fast | ~14 MB | [HF Download](https://huggingface.co/expertskb/nsfwpy/resolve/main/MobileNet-v2.onnx?download=true) |
| `mobilenet_v3` | MobileNet V3 | Balanced | ~22 MB | [HF Download](https://huggingface.co/expertskb/nsfwpy/resolve/main/MobileNet-v3.onnx?download=true) |
| `inception_v3` | Inception V3 | High Precision | ~95 MB | [HF Download](https://huggingface.co/expertskb/nsfwpy/resolve/main/Inception-v3.onnx?download=true) |

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

---

## 📜 License

MIT License
