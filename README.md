# NSFWPY 🔥

[![PyPI Version](https://img.shields.io/pypi/v/nsfwpy.svg?color=blue)](https://pypi.org/project/nsfwpy/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-expertskb%2Fnsfwpy-yellow)](https://huggingface.co/expertskb/nsfwpy)

High-performance, CPU-optimized Python 3.14 port of **NSFWJS** using **ONNX Runtime**.

Classify images into 5 canonical NSFW categories (**Drawing**, **Hentai**, **Neutral**, **Porn**, **Sexy**) with fast CPU inference, zero GPU requirement, HuggingFace auto-downloading, and a clean Python API + CLI.

---

## 🌟 Key Features

- 🐍 **Python 3.14 Native**: Fully compatible with Python 3.10+ and Python 3.14.
- ⚡ **CPU Optimized**: Powered by `onnxruntime` tuned for multi-threaded CPU execution with graph optimization.
- 📦 **HuggingFace Auto-Download**: Automatically downloads missing ONNX models from [Hugging Face (`expertskb/nsfwpy`)](https://huggingface.co/expertskb/nsfwpy).
- 🪶 **Ultra Lightweight**: Minimal dependencies (`numpy`, `pillow`, `onnxruntime`, `click`). No heavy web servers required.
- 🛠️ **CLI Tool Included**: Classify images directly from your terminal.

---

## 📥 Installation

```bash
# Install from PyPI
pip install nsfwpy
```

Or install locally for development:
```bash
git clone https://github.com/your-username/nsfwpy.git
cd nsfwpy
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

# Classify single image (File path, URL, bytes, or PIL Image)
results = model.classify("path/to/image.jpg", top_k=5)
print(results)
# Output:
# [
#   {'className': 'Neutral', 'probability': 0.85231},
#   {'className': 'Drawing', 'probability': 0.08412},
#   {'className': 'Sexy', 'probability': 0.04123},
#   {'className': 'Porn', 'probability': 0.01211},
#   {'className': 'Hentai', 'probability': 0.01023}
# ]

# Batch image classification
batch_results = model.classify_batch(["img1.jpg", "img2.png"])
```

### 2. Command Line Interface (CLI)

```bash
# Classify a local image file or URL
nsfwpy classify path/to/image.jpg

# Output formatted JSON
nsfwpy classify https://example.com/sample.jpg --json-out

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

- **[`examples/01_basic_classification.py`](examples/01_basic_classification.py)**: Local image classification.
- **[`examples/02_url_classification.py`](examples/02_url_classification.py)**: Remote image URL classification.
- **[`examples/03_batch_classification.py`](examples/03_batch_classification.py)**: High-throughput batch benchmarking.
- **[`examples/04_pil_and_bytes_classification.py`](examples/04_pil_and_bytes_classification.py)**: In-memory PIL and bytes buffer processing.
- **[`examples/05_custom_model_and_threading.py`](examples/05_custom_model_and_threading.py)**: Model backbones & ONNX CPU thread tuning.

Run any example using:
```bash
python examples/01_basic_classification.py
```

---

## 📜 License

MIT License
