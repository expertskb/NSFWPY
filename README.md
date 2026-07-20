# NSFWPY 🔥

High-performance, CPU-optimized Python 3.14 port of **NSFWJS** using **ONNX Runtime**.

Classify images into 5 canonical NSFW categories (**Drawing**, **Hentai**, **Neutral**, **Porn**, **Sexy**) with fast inference, zero GPU requirement, a clean Python API, CLI tool, REST server, and sleek Web UI dashboard.

---

## Features

- 🐍 **Python 3.14 Native**: Fully compatible with Python 3.14+ and virtual environment `.venv`.
- ⚡ **CPU Optimized**: Powered by `onnxruntime` tuned for multi-threaded CPU execution with graph optimization.
- 📦 **Multiple Backbones**: Supports `MobileNetV2`, `MobileNetV3`, and `InceptionV3`.
- 🌐 **REST API & Web UI**: Built-in FastAPI server with Swagger docs and drag-and-drop web dashboard.
- 🛠️ **CLI Utility**: Classify images directly from your terminal.

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/nsfwpy.git
cd nsfwpy

# Create virtual environment (Python 3.14)
python3.14 -m venv .venv
source .venv/bin/activate

# Install nsfwpy library
pip install -e .
```

---

## Quick Usage

### 1. Python Library Usage

```python
import nsfwpy
from PIL import Image

# Load model (default: mobilenet_v2)
model = nsfwpy.load_model("mobilenet_v2")

# Classify single image (file path, bytes, URL, or PIL Image)
results = model.classify("path/to/image.jpg")
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
# Classify an image file or URL
nsfwpy classify path/to/image.jpg

# Output formatted JSON
nsfwpy classify path/to/image.jpg --json-out

# Specify model
nsfwpy classify https://example.com/test.jpg --model inception_v3
```

### 3. Start Web Dashboard & REST API

```bash
# Launch server on http://localhost:8000
nsfwpy serve --port 8000
```

- Open `http://localhost:8000` in your browser for the Web Dashboard.
- Open `http://localhost:8000/docs` for the interactive Swagger API documentation.

---

## License

MIT License
