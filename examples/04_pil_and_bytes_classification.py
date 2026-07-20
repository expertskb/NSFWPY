"""
Example 04: In-Memory PIL & Raw Bytes Classification
------------------------------------------------------
Demonstrates how to classify raw image bytes (e.g. from web requests or database streams)
and PIL Image objects directly in memory without saving to disk.
"""

import io
from PIL import Image, ImageDraw
import nsfwpy

def main():
    print("=" * 60)
    print(" NSFWPY - Example 04: In-Memory PIL & Bytes Processing")
    print("=" * 60)

    model = nsfwpy.load_model()

    # 1. Create a PIL Image dynamically in memory
    print("\n[1] Creating synthetic PIL Image in memory...")
    pil_image = Image.new("RGB", (300, 300), color=(73, 109, 137))
    draw = ImageDraw.Draw(pil_image)
    draw.rectangle([50, 50, 250, 250], fill=(255, 255, 255))

    # Classify PIL Image directly
    pil_results = model.classify(pil_image, top_k=3)
    print("✓ PIL Image Prediction:")
    for res in pil_results:
        print(f"  • {res['className']:<10}: {res['probability']*100:.2f}%")

    # 2. Convert PIL Image to raw JPEG byte buffer (simulating web upload)
    print("\n[2] Converting PIL Image to raw Bytes buffer...")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    raw_bytes = buffer.getvalue()

    # Classify raw bytes directly
    bytes_results = model.classify(raw_bytes, top_k=3)
    print("✓ Raw Bytes Prediction:")
    for res in bytes_results:
        print(f"  • {res['className']:<10}: {res['probability']*100:.2f}%")

if __name__ == "__main__":
    main()
