"""
Example 01: Basic Image Classification
---------------------------------------
Demonstrates how to load the default model and classify a single image from a local file path.
"""

from pathlib import Path
import nsfwpy

def main():
    print("=" * 60)
    print(" NSFWPY - Example 01: Basic Local Image Classification")
    print("=" * 60)

    # 1. Load the pre-trained MobileNetV2 model
    print("\n[1/3] Loading NSFWPY model (MobileNetV2)...")
    model = nsfwpy.load_model("mobilenet_v2")
    print("✓ Model loaded successfully!")

    # 2. Specify target image path (using sample or synthetic)
    sample_path = Path("sample.jpg")

    # If sample doesn't exist, create a demo test image
    if not sample_path.exists():
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        img.save(sample_path)
        print(f"Created sample image at: {sample_path}")

    # 3. Classify the image
    print(f"\n[2/3] Classifying image: {sample_path}...")
    results = model.classify(sample_path, top_k=5)

    # 4. Print structured results
    print("\n[3/3] Classification Results:")
    print("-" * 45)
    print(f"{'Category':<12} | {'Probability':<12} | Score Bar")
    print("-" * 45)

    for item in results:
        category = item["className"]
        prob = item["probability"]
        percentage = prob * 100
        bar = "█" * int(prob * 20)
        print(f"{category:<12} | {percentage:>6.2f}%       | {bar}")

    print("-" * 45)

if __name__ == "__main__":
    main()
