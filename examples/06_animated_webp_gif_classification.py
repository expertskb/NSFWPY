"""
Example 06: Animated WebP & GIF Classification
-----------------------------------------------
Demonstrates how to classify animated WebP, GIF, and APNG images by sampling
keyframes across the animation duration.
"""

import io
import numpy as np
from PIL import Image
import nsfwpy

def main():
    print("=" * 60)
    print(" NSFWPY - Example 06: Animated WebP / GIF Classification")
    print("=" * 60)

    # 1. Load Model
    model = nsfwpy.load_model("mobilenet_v2")

    # 2. Create an in-memory synthetic animated WebP image (5 frames)
    print("\nCreating synthetic animated WebP image in memory...")
    frames = [
        Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        for _ in range(5)
    ]
    animated_bytes = io.BytesIO()
    frames[0].save(
        animated_bytes,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    raw_animated_data = animated_bytes.getvalue()

    # 3. Classify animated image keyframes
    print("Classifying Animated WebP keyframes...")
    results = model.classify(raw_animated_data, top_k=5, max_animated_frames=5)

    print("\nAnimated WebP Keyframe Aggregated Breakdown:")
    print("-" * 50)
    for p in results:
        category = p["className"]
        prob = p["probability"]
        percentage = prob * 100
        bar = "█" * int(prob * 25)
        print(f"  {category:<10} | {percentage:>6.2f}% | {bar}")
    print("-" * 50)

if __name__ == "__main__":
    main()
