"""
Example 06: Animated WebP & GIF Classification
-----------------------------------------------
Demonstrates how to classify animated WebP, GIF, and APNG images by sampling
keyframes across the animation duration.
"""

import nsfwpy

def main():
    print("=" * 60)
    print(" NSFWPY - Example 06: Animated WebP / GIF Classification")
    print("=" * 60)

    # Load Model
    model = nsfwpy.load_model()

    # Animated WebP URL
    animated_url = "https://example.com/hello.webp"
    print(f"\nClassifying Animated WebP URL:\n{animated_url}\n")

    # Classify animated image keyframes
    results = model.classify(animated_url, top_k=5, max_animated_frames=10)

    print("Animated WebP Keyframe Aggregated Breakdown:")
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
