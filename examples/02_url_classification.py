"""
Example 02: URL Image Classification
------------------------------------
Demonstrates how to classify an image directly from an HTTP/HTTPS web URL.
"""

import nsfwpy

def main():
    print("=" * 60)
    print(" NSFWPY - Example 02: Remote Image URL Classification")
    print("=" * 60)

    # 1. Load Model
    model = nsfwpy.load_model("mobilenet_v2")

    # 2. Remote image URL
    test_url = "https://picsum.photos/400/300"
    print(f"\nFetching and classifying URL:\n{test_url}\n")

    try:
        results = model.classify(test_url, top_k=5)

        print("Classification Breakdown:")
        print("-" * 45)
        for res in results:
            cat = res["className"]
            prob = res["probability"]
            print(f"  • {cat:<10}: {prob * 100:>6.2f}%")
        print("-" * 45)

        # Determine top classification
        top_category = results[0]["className"]
        confidence = results[0]["probability"] * 100
        print(f"\nPrimary Prediction: {top_category} ({confidence:.2f}% confidence)")

    except Exception as e:
        print(f"Error fetching/classifying image URL: {e}")

if __name__ == "__main__":
    main()
