"""
Example 03: High-Throughput Batch Image Classification
-------------------------------------------------------
Demonstrates how to process multiple images in parallel or in a single batch,
measuring CPU inference latency and throughput.
"""

import time
from pathlib import Path
import numpy as np
from PIL import Image
import nsfwpy

def main():
    print("=" * 60)
    print(" NSFWPY - Example 03: Batch Processing Benchmark")
    print("=" * 60)

    # 1. Load Model
    model = nsfwpy.load_model()

    # 2. Create batch of synthetic test images
    num_images = 10
    print(f"\nGenerating {num_images} test images for batch classification...")
    batch_images = [
        Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        for _ in range(num_images)
    ]

    # 3. Perform Batch Classification
    print(f"Running batch inference on {num_images} images...")
    start_time = time.perf_counter()

    batch_results = model.classify_batch(batch_images, top_k=3)

    total_latency_ms = (time.perf_counter() - start_time) * 1000
    avg_per_img_ms = total_latency_ms / num_images

    # 4. Display Results Summary
    print("\nBatch Summary:")
    print(f"  • Total Images Processed : {num_images}")
    print(f"  • Total Batch Time       : {total_latency_ms:.2f} ms")
    print(f"  • Average Latency/Image  : {avg_per_img_ms:.2f} ms")
    print(f"  • Throughput             : {(1000 / avg_per_img_ms):.1f} images/sec")

    print("\nSample Batch Results (First 3 Images):")
    print("-" * 50)
    for idx in range(min(3, len(batch_results))):
        top_pred = batch_results[idx][0]
        print(f"Image {idx+1}: Top Category = {top_pred['className']:<8} (Prob: {top_pred['probability']:.4f})")
    print("-" * 50)

if __name__ == "__main__":
    main()
