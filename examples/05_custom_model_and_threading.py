"""
Example 05: Custom Backbones & ONNX CPU Thread Optimization
------------------------------------------------------------
Demonstrates how to select different model backbones (MobileNetV2, MobileNetV3, InceptionV3),
and configure ONNX Runtime CPU multi-threading options.
"""

import time
import numpy as np
from PIL import Image
import nsfwpy

def main():
    print("=" * 60)
    print(" NSFWPY - Example 05: Model Backbones & Thread Optimization")
    print("=" * 60)

    # Synthetic test image
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

    backbones = ["mobilenet_v2", "mobilenet_v3", "inception_v3"]

    print("\nComparing Inference Speed Across Backbones:")
    print("-" * 55)
    print(f"{'Backbone':<15} | {'Load Time (ms)':<15} | {'Inference (ms)':<15}")
    print("-" * 55)

    for backbone in backbones:
        t0 = time.perf_counter()
        # Load with 4 CPU threads explicitly
        model = nsfwpy.load_model(backbone, num_threads=4)
        load_time = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        _ = model.classify(img, top_k=1)
        inf_time = (time.perf_counter() - t1) * 1000

        print(f"{backbone:<15} | {load_time:>12.2f} ms | {inf_time:>12.2f} ms")

    print("-" * 55)

if __name__ == "__main__":
    main()
