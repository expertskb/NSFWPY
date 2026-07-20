from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nsfwpy-onnx",
    version="1.1.2",
    description="Python 3.14 CPU-optimized ultra-lightweight NSFWPY image classification library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="expertskb",
    url="https://github.com/expertskb/nsfwpy-onnx",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "onnxruntime>=1.16.0",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "nsfwpy=nsfwpy.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
