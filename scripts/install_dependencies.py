"""
Dependency installation script for Deepfake Detection project.

This script is provided as a convenience utility, especially for
Google Colab or fresh environments. Standard installations should
prefer using requirements.txt.
"""

import subprocess
import sys


def install(package):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package]
    )


if __name__ == "__main__":
    packages = [
        "torch",
        "torchvision",
        "timm",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "numpy",
        "tqdm",
        "seaborn",
        "Pillow",
        "opencv-python-headless",
        "insightface==0.7.3",
        "onnxruntime-gpu"
    ]

    for pkg in packages:
        print(f"Installing {pkg}...")
        install(pkg)

    print("✅ All dependencies installed successfully.")

